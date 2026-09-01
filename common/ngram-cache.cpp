#include "ngram-cache.h"
#include "common.h"
#include "log.h"

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <thread>
#include <algorithm>

#ifdef _WIN32
#   ifndef NOMINMAX
#       define NOMINMAX
#   endif
#   include <windows.h>
#else
#   include <fcntl.h>
#   include <sys/mman.h>
#   include <sys/stat.h>
#   include <unistd.h>
#endif

// The indexed on-disk format: a header, an open addressing hash table of fixed size buckets and a
// flat array of (token, count) pairs. A lookup touches one bucket (plus the probes needed to
// resolve a collision) and one run of count pairs, so the file can be queried directly from a
// memory mapping - the pages come from the page cache, an SSD or persistent memory instead of the
// cache being read into RAM in full.
//
// All fields are stored in native byte order and with native alignment, like the legacy format.
// Cache files are therefore not portable between machines of different endianness.

struct common_ngram_cache_index_header {
    uint32_t magic;       // LLAMA_NGRAM_CACHE_INDEX_MAGIC
    uint32_t version;     // LLAMA_NGRAM_CACHE_INDEX_VERSION
    uint32_t ngram_max;   // LLAMA_NGRAM_MAX the file was written with
    uint32_t bucket_size; // sizeof(common_ngram_cache_index_bucket)
    uint64_t n_buckets;   // size of the hash table, always a power of two
    uint64_t n_entries;   // number of occupied buckets
    uint64_t n_counts;    // number of (token, count) pairs
    uint64_t off_buckets; // byte offset of the hash table
    uint64_t off_counts;  // byte offset of the (token, count) pairs
    uint64_t reserved;    // 0
};

struct common_ngram_cache_index_bucket {
    llama_token tokens[LLAMA_NGRAM_MAX];
    uint32_t    n_counts;   // number of (token, count) pairs, 0 marks an empty bucket
    uint32_t    padding;
    uint64_t    off_counts; // index of the first (token, count) pair of this n-gram
};

typedef common_ngram_cache_counts::token_count common_ngram_cache_token_count;

static_assert(sizeof(common_ngram_cache_index_header) == 64, "unexpected ngram cache header size");
static_assert(sizeof(common_ngram_cache_index_bucket) == 4*LLAMA_NGRAM_MAX + 16, "unexpected ngram cache bucket size");
static_assert(sizeof(common_ngram_cache_token_count)  == 8,  "unexpected ngram cache token count size");

// the hash table is kept below this load factor so that a lookup always terminates on an empty bucket
static constexpr uint64_t NGRAM_CACHE_INDEX_LOAD_NUM = 7;
static constexpr uint64_t NGRAM_CACHE_INDEX_LOAD_DEN = 10;

struct common_ngram_cache_mmap {
    common_ngram_cache_index_header hdr = {};

    const common_ngram_cache_index_bucket * buckets = nullptr;
    const common_ngram_cache_token_count  * counts  = nullptr;

    uint64_t mask = 0; // n_buckets - 1

    void * addr = nullptr;
    size_t size = 0;

#ifdef _WIN32
    HANDLE hFile    = INVALID_HANDLE_VALUE;
    HANDLE hMapping = NULL;
#else
    int fd = -1;
#endif

    common_ngram_cache_mmap(const std::string & filename, bool prefetch);
    ~common_ngram_cache_mmap();

    void unmap();

    common_ngram_cache_mmap(const common_ngram_cache_mmap &) = delete;
    common_ngram_cache_mmap & operator=(const common_ngram_cache_mmap &) = delete;

    common_ngram_cache_counts lookup(const common_ngram & ngram) const;
};

common_ngram_cache_mmap::common_ngram_cache_mmap(const std::string & filename, bool prefetch) {
#ifdef _WIN32
    hFile = CreateFileA(filename.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING,
                        FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS, NULL);
    if (hFile == INVALID_HANDLE_VALUE) {
        throw std::runtime_error("unable to open ngram cache " + filename);
    }

    LARGE_INTEGER file_size;
    if (!GetFileSizeEx(hFile, &file_size)) {
        CloseHandle(hFile);
        hFile = INVALID_HANDLE_VALUE;
        throw std::runtime_error("unable to determine the size of ngram cache " + filename);
    }
    size = (size_t) file_size.QuadPart;

    if (size >= sizeof(common_ngram_cache_index_header)) {
        hMapping = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
        if (hMapping != NULL) {
            addr = MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
        }
    }
    if (addr == nullptr) {
        if (hMapping != NULL) {
            CloseHandle(hMapping);
            hMapping = NULL;
        }
        CloseHandle(hFile);
        hFile = INVALID_HANDLE_VALUE;
        throw std::runtime_error("unable to memory-map ngram cache " + filename);
    }

    if (prefetch) {
        // best effort - PrefetchVirtualMemory is not available on all supported Windows versions
#   if _WIN32_WINNT >= 0x602
        BOOL (WINAPI *pPrefetchVirtualMemory) (HANDLE, ULONG_PTR, PWIN32_MEMORY_RANGE_ENTRY, ULONG);
        HMODULE hKernel32 = GetModuleHandleW(L"kernel32.dll");

        pPrefetchVirtualMemory = (decltype(pPrefetchVirtualMemory))(void *) GetProcAddress(hKernel32, "PrefetchVirtualMemory");

        if (pPrefetchVirtualMemory) {
            WIN32_MEMORY_RANGE_ENTRY range;
            range.VirtualAddress = addr;
            range.NumberOfBytes  = (SIZE_T) size;
            pPrefetchVirtualMemory(GetCurrentProcess(), 1, &range, 0);
        }
#   endif
    }
#else
    fd = open(filename.c_str(), O_RDONLY);
    if (fd < 0) {
        throw std::runtime_error("unable to open ngram cache " + filename);
    }

    struct stat sb;
    if (fstat(fd, &sb) != 0) {
        close(fd);
        fd = -1;
        throw std::runtime_error("unable to determine the size of ngram cache " + filename);
    }
    size = (size_t) sb.st_size;

    if (size >= sizeof(common_ngram_cache_index_header)) {
        addr = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (addr == MAP_FAILED) {
            addr = nullptr;
        }
    }
    if (addr == nullptr) {
        close(fd);
        fd = -1;
        throw std::runtime_error("unable to memory-map ngram cache " + filename);
    }

    // lookups jump around the table, and the whole point of mapping the file is to leave it on the
    // storage device, so tell the kernel not to read ahead unless a prefetch was requested
#   ifdef MADV_RANDOM
    madvise(addr, size, prefetch ? MADV_WILLNEED : MADV_RANDOM);
#   endif
#endif

    memcpy(&hdr, addr, sizeof(hdr));

    // the destructor does not run when a constructor throws, so the mapping has to be released here
    const auto fail = [&](const std::string & reason) {
        unmap();
        throw std::runtime_error("ngram cache " + filename + ": " + reason);
    };

    if (hdr.magic != LLAMA_NGRAM_CACHE_INDEX_MAGIC) {
        fail("not in the indexed format");
    }
    if (hdr.version != LLAMA_NGRAM_CACHE_INDEX_VERSION) {
        fail("unsupported index version " + std::to_string(hdr.version) +
             ", expected " + std::to_string(LLAMA_NGRAM_CACHE_INDEX_VERSION));
    }
    if (hdr.ngram_max != LLAMA_NGRAM_MAX || hdr.bucket_size != sizeof(common_ngram_cache_index_bucket)) {
        fail("was written by an incompatible build (ngram_max=" + std::to_string(hdr.ngram_max) + ")");
    }
    if (hdr.n_buckets == 0 || (hdr.n_buckets & (hdr.n_buckets - 1)) != 0) {
        fail("bucket count is not a power of two");
    }
    if (hdr.n_entries >= hdr.n_buckets) {
        fail("hash table has no empty bucket");
    }

    // the offsets are read from the file, so every size computation below has to be overflow safe
    const uint64_t max_buckets = (UINT64_MAX - hdr.off_buckets) / sizeof(common_ngram_cache_index_bucket);
    const uint64_t max_counts  = (UINT64_MAX - hdr.off_counts)  / sizeof(common_ngram_cache_token_count);
    if (hdr.n_buckets > max_buckets || hdr.n_counts > max_counts) {
        fail("index offsets are out of range");
    }
    if (hdr.off_buckets + hdr.n_buckets*sizeof(common_ngram_cache_index_bucket) > size ||
        hdr.off_counts  + hdr.n_counts *sizeof(common_ngram_cache_token_count)  > size) {
        fail("file is truncated");
    }
    if (hdr.off_buckets % alignof(common_ngram_cache_index_bucket) != 0 ||
        hdr.off_counts  % alignof(common_ngram_cache_token_count)  != 0) {
        fail("index offsets are misaligned");
    }

    buckets = (const common_ngram_cache_index_bucket *) ((const char *) addr + hdr.off_buckets);
    counts  = (const common_ngram_cache_token_count  *) ((const char *) addr + hdr.off_counts);
    mask    = hdr.n_buckets - 1;
}

void common_ngram_cache_mmap::unmap() {
#ifdef _WIN32
    if (addr != nullptr) {
        UnmapViewOfFile(addr);
    }
    if (hMapping != NULL) {
        CloseHandle(hMapping);
        hMapping = NULL;
    }
    if (hFile != INVALID_HANDLE_VALUE) {
        CloseHandle(hFile);
        hFile = INVALID_HANDLE_VALUE;
    }
#else
    if (addr != nullptr) {
        munmap(addr, size);
    }
    if (fd >= 0) {
        close(fd);
        fd = -1;
    }
#endif
    addr = nullptr;
}

common_ngram_cache_mmap::~common_ngram_cache_mmap() {
    unmap();
}

common_ngram_cache_counts common_ngram_cache_mmap::lookup(const common_ngram & ngram) const {
    common_ngram_cache_counts result;

    uint64_t idx = common_ngram_hash_indexed_function{}(ngram) & mask;

    // the table is never full, so this loop always hits an empty bucket eventually - the bound is
    // only there to keep a corrupted file from spinning forever
    for (uint64_t probe = 0; probe <= mask; ++probe) {
        const common_ngram_cache_index_bucket & bucket = buckets[idx];

        if (bucket.n_counts == 0) {
            return result;
        }
        if (memcmp(bucket.tokens, ngram.tokens, sizeof(ngram.tokens)) == 0) {
            if (bucket.off_counts + bucket.n_counts > hdr.n_counts) {
                LOG_ERR("%s: ngram cache bucket points outside of the count array, ignoring it\n", __func__);
                return result;
            }
            result.mapped = counts + bucket.off_counts;
            result.n      = (int32_t) bucket.n_counts;
            return result;
        }

        idx = (idx + 1) & mask;
    }

    return result;
}

bool common_ngram_cache_static::empty() const {
    return mapped ? mapped->hdr.n_entries == 0 : ram.empty();
}

common_ngram_cache_counts common_ngram_cache_static::get(const common_ngram & ngram) const {
    if (mapped) {
        return mapped->lookup(ngram);
    }

    common_ngram_cache_counts result;

    const common_ngram_cache::const_iterator part_it = ram.find(ngram);
    if (part_it == ram.end()) {
        return result;
    }

    result.buf.reserve(part_it->second.size());
    for (const std::pair<const llama_token, int32_t> & token_count : part_it->second) {
        result.buf.push_back({token_count.first, token_count.second});
    }
    result.n = (int32_t) result.buf.size();

    return result;
}

void common_ngram_cache_update(common_ngram_cache & ngram_cache, int ngram_min, int ngram_max,
                              std::vector<llama_token> & inp, int nnew, bool print_progress) {
    const int64_t t_start_ms = ggml_time_ms();
    const int64_t inp_size = inp.size();

    const int64_t n_todo = inp_size * (ngram_max - ngram_min + 1);
    int64_t n_done = 0;

    for (int64_t ngram_size = ngram_min; ngram_size <= ngram_max; ++ngram_size) {
        const int64_t i_start = std::max(inp_size - nnew, ngram_size);
        for (int64_t i = i_start; i < inp_size; ++i) {
            const int64_t ngram_start = i - ngram_size;
            common_ngram ngram(&inp[ngram_start], ngram_size);
            const llama_token token = inp[i];

            common_ngram_cache::iterator part_it = ngram_cache.find(ngram);
            if (part_it == ngram_cache.end()) {
                common_ngram_cache_part part;
                part.emplace(token, 1);
                ngram_cache.emplace(ngram, part);
            } else {
                common_ngram_cache_part::iterator token_count_it = part_it->second.find(token);
                if (token_count_it == part_it->second.end()) {
                    part_it->second.emplace(token, 1);
                } else {
                    token_count_it->second++;
                }
            }
            ++n_done;

            if (print_progress && n_done % 10000000 == 0) {
                const int64_t t_now_ms = ggml_time_ms();
                const int64_t eta_ms   = (inp_size*(ngram_max-ngram_min+1) - n_done) * (t_now_ms - t_start_ms) / n_done;
                const int64_t eta_min  = eta_ms / (60*1000);
                const int64_t eta_s    = (eta_ms - 60*1000*eta_min) / 1000;

                fprintf(stderr, "%s: %" PRId64 "/%" PRId64 " done, ETA: %02" PRId64 ":%02" PRId64 "\n", __func__, n_done, n_todo, eta_min, eta_s);
            }
        }
    }
}

// Helper function to get a token from the combined, speculative sequence of inp and draft.
static llama_token get_token(const std::vector<llama_token> & inp, const std::vector<llama_token> & draft, const size_t i) {
    return i < inp.size() ? inp[i] : draft[1 + i - inp.size()];
}

// If sample size or percentage are below these thresholds the draft is aborted early:
constexpr int    draft_min_sample_size_lax[LLAMA_NGRAM_MAX] = { 2,  2,  1,  1};
constexpr int        draft_min_percent_lax[LLAMA_NGRAM_MAX] = {66, 50, 50, 50};
constexpr int draft_min_sample_size_strict[LLAMA_NGRAM_MAX] = { 4,  3,  2,  2};
constexpr int     draft_min_percent_strict[LLAMA_NGRAM_MAX] = {75, 66, 66, 66};

// Helper function that tries to draft a token from only the static ngram cache:
static llama_token try_draft(const common_ngram_cache_counts & part_static) {
    if (part_static.empty()) {
        return LLAMA_TOKEN_NULL;
    }

    int max_count_static  = 0;
    int sum_count_static  = 0;
    llama_token max_token = LLAMA_TOKEN_NULL;

    for (const common_ngram_cache_counts::token_count & token_count_static : part_static) {
        const llama_token token = token_count_static.token;
        const int32_t count_static  = token_count_static.count;

        if (count_static > max_count_static) {
            max_token        = token;
            max_count_static = count_static;
        }
        sum_count_static += count_static;
    }

    if (sum_count_static < draft_min_sample_size_lax[LLAMA_NGRAM_STATIC-1]) {
        return LLAMA_TOKEN_NULL;
    }
    if (100*max_count_static < draft_min_percent_lax[LLAMA_NGRAM_STATIC-1]*sum_count_static) {
        return LLAMA_TOKEN_NULL;
    }
    return max_token;
}

// Try to draft a token from primary cache (context/dynamic), validate with static cache:
static llama_token try_draft(
    common_ngram_cache & nc_primary, const std::vector<common_ngram> & ngrams_primary, const common_ngram_cache_counts & part_static,
    const int * min_sample_size, const int * min_percent) {

    llama_token drafted_token = LLAMA_TOKEN_NULL;

    for (int i = ngrams_primary.size()-1; i >= 0 && drafted_token == LLAMA_TOKEN_NULL; --i) {
        const common_ngram ngram_primary = ngrams_primary[i];

        common_ngram_cache::iterator part_primary_it = nc_primary.find(ngram_primary);
        if (part_primary_it == nc_primary.end()) {
            continue;
        }
        const common_ngram_cache_part part_primary = part_primary_it->second;

        int max_count_primary = 0;
        int max_count_static  = 0;
        int sum_count_primary = 0;
        llama_token max_token = LLAMA_TOKEN_NULL;

        for (std::pair<llama_token, int> token_count_primary : part_primary) {
            const llama_token token = token_count_primary.first;

            const int32_t count_static_raw = part_static.find(token);

            const int32_t count_primary = token_count_primary.second;
            const int32_t count_static  = count_static_raw > 0 ? 100*count_static_raw : 1;

            if (count_primary*count_static > max_count_primary*max_count_static) {
                max_token         = token;
                max_count_primary = count_primary;
                max_count_static  = count_static;
            }
            sum_count_primary += count_primary;
        }

        if (sum_count_primary < min_sample_size[i]) {
            continue;
        }
        if (100*max_count_primary < min_percent[i]*sum_count_primary) {
            continue;;
        }
        drafted_token = max_token;
    }

    return drafted_token;
}

void common_ngram_cache_draft(
    std::vector<llama_token> & inp, std::vector<llama_token> & draft, int n_draft, int ngram_min, int ngram_max,
    common_ngram_cache & nc_context, common_ngram_cache & nc_dynamic, const common_ngram_cache_static & nc_static
) {
    GGML_ASSERT(draft.size() == 1);
    const int inp_size = inp.size();

    if (inp_size < LLAMA_NGRAM_STATIC) {
        return;
    }

    while ((int) draft.size()-1 < n_draft) {
        llama_token drafted_token = LLAMA_TOKEN_NULL;

        const int ngram_start_static = inp_size-LLAMA_NGRAM_STATIC + draft.size()-1;
        common_ngram ngram_static;
        for (int j = ngram_start_static; j < ngram_start_static + LLAMA_NGRAM_STATIC; ++j) {
            ngram_static.tokens[j-ngram_start_static] = get_token(inp, draft, j);
        }
        const common_ngram_cache_counts part_static = nc_static.get(ngram_static);

        // cd = context + dynamic
        std::vector<common_ngram> ngrams_cd;
        for (int ngram_size_cd = ngram_min; ngram_size_cd <= ngram_max; ++ngram_size_cd) {
            const int ngram_start_cd = inp_size-ngram_size_cd + draft.size()-1;
            common_ngram ngram_cd;
            for (int j = ngram_start_cd; j < ngram_start_cd + ngram_size_cd; ++j) {
                ngram_cd.tokens[j-ngram_start_cd] = get_token(inp, draft, j);
            }
            ngrams_cd.push_back(ngram_cd);
        }
        if (drafted_token == LLAMA_TOKEN_NULL) {
            drafted_token = try_draft(nc_context, ngrams_cd, part_static, draft_min_sample_size_lax, draft_min_percent_lax);
        }
        if (drafted_token == LLAMA_TOKEN_NULL) {
            drafted_token = try_draft(nc_dynamic, ngrams_cd, part_static, draft_min_sample_size_strict, draft_min_percent_strict);
        }
        if (drafted_token == LLAMA_TOKEN_NULL) {
            drafted_token = try_draft(part_static);
        }

        if (drafted_token == LLAMA_TOKEN_NULL) {
            break;
        }

        LOG_DBG(" - draft candidate: token=%d\n", drafted_token);
        draft.push_back(drafted_token);
    }
}

void common_ngram_cache_save(common_ngram_cache & ngram_cache, const std::string & filename) {
    std::ofstream file_out(filename, std::ios::binary);
    for (std::pair<common_ngram, common_ngram_cache_part> item : ngram_cache) {
        const common_ngram      ngram        = item.first;
        common_ngram_cache_part token_counts = item.second;
        GGML_ASSERT(!token_counts.empty());
        const int32_t ntokens = token_counts.size();
        GGML_ASSERT(ntokens > 0);

        file_out.write(reinterpret_cast<const char *>(&ngram),   sizeof(common_ngram));
        file_out.write(reinterpret_cast<const char *>(&ntokens), sizeof(int32_t));
        for (std::pair<llama_token, int32_t> item2 : token_counts) {
            const llama_token token = item2.first;
            const int32_t     count = item2.second;
            GGML_ASSERT(count > 0);

            file_out.write(reinterpret_cast<const char *>(&token), sizeof(llama_token));
            file_out.write(reinterpret_cast<const char *>(&count), sizeof(int32_t));
        }
    }
}

void common_ngram_cache_save_indexed(common_ngram_cache & ngram_cache, const std::string & filename) {
    const uint64_t n_entries = ngram_cache.size();

    // size the table so that it stays below the target load factor and always has an empty bucket
    uint64_t n_buckets = 16;
    while (n_buckets*NGRAM_CACHE_INDEX_LOAD_NUM < (n_entries + 1)*NGRAM_CACHE_INDEX_LOAD_DEN) {
        n_buckets *= 2;
    }
    const uint64_t mask = n_buckets - 1;

    std::vector<common_ngram_cache_index_bucket> buckets(n_buckets);
    memset(buckets.data(), 0, buckets.size()*sizeof(common_ngram_cache_index_bucket));

    std::vector<common_ngram_cache_token_count> counts;

    for (const std::pair<const common_ngram, common_ngram_cache_part> & item : ngram_cache) {
        const common_ngram             & ngram        = item.first;
        const common_ngram_cache_part  & token_counts = item.second;

        GGML_ASSERT(!token_counts.empty());

        uint64_t idx = common_ngram_hash_indexed_function{}(ngram) & mask;
        while (buckets[idx].n_counts != 0) {
            idx = (idx + 1) & mask;
        }

        common_ngram_cache_index_bucket & bucket = buckets[idx];
        memcpy(bucket.tokens, ngram.tokens, sizeof(ngram.tokens));
        bucket.off_counts = counts.size();
        bucket.n_counts   = (uint32_t) token_counts.size();

        for (const std::pair<const llama_token, int32_t> & token_count : token_counts) {
            GGML_ASSERT(token_count.second > 0);
            counts.push_back({token_count.first, token_count.second});
        }
    }

    common_ngram_cache_index_header hdr = {};
    hdr.magic       = LLAMA_NGRAM_CACHE_INDEX_MAGIC;
    hdr.version     = LLAMA_NGRAM_CACHE_INDEX_VERSION;
    hdr.ngram_max   = LLAMA_NGRAM_MAX;
    hdr.bucket_size = sizeof(common_ngram_cache_index_bucket);
    hdr.n_buckets   = n_buckets;
    hdr.n_entries   = n_entries;
    hdr.n_counts    = counts.size();
    hdr.off_buckets = sizeof(common_ngram_cache_index_header);
    hdr.off_counts  = hdr.off_buckets + n_buckets*sizeof(common_ngram_cache_index_bucket);

    std::ofstream file_out(filename, std::ios::binary);
    if (!file_out) {
        throw std::runtime_error("unable to write ngram cache " + filename);
    }

    file_out.write(reinterpret_cast<const char *>(&hdr), sizeof(hdr));
    file_out.write(reinterpret_cast<const char *>(buckets.data()), buckets.size()*sizeof(common_ngram_cache_index_bucket));
    if (!counts.empty()) {
        file_out.write(reinterpret_cast<const char *>(counts.data()), counts.size()*sizeof(common_ngram_cache_token_count));
    }

    if (!file_out) {
        throw std::runtime_error("failed to write ngram cache " + filename);
    }
}

enum common_ngram_cache_format common_ngram_cache_get_format(const std::string & filename) {
    std::ifstream file_in(filename, std::ios::binary);
    if (!file_in) {
        throw std::ifstream::failure("Unable to open file " + filename);
    }

    uint32_t magic = 0;
    if (!file_in.read(reinterpret_cast<char *>(&magic), sizeof(magic))) {
        // too short to hold a magic, so it can only be a (possibly empty) legacy cache
        return COMMON_NGRAM_CACHE_FORMAT_LEGACY;
    }

    return magic == LLAMA_NGRAM_CACHE_INDEX_MAGIC ? COMMON_NGRAM_CACHE_FORMAT_INDEXED : COMMON_NGRAM_CACHE_FORMAT_LEGACY;
}

// Read an indexed cache into RAM. Only used when the caller explicitly asks for a RAM resident
// cache - the point of the indexed format is to be queried from the mapping instead.
static common_ngram_cache common_ngram_cache_load_indexed(const std::string & filename) {
    const common_ngram_cache_mmap map(filename, /*prefetch =*/ true);

    common_ngram_cache ngram_cache;
    ngram_cache.reserve(map.hdr.n_entries);

    for (uint64_t i = 0; i < map.hdr.n_buckets; ++i) {
        const common_ngram_cache_index_bucket & bucket = map.buckets[i];
        if (bucket.n_counts == 0) {
            continue;
        }
        if (bucket.off_counts + bucket.n_counts > map.hdr.n_counts) {
            throw std::runtime_error("ngram cache " + filename + ": bucket points outside of the count array");
        }

        common_ngram ngram;
        memcpy(ngram.tokens, bucket.tokens, sizeof(ngram.tokens));

        common_ngram_cache_part token_counts;
        token_counts.reserve(bucket.n_counts);
        for (uint32_t j = 0; j < bucket.n_counts; ++j) {
            const common_ngram_cache_token_count & token_count = map.counts[bucket.off_counts + j];
            GGML_ASSERT(token_count.count > 0);
            token_counts.emplace(token_count.token, token_count.count);
        }

        ngram_cache.emplace(ngram, std::move(token_counts));
    }

    return ngram_cache;
}

static common_ngram_cache common_ngram_cache_load_legacy(const std::string & filename) {
    std::ifstream hashmap_file(filename, std::ios::binary);
    if (!hashmap_file) {
        throw std::ifstream::failure("Unable to open file " + filename);
    }
    common_ngram_cache ngram_cache;

    common_ngram ngram;
    int32_t     ntokens;
    llama_token token;
    int32_t     count;

    char * ngramc   = reinterpret_cast<char*>(&ngram);
    char * ntokensc = reinterpret_cast<char*>(&ntokens);
    char * tokenc   = reinterpret_cast<char*>(&token);
    char * countc   = reinterpret_cast<char*>(&count);
    while(hashmap_file.read(ngramc, sizeof(common_ngram))) {
        GGML_ASSERT(!hashmap_file.eof());
        GGML_ASSERT(hashmap_file.read(ntokensc, sizeof(int32_t)));
        GGML_ASSERT(ntokens > 0);
        common_ngram_cache_part token_counts;

        for (int i = 0; i < ntokens; ++i) {
            GGML_ASSERT(!hashmap_file.eof());
            GGML_ASSERT(hashmap_file.read(tokenc, sizeof(llama_token)));
            GGML_ASSERT(!hashmap_file.eof());
            GGML_ASSERT(hashmap_file.read(countc, sizeof(int32_t)));
            GGML_ASSERT(count > 0);
            token_counts.emplace(token, count);
        }

        ngram_cache.emplace(ngram, token_counts);
    }
    GGML_ASSERT(hashmap_file.eof());

    return ngram_cache;
}

common_ngram_cache common_ngram_cache_load(const std::string & filename) {
    if (common_ngram_cache_get_format(filename) == COMMON_NGRAM_CACHE_FORMAT_INDEXED) {
        return common_ngram_cache_load_indexed(filename);
    }

    return common_ngram_cache_load_legacy(filename);
}

common_ngram_cache_static common_ngram_cache_open_static(
        const std::string & filename, enum common_ngram_cache_mmap_mode mode, bool prefetch) {
    common_ngram_cache_static result;

    const enum common_ngram_cache_format format = common_ngram_cache_get_format(filename);

    if (mode == COMMON_NGRAM_CACHE_MMAP_ON && format != COMMON_NGRAM_CACHE_FORMAT_INDEXED) {
        throw std::runtime_error("ngram cache " + filename + " is in the legacy format and cannot be memory-mapped, "
                                 "convert it with llama-lookup-index or use --lookup-cache-mmap off");
    }

    if (mode != COMMON_NGRAM_CACHE_MMAP_OFF && format == COMMON_NGRAM_CACHE_FORMAT_INDEXED) {
        result.mapped = std::make_shared<common_ngram_cache_mmap>(filename, prefetch);

        LOG_INF("%s: memory-mapped static lookup cache %s (%" PRIu64 " n-grams, %.2f MiB)\n",
                __func__, filename.c_str(), result.mapped->hdr.n_entries, result.mapped->size/(1024.0*1024.0));

        return result;
    }

    result.ram = common_ngram_cache_load(filename);

    LOG_INF("%s: loaded static lookup cache %s into RAM (%zu n-grams)\n", __func__, filename.c_str(), result.ram.size());

    return result;
}

void common_ngram_cache_merge(common_ngram_cache & ngram_cache_target, common_ngram_cache & ngram_cache_add) {
    for (std::pair<common_ngram, common_ngram_cache_part> ngram_part : ngram_cache_add) {
        const common_ngram      ngram = ngram_part.first;
        common_ngram_cache_part  part = ngram_part.second;

        common_ngram_cache::iterator part_merged_it = ngram_cache_target.find(ngram);
        if (part_merged_it == ngram_cache_target.end()) {
            ngram_cache_target.emplace(ngram, part);
            continue;
        }

        for (std::pair<llama_token, int32_t> token_count : part) {
            const llama_token token = token_count.first;
            const int32_t     count = token_count.second;
            GGML_ASSERT(count > 0);

            common_ngram_cache_part::iterator token_count_merged_it = part_merged_it->second.find(token);
            if (token_count_merged_it == part_merged_it->second.end()) {
                part_merged_it->second.emplace(token, count);
                continue;
            }

            token_count_merged_it->second += count;
        }
    }
}
