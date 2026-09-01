#pragma once

#include "llama.h"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#define LLAMA_NGRAM_MIN    1
#define LLAMA_NGRAM_MAX    4
#define LLAMA_NGRAM_STATIC 2

// Data structures to map n-grams to empirical token probabilities:

struct common_ngram {
    llama_token tokens[LLAMA_NGRAM_MAX];

    common_ngram() {
        for (int i = 0; i < LLAMA_NGRAM_MAX; ++i) {
            tokens[i] = LLAMA_TOKEN_NULL;
        }
    }

    common_ngram(const llama_token * input, const int ngram_size) {
        for (int i = 0; i < LLAMA_NGRAM_MAX; ++i) {
            tokens[i] = i < ngram_size ? input[i] : LLAMA_TOKEN_NULL;
        }
    }

    bool operator==(const common_ngram & other) const {
        for (int i = 0; i < LLAMA_NGRAM_MAX; ++i) {
            if (tokens[i] != other.tokens[i]) {
                return false;
            }
        }
        return true;
    }
};

struct common_token_hash_function {
    size_t operator()(const llama_token token) const {
        // see https://probablydance.com/2018/06/16/fibonacci-hashing-the-optimization-that-the-world-forgot-or-a-better-alternative-to-integer-modulo/
        return token * 11400714819323198485llu;
    }
};

struct common_ngram_hash_function {
    size_t operator()(const common_ngram & ngram) const {
        size_t hash = common_token_hash_function{}(ngram.tokens[0]);
        for (int i = 1; i < LLAMA_NGRAM_MAX; ++i) {
            hash ^= common_token_hash_function{}(ngram.tokens[i]);
        }
        return hash;
    }
};

// token -> number of times token has been seen
typedef std::unordered_map<llama_token, int32_t> common_ngram_cache_part;

// n-gram -> empirical distribution of following tokens
typedef std::unordered_map<common_ngram, common_ngram_cache_part, common_ngram_hash_function> common_ngram_cache;

// Hash used by the indexed (memory-mappable) on-disk format.
// Unlike common_ngram_hash_function this mixes the token position into the hash, which is a
// requirement for the open addressing scheme of the on-disk table. It must never change without
// bumping LLAMA_NGRAM_CACHE_INDEX_VERSION, since it is baked into the bucket layout of the file.
struct common_ngram_hash_indexed_function {
    uint64_t operator()(const common_ngram & ngram) const {
        uint64_t hash = 0xcbf29ce484222325ull; // FNV-1a offset basis
        for (int i = 0; i < LLAMA_NGRAM_MAX; ++i) {
            hash ^= (uint64_t) (uint32_t) ngram.tokens[i];
            hash *= 0x100000001b3ull; // FNV-1a prime
            hash ^= hash >> 29;
        }
        // final avalanche (splitmix64 finalizer)
        hash ^= hash >> 30;
        hash *= 0xbf58476d1ce4e5b9ull;
        hash ^= hash >> 27;
        hash *= 0x94d049bb133111ebull;
        hash ^= hash >> 31;
        return hash;
    }
};

// The token counts of a single n-gram, as returned by a lookup in a static cache.
// The number of distinct continuations of an n-gram is small, so a flat array is both more compact
// and faster to search than a hash map - and it can be pointed straight at memory-mapped data.
struct common_ngram_cache_counts {
    struct token_count {
        llama_token token;
        int32_t     count;
    };

    // counts read from a memory-mapped file are referenced in place, counts read from a RAM
    // resident cache are collected in buf - exactly one of the two is used
    const token_count *      mapped = nullptr;
    int32_t                  n      = 0;
    std::vector<token_count> buf;

    const token_count * data() const { return mapped ? mapped : buf.data(); }

    bool empty() const { return n == 0; }

    const token_count * begin() const { return data(); }
    const token_count * end()   const { return data() + n; }

    // number of times token was observed after the n-gram, 0 if it never was
    int32_t find(const llama_token token) const {
        const token_count * tc = data();
        for (int32_t i = 0; i < n; ++i) {
            if (tc[i].token == token) {
                return tc[i].count;
            }
        }
        return 0;
    }
};

// Whether to memory-map a static ngram cache instead of reading it into RAM.
enum common_ngram_cache_mmap_mode {
    COMMON_NGRAM_CACHE_MMAP_AUTO = 0, // memory-map the file if it is in the indexed format
    COMMON_NGRAM_CACHE_MMAP_ON,       // require the indexed format and memory-map it
    COMMON_NGRAM_CACHE_MMAP_OFF,      // always read the whole cache into RAM
};

// On-disk format of a lookup cache file.
enum common_ngram_cache_format {
    COMMON_NGRAM_CACHE_FORMAT_LEGACY = 0, // flat record stream, has to be read into RAM in full
    COMMON_NGRAM_CACHE_FORMAT_INDEXED,    // hash-indexed, can be queried in place from a mapping
};

#define LLAMA_NGRAM_CACHE_INDEX_MAGIC   0x43474e4c // "LNGC"
#define LLAMA_NGRAM_CACHE_INDEX_VERSION 1

struct common_ngram_cache_mmap;

// A static (read-only) ngram cache, backed either by RAM or by a memory-mapped indexed file.
// Copies share the same mapping, so it is cheap to hand out one per sequence.
struct common_ngram_cache_static {
    common_ngram_cache                      ram;    // used when the cache is RAM resident
    std::shared_ptr<common_ngram_cache_mmap> mapped; // used when the cache is memory-mapped

    common_ngram_cache_static() = default;
    common_ngram_cache_static(common_ngram_cache cache) : ram(std::move(cache)) {}

    bool empty() const;

    // the counts of ngram, empty if the cache does not contain it
    common_ngram_cache_counts get(const common_ngram & ngram) const;

    // whether lookups are served from a memory mapping instead of RAM
    bool is_mapped() const {
        return mapped != nullptr;
    }
};

// Update an ngram cache with tokens.
// ngram_cache:         the cache to modify.
// ngram_min/ngram_max: the min/max size of the ngrams to extract from inp_data.
// inp_data:            the token sequence with which to update ngram_cache.
// nnew:                how many new tokens have been appended to inp_data since the last call to this function.
// print_progress:      whether to print progress to stderr.
//
// In order to get correct results inp_data can ONLY BE APPENDED TO.
// Changes in the middle need a complete rebuild.
void common_ngram_cache_update(
    common_ngram_cache & ngram_cache, int ngram_min, int ngram_max, std::vector<llama_token> & inp_data, int nnew, bool print_progress);

// Try to draft tokens from ngram caches.
// inp:                the tokens generated so far.
// draft:              the token sequence to draft. Expected to initially contain the previously sampled token.
// n_draft:            maximum number of tokens to add to draft.
// ngram_min/gram_max: the min/max size of the ngrams in nc_context and nc_dynamic.
// nc_context:         ngram cache based on current context.
// nc_dynamic:         ngram cache based on previous user generations.
// nc_static:          ngram cache generated from a large text corpus, used for validation.
void common_ngram_cache_draft(
    std::vector<llama_token> & inp, std::vector<llama_token> & draft, int n_draft, int ngram_min, int ngram_max,
    common_ngram_cache & nc_context, common_ngram_cache & nc_dynamic, const common_ngram_cache_static & nc_static);

// Save an ngram cache to a file.
// ngram_cache: the ngram cache to save.
// filename:    the path under which to save the ngram cache.
void common_ngram_cache_save(common_ngram_cache & ngram_cache, const std::string & filename);

// Save an ngram cache to a file in the indexed format, which can be queried directly from a
// memory mapping without reading it into RAM. Use this for large static caches that are meant to
// live on an SSD or on persistent memory (e.g. Intel Optane).
// ngram_cache: the ngram cache to save.
// filename:    the path under which to save the ngram cache.
void common_ngram_cache_save_indexed(common_ngram_cache & ngram_cache, const std::string & filename);

// Determine the on-disk format of a lookup cache file.
// filename: the path of the ngram cache file.
// returns:  the detected format. Throws if the file cannot be read.
enum common_ngram_cache_format common_ngram_cache_get_format(const std::string & filename);

// Load an ngram cache saved with common_ngram_cache_save or common_ngram_cache_save_indexed.
// The whole cache is read into RAM, regardless of the on-disk format.
// filename: the path from which to load the ngram cache.
// returns:  an ngram cache containing the information saved to filename.
common_ngram_cache common_ngram_cache_load(const std::string & filename);

// Open a static ngram cache for lookups.
// A cache in the indexed format is memory-mapped and queried in place, so that its pages are
// served from the file system (page cache, SSD or persistent memory) instead of occupying RAM.
// filename: the path from which to open the ngram cache.
// mode:     whether to memory-map the file, see common_ngram_cache_mmap_mode.
// prefetch: hint the OS to read the mapping ahead of time. Leave disabled for SSDs and persistent
//           memory, where the point of mapping the file is to *not* pull it into RAM.
// returns:  a handle that serves lookups either from RAM or from the mapping.
common_ngram_cache_static common_ngram_cache_open_static(
    const std::string & filename, enum common_ngram_cache_mmap_mode mode, bool prefetch);

// Merge two ngram caches.
// ngram_cache_target: the ngram cache to which to add the information from ngram_cache_add.
// ngram_cache_add:    the ngram cache to add to ngram_cache_target.
void common_ngram_cache_merge(common_ngram_cache & ngram_cache_target, common_ngram_cache & ngram_cache_add);
