// Tests for the lookup (n-gram) cache, in particular the indexed on-disk format that is queried
// directly from a memory mapping instead of being read into RAM.

#include "ngram-cache.h"

#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#undef NDEBUG
#include <cassert>

static std::string tmp_path(const char * name) {
    const char * dir = getenv("TMPDIR");
    return std::string(dir ? dir : "/tmp") + "/llama-test-ngram-cache-" + name + ".bin";
}

// a deterministic token sequence with plenty of repetition, so that the cache has both n-grams
// with a single continuation and n-grams with many
static std::vector<llama_token> make_tokens(size_t n) {
    std::mt19937 rng(1234);
    std::vector<llama_token> tokens;
    tokens.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        tokens.push_back((llama_token) (rng() % 97));
    }
    return tokens;
}

static void assert_same_counts(const common_ngram_cache & expected, const common_ngram_cache_static & actual) {
    for (const auto & item : expected) {
        const common_ngram_cache_counts counts = actual.get(item.first);

        assert(!counts.empty());
        assert((size_t) counts.n == item.second.size());

        for (const auto & token_count : item.second) {
            assert(counts.find(token_count.first) == token_count.second);
        }

        // every token reported by the lookup has to be in the reference cache as well
        for (const common_ngram_cache_counts::token_count & tc : counts) {
            const auto it = item.second.find(tc.token);
            assert(it != item.second.end());
            assert(it->second == tc.count);
        }
    }
}

int main() {
    std::vector<llama_token> tokens = make_tokens(20000);

    common_ngram_cache cache;
    common_ngram_cache_update(cache, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX, tokens, tokens.size(), false);
    assert(!cache.empty());

    const std::string path_legacy  = tmp_path("legacy");
    const std::string path_indexed = tmp_path("indexed");

    common_ngram_cache_save(cache, path_legacy);
    common_ngram_cache_save_indexed(cache, path_indexed);

    // the format of each file is detected correctly
    assert(common_ngram_cache_get_format(path_legacy)  == COMMON_NGRAM_CACHE_FORMAT_LEGACY);
    assert(common_ngram_cache_get_format(path_indexed) == COMMON_NGRAM_CACHE_FORMAT_INDEXED);

    // reading the indexed file into RAM gives back the original cache
    {
        const common_ngram_cache loaded = common_ngram_cache_load(path_indexed);
        assert(loaded.size() == cache.size());
        assert_same_counts(cache, common_ngram_cache_static(loaded));
    }

    // the legacy file still round-trips
    {
        const common_ngram_cache loaded = common_ngram_cache_load(path_legacy);
        assert(loaded.size() == cache.size());
        assert_same_counts(cache, common_ngram_cache_static(loaded));
    }

    // lookups served from the mapping agree with the RAM resident cache
    {
        const common_ngram_cache_static mapped =
            common_ngram_cache_open_static(path_indexed, COMMON_NGRAM_CACHE_MMAP_ON, /*prefetch =*/ false);
        assert(mapped.is_mapped());
        assert(!mapped.empty());
        assert_same_counts(cache, mapped);

        // n-grams that are not in the cache have to miss rather than return a wrong bucket
        size_t n_missing = 0;
        std::mt19937 rng(4321);
        for (size_t i = 0; i < 10000; ++i) {
            common_ngram ngram;
            for (int j = 0; j < LLAMA_NGRAM_MAX; ++j) {
                ngram.tokens[j] = (llama_token) (100000 + rng() % 100000);
            }
            if (cache.find(ngram) != cache.end()) {
                continue;
            }
            ++n_missing;
            assert(mapped.get(ngram).empty());
        }
        assert(n_missing > 0);
    }

    // auto mode maps an indexed cache and reads a legacy one into RAM
    {
        const common_ngram_cache_static automatic =
            common_ngram_cache_open_static(path_indexed, COMMON_NGRAM_CACHE_MMAP_AUTO, /*prefetch =*/ false);
        assert(automatic.is_mapped());

        const common_ngram_cache_static legacy =
            common_ngram_cache_open_static(path_legacy, COMMON_NGRAM_CACHE_MMAP_AUTO, /*prefetch =*/ false);
        assert(!legacy.is_mapped());
        assert_same_counts(cache, legacy);
    }

    // an indexed cache can also be forced into RAM
    {
        const common_ngram_cache_static in_ram =
            common_ngram_cache_open_static(path_indexed, COMMON_NGRAM_CACHE_MMAP_OFF, /*prefetch =*/ false);
        assert(!in_ram.is_mapped());
        assert_same_counts(cache, in_ram);
    }

    // a legacy cache cannot be memory-mapped
    {
        bool threw = false;
        try {
            common_ngram_cache_open_static(path_legacy, COMMON_NGRAM_CACHE_MMAP_ON, /*prefetch =*/ false);
        } catch (const std::exception &) {
            threw = true;
        }
        assert(threw);
    }

    // drafting produces the same tokens whether the static cache is mapped or resident
    {
        const common_ngram_cache_static mapped =
            common_ngram_cache_open_static(path_indexed, COMMON_NGRAM_CACHE_MMAP_ON, /*prefetch =*/ false);
        const common_ngram_cache_static in_ram(cache);

        common_ngram_cache empty_context;
        common_ngram_cache empty_dynamic;

        std::vector<llama_token> inp(tokens.begin(), tokens.begin() + 256);

        std::vector<llama_token> draft_mapped = { inp.back() };
        std::vector<llama_token> draft_in_ram = { inp.back() };

        common_ngram_cache_draft(inp, draft_mapped, 8, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX, empty_context, empty_dynamic, mapped);
        common_ngram_cache_draft(inp, draft_in_ram, 8, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX, empty_context, empty_dynamic, in_ram);

        assert(draft_mapped == draft_in_ram);
    }

    // an empty cache maps and misses cleanly
    {
        const std::string path_empty = tmp_path("empty");
        common_ngram_cache empty_cache;
        common_ngram_cache_save_indexed(empty_cache, path_empty);

        const common_ngram_cache_static mapped =
            common_ngram_cache_open_static(path_empty, COMMON_NGRAM_CACHE_MMAP_ON, /*prefetch =*/ false);
        assert(mapped.is_mapped());
        assert(mapped.empty());
        assert(mapped.get(common_ngram(tokens.data(), LLAMA_NGRAM_MAX)).empty());

        remove(path_empty.c_str());
    }

    remove(path_legacy.c_str());
    remove(path_indexed.c_str());

    printf("%s: all tests passed\n", __func__);

    return 0;
}
