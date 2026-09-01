#include "ggml.h"
#include "llama.h"
#include "common.h"
#include "ngram-cache.h"

#include <clocale>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <string>
#include <vector>

static void print_usage(char * argv0) {
    fprintf(stderr, "Converts a lookup cache into the indexed format, which can be queried directly\n");
    fprintf(stderr, "from a memory mapping instead of being read into RAM. Use this for large static\n");
    fprintf(stderr, "caches that are meant to stay on an SSD or on persistent memory.\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Usage: %s [--help] [--to-legacy] lookup_in.bin lookup_out.bin\n", argv0);
    fprintf(stderr, "       %s --info lookup.bin\n", argv0);
    fprintf(stderr, "\n");
    fprintf(stderr, "  --to-legacy  write the legacy (RAM only) format instead of the indexed one\n");
    fprintf(stderr, "  --info       print the format of a lookup cache and exit\n");
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    bool to_legacy = false;
    bool info_only = false;

    std::vector<std::string> files;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        }
        if (arg == "--to-legacy") {
            to_legacy = true;
            continue;
        }
        if (arg == "--info") {
            info_only = true;
            continue;
        }
        files.push_back(arg);
    }

    if (files.size() != (info_only ? 1u : 2u)) {
        print_usage(argv[0]);
        return 1;
    }

    try {
        const enum common_ngram_cache_format format = common_ngram_cache_get_format(files[0]);
        const char * format_name = format == COMMON_NGRAM_CACHE_FORMAT_INDEXED ? "indexed" : "legacy";

        if (info_only) {
            fprintf(stderr, "%s: %s is in the %s format\n", __func__, files[0].c_str(), format_name);

            const common_ngram_cache ngram_cache = common_ngram_cache_load(files[0]);

            size_t n_counts = 0;
            for (const auto & item : ngram_cache) {
                n_counts += item.second.size();
            }

            fprintf(stderr, "%s: %zu n-grams, %zu token counts\n", __func__, ngram_cache.size(), n_counts);
            return 0;
        }

        fprintf(stderr, "%s: loading %s (%s format)\n", __func__, files[0].c_str(), format_name);
        common_ngram_cache ngram_cache = common_ngram_cache_load(files[0]);

        fprintf(stderr, "%s: writing %zu n-grams to %s (%s format)\n",
                __func__, ngram_cache.size(), files[1].c_str(), to_legacy ? "legacy" : "indexed");

        if (to_legacy) {
            common_ngram_cache_save(ngram_cache, files[1]);
        } else {
            common_ngram_cache_save_indexed(ngram_cache, files[1]);
        }
    } catch (const std::exception & err) {
        fprintf(stderr, "%s: %s\n", __func__, err.what());
        return 1;
    }

    return 0;
}
