# llama.cpp/examples/lookup

Demonstration of Prompt Lookup Decoding

https://github.com/apoorvumang/prompt-lookup-decoding

The key parameters for lookup decoding are `ngram_min`, `ngram_max` and `n_draft`. The first two determine the size of the ngrams to search for in the prompt for a match. The latter specifies how many subsequent tokens to draft if a match is found.

More info:

https://github.com/ggml-org/llama.cpp/pull/4484
https://github.com/ggml-org/llama.cpp/issues/4226

## Keeping a large static cache out of RAM

A static lookup cache built from a large corpus can be hundreds of megabytes or more, and it used to
be read into RAM in full at startup - once per sequence, in the case of the server. `llama-lookup-index`
converts a cache into an *indexed* format that can be queried directly from a memory mapping, so its
pages are served by the file system (page cache, SSD, or persistent memory such as Intel Optane)
instead of occupying RAM:

```bash
# one-off conversion of an existing cache
llama-lookup-index lookup-cache.bin /mnt/optane/lookup-cache-indexed.bin

# inspect a cache
llama-lookup-index --info /mnt/optane/lookup-cache-indexed.bin
```

The indexed cache is used like any other static cache. `--lookup-cache-mmap` controls how it is
accessed:

```bash
llama-server -m model.gguf --spec-type ngram-cache \
    -lcs /mnt/optane/lookup-cache-indexed.bin -lcm on
```

* `auto` (default) - memory-map the cache if it is in the indexed format, otherwise read it into RAM.
  Legacy caches keep working unchanged.
* `on` - require the indexed format and memory-map it. Fails with an error if the cache still is in
  the legacy format, so that a cache is never silently pulled into RAM.
* `off` - always read the whole cache into RAM, whatever its format.

`--lookup-cache-prefetch` asks the OS to read the whole mapping ahead of time. Leave it disabled for
caches on an SSD or on persistent memory - the point of mapping the file there is to *not* pull it
into RAM.

A lookup costs one hash table probe (plus any probes needed to resolve a collision) and one read of
the matching token counts, so a mapped cache adds one or two storage accesses per drafted token.
That is essentially free on persistent memory and on a page cache hit, and it is the reason the
mapping is advised as random access rather than read ahead. The trade-off is size: the indexed file
is roughly 2-3x larger on disk than the legacy one, because it stores an explicit hash table.

Measured on a 2M n-gram cache (159 MiB indexed / 69 MiB legacy):

| | resident memory after opening | after 200 lookups |
|---|---|---|
| `-lcm off` (RAM) | 383 MiB | 383 MiB |
| `-lcm on` (mapped) | 2.6 MiB | 127 MiB of page cache, reclaimable |

The indexed format stores its fields in native byte order, like the legacy format, so cache files
are not portable between machines of different endianness.
