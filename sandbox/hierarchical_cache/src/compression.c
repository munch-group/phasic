#include <markov_cache/internal.h>
#include <stdlib.h>

size_t compress_data(const uint8_t *data, size_t size,
                    ZSTD_CDict *dict,
                    uint8_t **out_compressed) {
    if (!data || size == 0 || !out_compressed) return 0;
    
    size_t max_compressed = ZSTD_compressBound(size);
    *out_compressed = malloc(max_compressed);
    if (!*out_compressed) return 0;
    
    ZSTD_CCtx *cctx = ZSTD_createCCtx();
    if (!cctx) {
        free(*out_compressed);
        *out_compressed = NULL;
        return 0;
    }
    
    size_t compressed_size;
    
    if (dict) {
        compressed_size = ZSTD_compress_usingCDict(
            cctx, *out_compressed, max_compressed,
            data, size, dict);
    } else {
        compressed_size = ZSTD_compressCCtx(
            cctx, *out_compressed, max_compressed,
            data, size, 3);  // Compression level 3
    }
    
    ZSTD_freeCCtx(cctx);
    
    if (ZSTD_isError(compressed_size)) {
        free(*out_compressed);
        *out_compressed = NULL;
        return 0;
    }
    
    // Shrink to actual size
    *out_compressed = realloc(*out_compressed, compressed_size);
    
    return compressed_size;
}

size_t decompress_data(const uint8_t *compressed, size_t compressed_size,
                      ZSTD_DDict *dict,
                      uint8_t **out_data) {
    if (!compressed || compressed_size == 0 || !out_data) return 0;
    
    // Get decompressed size
    unsigned long long uncompressed_size = ZSTD_getFrameContentSize(
        compressed, compressed_size);
    
    if (uncompressed_size == ZSTD_CONTENTSIZE_ERROR ||
        uncompressed_size == ZSTD_CONTENTSIZE_UNKNOWN) {
        return 0;
    }
    
    *out_data = malloc(uncompressed_size);
    if (!*out_data) return 0;
    
    ZSTD_DCtx *dctx = ZSTD_createDCtx();
    if (!dctx) {
        free(*out_data);
        *out_data = NULL;
        return 0;
    }
    
    size_t result;
    
    if (dict) {
        result = ZSTD_decompress_usingDDict(
            dctx, *out_data, uncompressed_size,
            compressed, compressed_size, dict);
    } else {
        result = ZSTD_decompressDCtx(
            dctx, *out_data, uncompressed_size,
            compressed, compressed_size);
    }
    
    ZSTD_freeDCtx(dctx);
    
    if (ZSTD_isError(result)) {
        free(*out_data);
        *out_data = NULL;
        return 0;
    }
    
    return result;
}
