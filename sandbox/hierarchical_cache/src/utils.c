#include <markov_cache/internal.h>
#include <string.h>
#include <sys/stat.h>
#include <errno.h>
#include <stdio.h>

int mkdir_recursive(const char *path) {
    char tmp[512];
    char *p = NULL;
    size_t len;
    
    snprintf(tmp, sizeof(tmp), "%s", path);
    len = strlen(tmp);
    
    if (tmp[len - 1] == '/') {
        tmp[len - 1] = 0;
    }
    
    for (p = tmp + 1; *p; p++) {
        if (*p == '/') {
            *p = 0;
            if (mkdir(tmp, 0755) != 0 && errno != EEXIST) {
                return -1;
            }
            *p = '/';
        }
    }
    
    if (mkdir(tmp, 0755) != 0 && errno != EEXIST) {
        return -1;
    }
    
    return 0;
}

void hash_to_hex(const uint8_t *hash, char *hex, size_t hex_size) {
    if (!hash || !hex || hex_size < 65) return;
    
    for (int i = 0; i < 32; i++) {
        sprintf(hex + i * 2, "%02x", hash[i]);
    }
    hex[64] = '\0';
}

void get_chunk_path(const char *cache_dir,
                   const uint8_t *orig_hash,
                   const uint8_t *dag_hash,
                   char *path,
                   size_t path_size) {
    if (!cache_dir || !orig_hash || !dag_hash || !path) return;
    
    // Use first 4 bytes of orig_hash for directory structure
    snprintf(path, path_size, "%s/chunks/%02x/%02x/",
             cache_dir, orig_hash[0], orig_hash[1]);
    
    // Convert orig_hash to hex for filename
    char hex[65];
    hash_to_hex(orig_hash, hex, sizeof(hex));
    strncat(path, hex, path_size - strlen(path) - 1);
    strncat(path, ".zst", path_size - strlen(path) - 1);
}

int cache_key_compare(const CacheKey *a, const CacheKey *b) {
    if (!a || !b) return -1;
    
    int cmp = memcmp(a->orig_hash, b->orig_hash, 32);
    if (cmp != 0) return cmp;
    
    return memcmp(a->dag_hash, b->dag_hash, 32);
}
