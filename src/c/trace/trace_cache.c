/*
 * MIT License
 *
 * Copyright (c) 2021 Tobias Røikjer
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:

 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.

 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/**
 * @file trace_cache.c
 * @brief Cache I/O operations for elimination traces
 *
 * Provides disk caching of elimination traces to avoid redundant computation.
 * Traces are stored in ~/.phasic_cache/traces/ as JSON files named by their
 * hash.
 */

#include "trace_internal.h"
#include "../phasic_log.h"

int get_cache_dir(char *buffer, size_t buffer_size) {
    const char *home = getenv("HOME");
    if (home == NULL) {
        PTD_LOG_WARNING("Cache unavailable: HOME environment variable not set");
        sprintf((char*)ptd_err, "HOME environment variable not set");
        return -1;
    }

    // Build path: ~/.phasic_cache/traces
    int ret = snprintf(buffer, buffer_size, "%s/.phasic_cache/traces", home);
    if (ret < 0 || (size_t)ret >= buffer_size) {
        PTD_LOG_ERROR("Cache directory path too long");
        sprintf((char*)ptd_err, "Cache directory path too long");
        return -1;
    }

    // Create directory if it doesn't exist (mkdir -p)
    char parent_dir[PATH_MAX];
    snprintf(parent_dir, sizeof(parent_dir), "%s/.phasic_cache", home);

    // Create parent directory
    struct stat st = {0};
    if (stat(parent_dir, &st) == -1) {
        if (mkdir(parent_dir, 0755) == -1) {
            PTD_LOG_WARNING("Failed to create cache directory: %s", parent_dir);
            sprintf((char*)ptd_err, "Failed to create cache directory: %s", parent_dir);
            return -1;
        }
        PTD_LOG_DEBUG("Created cache directory: %s", parent_dir);
    }

    // Create traces subdirectory
    if (stat(buffer, &st) == -1) {
        if (mkdir(buffer, 0755) == -1) {
            PTD_LOG_WARNING("Failed to create traces directory: %s", buffer);
            sprintf((char*)ptd_err, "Failed to create traces directory: %s", buffer);
            return -1;
        }
        PTD_LOG_DEBUG("Created traces directory: %s", buffer);
    }

    return 0;
}

struct ptd_elimination_trace *load_trace_from_cache(const char *hash_hex) {
    if (hash_hex == NULL) return NULL;

    PTD_LOG_DEBUG("Attempting to load trace from cache: %.16s...", hash_hex);

    // Get cache directory
    char cache_dir[PATH_MAX];
    if (get_cache_dir(cache_dir, sizeof(cache_dir)) != 0) {
        PTD_LOG_DEBUG("Cache directory unavailable");
        return NULL;  // Cache directory unavailable
    }

    // Build cache file path
    char cache_file[PATH_MAX];
    snprintf(cache_file, sizeof(cache_file), "%s/%s.json", cache_dir, hash_hex);

    // Check if file exists
    FILE *f = fopen(cache_file, "r");
    if (f == NULL) {
        PTD_LOG_DEBUG("Cache miss for hash %.16s...", hash_hex);
        return NULL;  // Cache miss
    }

    // Read file into buffer
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (file_size <= 0 || file_size > 100*1024*1024) {  // Max 100MB
        PTD_LOG_WARNING("Cache file invalid size (%ld bytes) for hash %.16s...",
                        file_size, hash_hex);
        fclose(f);
        return NULL;
    }

    char *json = malloc(file_size + 1);
    if (json == NULL) {
        PTD_LOG_ERROR("Failed to allocate memory for cache file (%ld bytes)", file_size);
        fclose(f);
        return NULL;
    }

    size_t read = fread(json, 1, file_size, f);
    fclose(f);

    if ((long)read != file_size) {
        PTD_LOG_WARNING("Failed to read complete cache file for hash %.16s...", hash_hex);
        free(json);
        return NULL;
    }

    json[file_size] = '\0';

    // Deserialize
    struct ptd_elimination_trace *trace = json_to_trace(json);
    free(json);

    if (trace != NULL) {
        PTD_LOG_INFO("Cache hit: loaded trace for hash %.16s... (%ld bytes)",
                     hash_hex, file_size);
    } else {
        PTD_LOG_WARNING("Cache file corrupt: failed to deserialize trace for hash %.16s...",
                        hash_hex);
    }

    return trace;
}

bool save_trace_to_cache(const char *hash_hex, const struct ptd_elimination_trace *trace) {
    if (hash_hex == NULL || trace == NULL) return false;

    PTD_LOG_DEBUG("Attempting to save trace to cache: %.16s...", hash_hex);

    // Get cache directory
    char cache_dir[PATH_MAX];
    if (get_cache_dir(cache_dir, sizeof(cache_dir)) != 0) {
        PTD_LOG_DEBUG("Cache directory unavailable, cannot save trace");
        return false;  // Silently fail if cache unavailable
    }

    // Build cache file path
    char cache_file[PATH_MAX];
    snprintf(cache_file, sizeof(cache_file), "%s/%s.json", cache_dir, hash_hex);

    // Serialize to JSON
    char *json = trace_to_json(trace);
    if (json == NULL) {
        PTD_LOG_ERROR("Failed to serialize trace to JSON for hash %.16s...", hash_hex);
        return false;
    }

    // Write to file
    FILE *f = fopen(cache_file, "w");
    if (f == NULL) {
        PTD_LOG_WARNING("Failed to open cache file for writing: %s", cache_file);
        free(json);
        return false;
    }

    size_t len = strlen(json);
    size_t written = fwrite(json, 1, len, f);
    fclose(f);
    free(json);

    if (written == len) {
        PTD_LOG_INFO("Saved trace to cache: %.16s... (%zu bytes)", hash_hex, len);
        return true;
    } else {
        PTD_LOG_ERROR("Failed to write complete cache file for hash %.16s... (%zu/%zu bytes)",
                      hash_hex, written, len);
        return false;
    }
}
