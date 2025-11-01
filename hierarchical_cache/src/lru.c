#include <markov_cache/internal.h>
#include <stdlib.h>
#include <string.h>

typedef struct LRUNode {
    CacheKey key;
    CSRGraph *csr;
    size_t size_bytes;
    struct LRUNode *prev;
    struct LRUNode *next;
} LRUNode;

struct LRUCache {
    LRUNode *head;
    LRUNode *tail;
    size_t current_size;
    size_t max_size;
};

static uint32_t hash_key(const CacheKey *key) {
    uint32_t h = 0;
    for (int i = 0; i < 32; i++) {
        h = h * 31 + key->orig_hash[i];
    }
    for (int i = 0; i < 32; i++) {
        h = h * 31 + key->dag_hash[i];
    }
    return h;
}

LRUCache* lru_create(size_t max_size_bytes) {
    if (max_size_bytes == 0) return NULL;
    
    LRUCache *cache = malloc(sizeof(LRUCache));
    if (!cache) return NULL;
    
    cache->head = NULL;
    cache->tail = NULL;
    cache->current_size = 0;
    cache->max_size = max_size_bytes;
    
    return cache;
}

static void remove_node(LRUCache *cache, LRUNode *node) {
    if (!node) return;
    
    if (node->prev) {
        node->prev->next = node->next;
    } else {
        cache->head = node->next;
    }
    
    if (node->next) {
        node->next->prev = node->prev;
    } else {
        cache->tail = node->prev;
    }
}

static void add_to_front(LRUCache *cache, LRUNode *node) {
    if (!node) return;
    
    node->next = cache->head;
    node->prev = NULL;
    
    if (cache->head) {
        cache->head->prev = node;
    }
    cache->head = node;
    
    if (!cache->tail) {
        cache->tail = node;
    }
}

static void evict_lru(LRUCache *cache) {
    if (!cache->tail) return;
    
    LRUNode *node = cache->tail;
    
    // Remove from list
    remove_node(cache, node);
    
    // Update size
    cache->current_size -= node->size_bytes;
    
    // Free (don't free CSR as it might be in use)
    free(node);
}

CSRGraph* lru_get(LRUCache *cache, const CacheKey *key) {
    if (!cache || !key) return NULL;
    
    // Linear search (simple but works)
    LRUNode *node = cache->head;
    while (node) {
        if (cache_key_compare(&node->key, key) == 0) {
            // Move to front (most recently used)
            if (node != cache->head) {
                remove_node(cache, node);
                add_to_front(cache, node);
            }
            return node->csr;
        }
        node = node->next;
    }
    
    return NULL;
}

void lru_put(LRUCache *cache, const CacheKey *key, CSRGraph *csr) {
    if (!cache || !key || !csr) return;
    
    // Check if already exists
    LRUNode *existing = cache->head;
    while (existing) {
        if (cache_key_compare(&existing->key, key) == 0) {
            // Already cached, just move to front
            if (existing != cache->head) {
                remove_node(cache, existing);
                add_to_front(cache, existing);
            }
            return;
        }
        existing = existing->next;
    }
    
    // Estimate size
    size_t size = sizeof(CSRGraph) +
                 (csr->num_nodes + 1) * sizeof(uint32_t) +
                 csr->num_edges * sizeof(uint32_t) +
                 csr->num_edges * sizeof(double);
    
    // Evict until we have space
    while (cache->current_size + size > cache->max_size && cache->tail) {
        evict_lru(cache);
    }
    
    // Don't cache if too large
    if (size > cache->max_size) {
        return;
    }
    
    // Create node
    LRUNode *node = malloc(sizeof(LRUNode));
    if (!node) return;
    
    node->key = *key;
    node->csr = csr;
    node->size_bytes = size;
    node->prev = NULL;
    node->next = NULL;
    
    // Add to front of list
    add_to_front(cache, node);
    
    cache->current_size += size;
}

void lru_free(LRUCache *cache) {
    if (!cache) return;
    
    LRUNode *node = cache->head;
    while (node) {
        LRUNode *next = node->next;
        // Don't free CSR as it's managed externally
        free(node);
        node = next;
    }
    
    free(cache);
}
