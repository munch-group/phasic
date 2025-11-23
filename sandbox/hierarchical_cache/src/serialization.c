#include <markov_cache/internal.h>
#include <stdlib.h>
#include <string.h>

size_t serialize_csr(const CSRGraph *csr, uint8_t **out_buffer) {
    if (!csr || !out_buffer) return 0;
    
    // Calculate size needed
    size_t size = 0;
    size += sizeof(uint32_t) * 2;  // num_nodes, num_edges
    
    // row_ptr deltas
    for (uint32_t i = 0; i <= csr->num_nodes; i++) {
        uint32_t delta = (i == 0) ? csr->row_ptr[0] : 
                        csr->row_ptr[i] - csr->row_ptr[i - 1];
        size += varint_size(delta);
    }
    
    // col_idx deltas (within each row)
    uint32_t prev_col = 0;
    for (uint32_t row = 0; row < csr->num_nodes; row++) {
        for (uint32_t i = csr->row_ptr[row]; i < csr->row_ptr[row + 1]; i++) {
            uint32_t delta;
            if (i == csr->row_ptr[row]) {
                delta = csr->col_idx[i];
                prev_col = csr->col_idx[i];
            } else {
                delta = csr->col_idx[i] - prev_col;
                prev_col = csr->col_idx[i];
            }
            size += varint_size(delta);
        }
    }
    
    // values (double precision)
    size += csr->num_edges * sizeof(double);
    
    // Allocate buffer
    *out_buffer = malloc(size);
    if (!*out_buffer) return 0;
    
    uint8_t *ptr = *out_buffer;
    
    // Write num_nodes and num_edges
    memcpy(ptr, &csr->num_nodes, sizeof(uint32_t));
    ptr += sizeof(uint32_t);
    memcpy(ptr, &csr->num_edges, sizeof(uint32_t));
    ptr += sizeof(uint32_t);
    
    // Write row_ptr deltas
    for (uint32_t i = 0; i <= csr->num_nodes; i++) {
        uint32_t delta = (i == 0) ? csr->row_ptr[0] :
                        csr->row_ptr[i] - csr->row_ptr[i - 1];
        ptr += write_varint(ptr, delta);
    }
    
    // Write col_idx deltas
    prev_col = 0;
    for (uint32_t row = 0; row < csr->num_nodes; row++) {
        for (uint32_t i = csr->row_ptr[row]; i < csr->row_ptr[row + 1]; i++) {
            uint32_t delta;
            if (i == csr->row_ptr[row]) {
                delta = csr->col_idx[i];
                prev_col = csr->col_idx[i];
            } else {
                delta = csr->col_idx[i] - prev_col;
                prev_col = csr->col_idx[i];
            }
            ptr += write_varint(ptr, delta);
        }
    }
    
    // Write values
    memcpy(ptr, csr->values, csr->num_edges * sizeof(double));
    ptr += csr->num_edges * sizeof(double);
    
    return ptr - *out_buffer;
}

CSRGraph* deserialize_csr(const uint8_t *buffer, size_t size) {
    if (!buffer || size < 8) return NULL;
    
    const uint8_t *ptr = buffer;
    
    CSRGraph *csr = malloc(sizeof(CSRGraph));
    if (!csr) return NULL;
    
    // Read num_nodes and num_edges
    memcpy(&csr->num_nodes, ptr, sizeof(uint32_t));
    ptr += sizeof(uint32_t);
    memcpy(&csr->num_edges, ptr, sizeof(uint32_t));
    ptr += sizeof(uint32_t);
    
    // Allocate arrays
    csr->row_ptr = malloc((csr->num_nodes + 1) * sizeof(uint32_t));
    csr->col_idx = malloc(csr->num_edges * sizeof(uint32_t));
    csr->values = malloc(csr->num_edges * sizeof(double));
    
    if (!csr->row_ptr || !csr->col_idx || !csr->values) {
        csr_free(csr);
        return NULL;
    }
    
    // Read row_ptr deltas
    for (uint32_t i = 0; i <= csr->num_nodes; i++) {
        uint64_t delta;
        ptr += read_varint(ptr, &delta);
        csr->row_ptr[i] = (i == 0) ? (uint32_t)delta :
                         csr->row_ptr[i - 1] + (uint32_t)delta;
    }
    
    // Read col_idx deltas
    uint32_t prev_col = 0;
    for (uint32_t row = 0; row < csr->num_nodes; row++) {
        for (uint32_t i = csr->row_ptr[row]; i < csr->row_ptr[row + 1]; i++) {
            uint64_t delta;
            ptr += read_varint(ptr, &delta);
            
            if (i == csr->row_ptr[row]) {
                csr->col_idx[i] = (uint32_t)delta;
                prev_col = csr->col_idx[i];
            } else {
                csr->col_idx[i] = prev_col + (uint32_t)delta;
                prev_col = csr->col_idx[i];
            }
        }
    }
    
    // Read values
    memcpy(csr->values, ptr, csr->num_edges * sizeof(double));
    ptr += csr->num_edges * sizeof(double);
    
    return csr;
}
