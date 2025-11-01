#include <markov_cache/internal.h>

size_t write_varint(uint8_t *buffer, uint64_t value) {
    size_t bytes = 0;
    
    while (value >= 0x80) {
        buffer[bytes++] = (uint8_t)((value & 0x7F) | 0x80);
        value >>= 7;
    }
    buffer[bytes++] = (uint8_t)value;
    
    return bytes;
}

size_t read_varint(const uint8_t *buffer, uint64_t *value) {
    *value = 0;
    size_t bytes = 0;
    uint32_t shift = 0;
    
    while (1) {
        uint8_t byte = buffer[bytes++];
        *value |= (uint64_t)(byte & 0x7F) << shift;
        
        if ((byte & 0x80) == 0) {
            break;
        }
        shift += 7;
    }
    
    return bytes;
}

size_t varint_size(uint64_t value) {
    size_t bytes = 1;
    
    while (value >= 0x80) {
        bytes++;
        value >>= 7;
    }
    
    return bytes;
}
