#ifndef HTP_Q3K_COMPACT_H
#define HTP_Q3K_COMPACT_H

#include <stddef.h>
#include <stdint.h>

// One group covers 32 rows and 256 K values. Eight 384-byte quant planes
// precede 32 fp16 multipliers and 12 columns of 32 packed scale bytes.
#define HTP_Q3K_COMPACT_GROUP_BYTES   3520
#define HTP_Q3K_COMPACT_QUANTS_BYTES  3072
#define HTP_Q3K_COMPACT_SCALES_OFFSET 3136

// K must be a multiple of 256. Only the end of the 32-row slab is padded.
static inline size_t htp_q3k_compact_slab_size(uint32_t k) {
    const uint64_t bytes = ((uint64_t) (k / 256) * HTP_Q3K_COMPACT_GROUP_BYTES + 127) & ~(uint64_t) 127;
    return bytes <= SIZE_MAX ? (size_t) bytes : 0;
}

#endif // HTP_Q3K_COMPACT_H
