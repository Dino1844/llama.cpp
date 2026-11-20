#pragma once

#include "llama-mmap.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

struct ggml_tensor;

struct llama_moe_packed_tensor {
    const ggml_tensor * tensor;
    size_t packed_size;
};

struct llama_moe_packed_timing {
    int64_t pread_us = 0;
    int64_t crc_us = 0;
    int64_t crc_cpu_us = 0;
    int64_t max_pread_us = 0;
    int64_t max_crc_us = 0;

    void add(const llama_moe_packed_timing & sample);
};

class llama_moe_packed {
public:
    llama_moe_packed(const char * directory, const llama_file_identity & source, size_t threads);
    ~llama_moe_packed();

    const char * layout() const;
    bool add_layer(int32_t layer, const std::vector<llama_moe_packed_tensor> & tensors, bool compact_q3k);
    using completion = std::function<void(size_t miss, const uint8_t * record, const std::vector<size_t> & family_offsets)>;

    // Read and validate one record on the calling thread. Returns the record size.
    size_t read_one(int32_t layer, int32_t expert, const completion & on_complete,
                    llama_moe_packed_timing * timing = nullptr) const;

    // One controlling thread must join each batch before starting another or destroying the reader.
    // Keep experts unchanged and timing valid until join_async returns or throws.
    // Worker callbacks may run concurrently and must not join the batch. Record data is valid only during the callback and is null on error.
    void start_async(int32_t layer, const std::vector<int32_t> & experts, completion on_complete,
                     llama_moe_packed_timing * timing = nullptr);
    // Waits for all callbacks before returning the payload bytes or rethrowing a worker error.
    size_t join_async();

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
