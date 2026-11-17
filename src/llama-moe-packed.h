#pragma once

#include "llama-mmap.h"

#include <cstddef>
#include <cstdint>
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

    bool add_layer(int32_t layer, const std::vector<llama_moe_packed_tensor> & tensors);
    size_t read(int32_t layer, const std::vector<int32_t> & experts, llama_moe_packed_timing * timing = nullptr);

    // Valid until the next read. Families use the order passed to add_layer.
    const uint8_t * data(size_t miss, size_t family) const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
