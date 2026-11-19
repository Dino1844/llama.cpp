#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

struct llama_moe_window_layer {
    int32_t n_experts;
    int32_t n_slots;
};

struct llama_moe_window_fill {
    int32_t expert;
    int32_t slot;
};

struct llama_moe_window_stats {
    size_t hits = 0;
    size_t joined = 0;
    size_t demand = 0;
    size_t prefetch = 0;
    size_t failed = 0;
    size_t cancelled = 0;
    size_t far_queued = 0;
    size_t near_queued = 0;
};

// Holds independent per-layer device windows. Loaders finish their writes before returning.
class llama_moe_window {
public:
    static constexpr int32_t max_depth = 8;

    using prefetch_load = std::function<void(int32_t, llama_moe_window_fill)>;
    using demand_load = std::function<void(int32_t, const std::vector<llama_moe_window_fill> &)>;

    llama_moe_window(const std::vector<llama_moe_window_layer> & layers, int32_t depth, size_t threads,
                     prefetch_load prefetch, demand_load demand);
    ~llama_moe_window();

    void begin_batch();
    // Release consumer pins only after the graph scheduler has synchronized.
    void release_all();
    void disable_prefetch(int32_t layer);

    // Calls are serial within a batch. The scheduler makes routing inputs ready before resolving.
    void resolve(int32_t layer, const int32_t * ids, size_t count, int32_t * slots,
                 const std::vector<int32_t> & near, const std::vector<int32_t> & far);
    llama_moe_window_stats stats() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
