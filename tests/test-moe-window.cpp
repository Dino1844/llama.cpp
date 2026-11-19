#include "testing.h"
#include "../src/llama-moe-window.h"

#include <array>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <stdexcept>
#include <vector>

struct prefetch_gate {
    std::mutex mutex;
    std::condition_variable changed;
    bool started = false;
    bool released = false;

    void load(int32_t, llama_moe_window_fill) {
        std::unique_lock<std::mutex> lock(mutex);
        started = true;
        changed.notify_all();
        changed.wait(lock, [&] { return released; });
    }

    bool wait_started() {
        std::unique_lock<std::mutex> lock(mutex);
        return changed.wait_for(lock, std::chrono::seconds(3), [&] { return started; });
    }

    void open() {
        std::lock_guard<std::mutex> lock(mutex);
        released = true;
        changed.notify_all();
    }
};

struct prefetch_gate_release {
    prefetch_gate & gate;
    ~prefetch_gate_release() { gate.open(); }
};

static void test_depth_zero(testing & t) {
    size_t reads = 0;
    std::array<int32_t, 8> weights;
    weights.fill(-1);
    llama_moe_window window({{32, 8}}, 0, 0, {}, [&](int32_t, const auto & fills) {
        reads += fills.size();
        for (const auto & fill : fills) {
            weights[fill.slot] = fill.expert;
        }
    });

    const std::array<int32_t, 5> ids = {1, 2, 3, 2, 1};
    std::array<int32_t, 5> slots;
    window.begin_batch();
    window.resolve(0, ids.data(), ids.size(), slots.data(), {4, 5}, {6, 7});
    t.assert_equal("load each distinct expert once", size_t(3), reads);
    t.assert_equal(slots[0], slots[4]);
    t.assert_equal(slots[1], slots[3]);
    window.release_all();
    window.begin_batch();
    window.resolve(0, ids.data(), ids.size(), slots.data(), {}, {});
    t.assert_equal("keep resident experts across batches", size_t(3), reads);
    t.assert_equal("depth zero queues no prefetch", size_t(0), window.stats().far_queued);
    for (size_t i = 0; i < ids.size(); ++i) {
        t.assert_equal(ids[i], weights[slots[i]]);
    }
}

static void test_explicit_lifecycle(testing & t) {
    llama_moe_window window({{8, 2}, {8, 2}}, 0, 0, {}, [](int32_t, const auto &) {});
    std::array<int32_t, 2> ids = {0, 1}, slots;
    bool rejected = false;
    try {
        window.resolve(0, ids.data(), ids.size(), slots.data(), {}, {});
    } catch (const std::logic_error &) {
        rejected = true;
    }
    t.assert_true("resolve requires an explicit batch", rejected);

    window.begin_batch();
    window.resolve(0, ids.data(), ids.size(), slots.data(), {}, {});
    window.resolve(1, ids.data(), ids.size(), slots.data(), {}, {});
    window.begin_batch();
    ids = {2, 3};
    rejected = false;
    try {
        window.resolve(0, ids.data(), ids.size(), slots.data(), {}, {});
    } catch (const std::runtime_error &) {
        rejected = true;
    }
    t.assert_true("batch change preserves consumer pins until release", rejected);
    window.release_all();
    window.begin_batch();
    window.resolve(0, ids.data(), ids.size(), slots.data(), {}, {});
    t.assert_true("released slots can be reused", slots[0] >= 0 && slots[1] >= 0 && slots[0] != slots[1]);
}

static void test_terminal_budget(testing & t) {
    prefetch_gate gate;
    llama_moe_window window({{32, 16}, {32, 16}, {32, 16}}, 4, 1,
        [&](int32_t layer, llama_moe_window_fill fill) { gate.load(layer, fill); },
        [](int32_t, const auto &) {});
    prefetch_gate_release release{gate};
    window.begin_batch();
    int32_t id = 0, slot = -1;
    window.resolve(0, &id, 1, &slot, {}, {1, 2, 3, 4});
    if (t.assert_true("far read started", gate.wait_started())) {
        window.resolve(1, &id, 1, &slot, {5, 6, 7, 8}, {});
        const auto stats = window.stats();
        t.assert_equal("terminal far and near share one budget", size_t(4), stats.far_queued + stats.near_queued);
    }
}

static void test_disable_prefetch(testing & t) {
    prefetch_gate gate;
    llama_moe_window window(std::vector<llama_moe_window_layer>(4, {32, 16}), 4, 1,
        [&](int32_t layer, llama_moe_window_fill fill) { gate.load(layer, fill); },
        [](int32_t, const auto &) {});
    prefetch_gate_release release{gate};
    window.begin_batch();
    int32_t id = 0, slot = -1;
    window.resolve(0, &id, 1, &slot, {}, {1, 2, 3, 4});
    if (t.assert_true("far read started", gate.wait_started())) {
        window.disable_prefetch(2);
        const auto before = window.stats();
        t.assert_equal("cancel the queued read without discarding the active one", size_t(1), before.cancelled);
        window.resolve(1, &id, 1, &slot, {5, 6, 7, 8}, {});
        t.assert_equal("disabled layer accepts no near requests", before.near_queued, window.stats().near_queued);
        window.release_all();
        window.begin_batch();
        window.resolve(0, &id, 1, &slot, {}, {5, 6, 7, 8});
        t.assert_equal("disabled layer accepts no requests in a later batch", before.far_queued, window.stats().far_queued);
    }
}

static void test_prefetch_failure(testing & t) {
    prefetch_gate gate;
    std::vector<llama_moe_window_fill> demand;
    llama_moe_window window(std::vector<llama_moe_window_layer>(3, {32, 16}), 4, 1,
        [&](int32_t layer, llama_moe_window_fill fill) {
            gate.load(layer, fill);
            throw std::runtime_error("prefetch read failed");
        },
        [&](int32_t layer, const auto & fills) {
            if (layer == 2) {
                demand = fills;
            }
        });
    prefetch_gate_release release{gate};
    window.begin_batch();
    int32_t id = 0, slot = -1;
    window.resolve(0, &id, 1, &slot, {}, {1});
    const bool started = gate.wait_started();
    gate.open();
    if (!t.assert_true("prefetch read started", started)) {
        return;
    }
    id = 1;
    window.resolve(2, &id, 1, &slot, {}, {});
    t.assert_equal("failed prefetch is retried on demand", size_t(1), demand.size());
    if (!demand.empty()) {
        t.assert_equal(id, demand.front().expert);
        t.assert_equal(slot, demand.front().slot);
    }
    t.assert_equal(size_t(1), window.stats().failed);
    const auto before = window.stats();
    window.release_all();
    window.begin_batch();
    id = 0;
    window.resolve(0, &id, 1, &slot, {}, {2});
    t.assert_equal("failed layer queues no further prefetch", before.far_queued, window.stats().far_queued);
}

int main() {
    testing t;
    t.test("depth_zero", test_depth_zero);
    t.test("explicit_lifecycle", test_explicit_lifecycle);
    t.test("terminal_budget", test_terminal_budget);
    t.test("disable_prefetch", test_disable_prefetch);
    t.test("prefetch_failure", test_prefetch_failure);
    return t.summary();
}
