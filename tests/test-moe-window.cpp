#include "testing.h"
#include "../src/llama-moe-window.h"

#include <array>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <stdexcept>
#include <thread>
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

static void test_demand_failure_resumes_prefetch(testing & t) {
    prefetch_gate active, queued;
    queued.open();
    llama_moe_window window(std::vector<llama_moe_window_layer>(3, {32, 16}), 4, 1,
        [&](int32_t il, llama_moe_window_fill fill) {
            auto & gate = fill.expert == 1 ? active : queued;
            gate.load(il, fill);
        },
        [&](int32_t il, const auto &) {
            if (il == 1) {
                active.open();
                throw std::runtime_error("demand read failed");
            }
        });
    prefetch_gate_release release{active};

    window.begin_batch();
    int32_t id = 0, slot = -1;
    window.resolve(0, &id, 1, &slot, {}, {1, 2});
    if (!t.assert_true("first prefetch started", active.wait_started())) {
        return;
    }
    bool failed = false;
    try {
        window.resolve(1, &id, 1, &slot, {}, {});
    } catch (const std::runtime_error &) {
        failed = true;
    }
    t.assert_true("report the demand failure", failed);
    t.assert_true("queued prefetch resumes after failed demand", queued.wait_started());
}

static void test_deferred_completion(testing & t) {
    std::array<int32_t, 2> weights = {-1, -1};
    auto load = [&](int32_t, const auto & fills) {
        for (const auto & fill : fills) {
            weights[fill.slot] = fill.expert;
        }
    };
    llama_moe_window window({{8, 2}, {8, 2}}, 1, 1, [](int32_t, auto) {}, load);
    const std::array<int32_t, 4> ids = {1, 2, 2, 1};
    std::array<int32_t, 4> slots;
    window.begin_batch();
    window.resolve(0, ids.data(), 1, slots.data(), {}, {});
    const int32_t hit = slots[0];
    window.release_all();
    window.begin_batch();

    llama_moe_window_fill offered = {-1, -1};
    window.resolve(0, ids.data(), ids.size(), slots.data(), {3}, {}, [&](const auto & fill) {
        offered = fill;
        return true;
    });
    t.assert_equal("offer the distinct miss", 2, offered.expert);
    t.assert_equal("publish the reserved slot in each missing lane", offered.slot, slots[1]);
    t.assert_equal(slots[1], slots[2]);
    t.assert_equal(hit, slots[3]);
    t.assert_equal("keep the demand idle until dispatch", size_t(1), window.stats().demand);
    t.assert_equal("start predictions after the fill", size_t(0), window.stats().far_queued);
    if (!t.assert_true("valid reserved slot", offered.slot >= 0 && offered.slot < int32_t(weights.size()))) {
        return;
    }
    t.assert_equal(-1, weights[offered.slot]);

    const auto caller = std::this_thread::get_id();
    window.finish_deferred([&](int32_t layer, const auto & fills) {
        t.assert_true("fill on the caller thread", std::this_thread::get_id() == caller);
        t.assert_equal(0, layer);
        t.assert_equal(size_t(1), fills.size());
        load(layer, fills);
    });
    t.assert_equal(2, weights[offered.slot]);
    t.assert_equal(size_t(2), window.stats().demand);
    t.assert_equal(size_t(1), window.stats().far_queued);

    window.release_all();
    window.begin_batch();
    const int32_t new_id = 4;
    window.resolve(0, &new_id, 1, slots.data(), {}, {});
    t.assert_equal("the final hit stays younger than the miss", offered.slot, slots[0]);
}

static void test_deferred_decline(testing & t) {
    size_t offers = 0;
    auto defer = [&](const auto &) {
        ++offers;
        return false;
    };
    llama_moe_window window({{8, 2}}, 0, 0, {}, [](int32_t, const auto &) {});
    std::array<int32_t, 2> ids = {1, 2}, slots;
    window.begin_batch();
    window.resolve(0, ids.data(), ids.size(), slots.data(), {}, {}, defer);
    t.assert_equal("multiple misses take the synchronous path", size_t(0), offers);
    window.release_all();
    window.begin_batch();
    ids = {2, 3};
    window.resolve(0, ids.data(), ids.size(), slots.data(), {}, {}, defer);
    t.assert_equal(size_t(1), offers);
    t.assert_equal("decline loads the miss synchronously", size_t(3), window.stats().demand);
}

static void test_deferred_cleanup(testing & t) {
    auto load = [](int32_t, const auto &) {};
    auto defer = [](const auto &) { return true; };
    llama_moe_window window({{8, 2}}, 0, 0, {}, load);
    std::array<int32_t, 2> ids = {1, 2}, slots;
    window.begin_batch();
    window.resolve(0, ids.data(), 1, slots.data(), {}, {});
    const int32_t hit = slots[0];
    window.release_all();
    window.begin_batch();

    window.resolve(0, ids.data(), ids.size(), slots.data(), {}, {}, defer);
    bool failed = false;
    try {
        window.finish_deferred([](int32_t, const auto &) { throw std::runtime_error("read failed"); });
    } catch (const std::runtime_error &) {
        failed = true;
    }
    t.assert_true("report deferred load failure", failed);
    window.release_all();
    window.begin_batch();

    window.resolve(0, ids.data(), ids.size(), slots.data(), {}, {}, defer);
    window.begin_batch();
    window.resolve(0, ids.data(), 1, slots.data(), {}, {});
    t.assert_equal("cancelled demand can be omitted in the next batch", size_t(2), window.stats().demand);
    window.release_all();
    window.begin_batch();

    window.resolve(0, ids.data(), ids.size(), slots.data(), {}, {}, defer);
    window.release_all();
    window.begin_batch();
    window.resolve(0, ids.data(), ids.size(), slots.data(), {}, {});
    t.assert_equal("retry the missing expert after cancellation", size_t(3), window.stats().demand);
    t.assert_equal("retain the original hit", hit, slots[0]);
}

int main() {
    testing t;
    t.test("depth_zero", test_depth_zero);
    t.test("explicit_lifecycle", test_explicit_lifecycle);
    t.test("terminal_budget", test_terminal_budget);
    t.test("disable_prefetch", test_disable_prefetch);
    t.test("prefetch_failure", test_prefetch_failure);
    t.test("demand_failure_resumes_prefetch", test_demand_failure_resumes_prefetch);
    t.test("deferred_completion", test_deferred_completion);
    t.test("deferred_decline", test_deferred_decline);
    t.test("deferred_cleanup", test_deferred_cleanup);
    return t.summary();
}
