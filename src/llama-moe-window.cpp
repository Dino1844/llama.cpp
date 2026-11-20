#include "llama-moe-window.h"

#include <algorithm>
#include <array>
#include <condition_variable>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

namespace {

enum class phase { absent, queued, loading, resident };
enum heap_id { free_slots, lru, hinted_slots, n_heaps };

struct window_layer {
    struct entry {
        phase state = phase::absent;
        int32_t slot = -1;
        bool pinned = false;
        bool hinted = false;
    };

    struct slot {
        int32_t expert = -1;
        uint64_t age = 0;
        int32_t heap = -1;
        size_t position = 0;
    };

    std::vector<entry> experts;
    std::vector<slot> slots;
    std::array<std::vector<int32_t>, n_heaps> heaps;
    std::vector<int32_t> selected;
    std::vector<int32_t> hinted;
    uint64_t clock = 0;
    size_t prefetch_loading = 0;
    bool prefetch_enabled = true;

    // A plan survives the far job finishing before the adjacent layer refines it.
    struct {
        uint64_t epoch = 0;
        bool refined = false;
        std::vector<int32_t> reads;
    } plan;
    uint64_t hints_epoch = 0;
    std::vector<int32_t> far_hints;
    std::vector<int32_t> near_hints;

    window_layer(const llama_moe_window_layer & cfg, int32_t depth) :
            experts(cfg.n_experts), slots(cfg.n_slots) {
        for (auto & heap : heaps) {
            heap.reserve(slots.size());
        }
        selected.reserve(experts.size());
        hinted.reserve(experts.size());
        plan.reads.reserve(depth);
        far_hints.reserve(experts.size());
        near_hints.reserve(experts.size());
        for (size_t i = 0; i < slots.size(); ++i) {
            insert(static_cast<int32_t>(i), free_slots);
        }
    }

    bool earlier(int32_t a, int32_t b) const {
        return slots[a].age != slots[b].age ? slots[a].age < slots[b].age : a < b;
    }

    void exchange(std::vector<int32_t> & heap, size_t a, size_t b) {
        std::swap(heap[a], heap[b]);
        slots[heap[a]].position = a;
        slots[heap[b]].position = b;
    }

    void sift_up(std::vector<int32_t> & heap, size_t i) {
        for (size_t p = (i - 1) / 2; i && earlier(heap[i], heap[p]); p = (i - 1) / 2) {
            exchange(heap, i, p);
            i = p;
        }
    }

    void sift_down(std::vector<int32_t> & heap, size_t i) {
        for (;;) {
            size_t best = i;
            for (size_t j = 2*i + 1; j < heap.size() && j <= 2*i + 2; ++j) {
                if (earlier(heap[j], heap[best])) {
                    best = j;
                }
            }
            if (best == i) {
                return;
            }
            exchange(heap, i, best);
            i = best;
        }
    }

    void insert(int32_t s, heap_id where) {
        auto & heap = heaps[where];
        slots[s].heap = where;
        slots[s].position = heap.size();
        heap.push_back(s);
        sift_up(heap, heap.size() - 1);
    }

    void remove(int32_t s) {
        slot & v = slots[s];
        if (v.heap < 0) {
            return;
        }
        auto & heap = heaps[v.heap];
        const size_t i = v.position;
        exchange(heap, i, heap.size() - 1);
        heap.pop_back();
        v.heap = -1;
        if (i < heap.size()) {
            if (i && earlier(heap[i], heap[(i - 1) / 2])) {
                sift_up(heap, i);
            } else {
                sift_down(heap, i);
            }
        }
    }

    void make_available(int32_t e) {
        const auto & v = experts[e];
        if (v.state == phase::resident && !v.pinned) {
            insert(v.slot, v.hinted ? hinted_slots : lru);
        }
    }

    int32_t claim(int32_t e, bool background) {
        auto & free = heaps[free_slots];
        const auto & ordinary = heaps[lru];
        const auto & hints = heaps[hinted_slots];
        int32_t s = free.empty() ? -1 : free.front();
        if (s < 0) {
            s = ordinary.empty() ? -1 : ordinary.front();
            if (!hints.empty() && (s < 0 || (background && earlier(hints.front(), s)))) {
                s = hints.front();
            }
        }
        if (s >= 0) {
            remove(s);
            const int32_t old = slots[s].expert;
            if (old >= 0) {
                experts[old].state = phase::absent;
                experts[old].slot = -1;
            }
            slots[s].expert = e;
            experts[e].state = phase::loading;
            experts[e].slot = s;
        }
        return s;
    }

    void complete(llama_moe_window_fill f, bool success) {
        entry & e = experts[f.expert];
        if (success) {
            e.state = phase::resident;
            slots[f.slot].age = ++clock;
            make_available(f.expert);
        } else {
            e.state = phase::absent;
            e.slot = -1;
            slots[f.slot].expert = -1;
            slots[f.slot].age = 0;
            insert(f.slot, free_slots);
        }
    }

    void set_hint(int32_t id, bool value) {
        entry & e = experts[id];
        if (e.state == phase::resident) {
            remove(e.slot);
        }
        e.hinted = value;
        make_available(id);
    }

    void apply_hints(uint64_t epoch) {
        for (int32_t e : hinted) {
            set_hint(e, false);
        }
        hinted.clear();
        if (hints_epoch != epoch) {
            return;
        }
        for (const auto * prediction : { &far_hints, &near_hints }) {
            for (int32_t e : *prediction) {
                if (e >= 0 && static_cast<size_t>(e) < experts.size() && !experts[e].hinted) {
                    set_hint(e, true);
                    hinted.push_back(e);
                }
            }
        }
        far_hints.clear();
        near_hints.clear();
    }

    void release() {
        for (int32_t e : selected) {
            experts[e].pinned = false;
            make_available(e);
        }
        selected.clear();
    }
};

} // namespace

struct llama_moe_window::impl {
    static constexpr size_t queue_depth = 4;

    struct job {
        int32_t layer = -1;
        uint64_t epoch = 0;
        size_t remaining = 0;
    };

    struct task {
        size_t job;
        int32_t expert;
    };

    enum class prediction_stage { far, near, terminal };

    std::vector<window_layer> layers;
    int32_t last_window = -1;
    uint64_t epoch = 0;
    const int32_t depth;
    const prefetch_load prefetch;
    const demand_load demand;

    mutable std::mutex mutex;
    std::condition_variable work_ready;
    std::condition_variable record_ready;
    std::array<job, queue_depth> jobs;
    std::vector<task> queue;
    std::vector<std::thread> workers;
    bool stopping = false;
    llama_moe_window_stats totals;

    std::vector<llama_moe_window_fill> owned;
    std::vector<int32_t> joined;
    struct {
        int32_t layer = -1;
        std::vector<int32_t> slots;
        std::vector<int32_t> near;
        std::vector<int32_t> far;
    } deferred;

    impl(const std::vector<llama_moe_window_layer> & configs, int32_t depth, size_t threads,
            prefetch_load prefetch, demand_load demand) :
            depth(depth), prefetch(std::move(prefetch)), demand(std::move(demand)) {
        if (depth < 0 || depth > max_depth || threads > 64 || !this->demand ||
                (depth > 0 && (threads == 0 || !this->prefetch))) {
            throw std::invalid_argument("invalid MoE window configuration");
        }
        queue.reserve(queue_depth * depth);
        layers.reserve(configs.size());
        size_t max_slots = 0;
        for (const auto & cfg : configs) {
            if (cfg.n_slots < 0 || cfg.n_experts < 0 ||
                    (cfg.n_slots && (cfg.n_experts == 0 || cfg.n_slots <= depth))) {
                throw std::invalid_argument("invalid MoE window dimensions");
            }
            if (cfg.n_slots) {
                last_window = static_cast<int32_t>(layers.size());
                max_slots = std::max(max_slots, static_cast<size_t>(cfg.n_slots));
            }
            layers.emplace_back(cfg, depth);
        }
        owned.reserve(max_slots);
        joined.reserve(max_slots);
        const size_t n_workers = depth > 0 ? threads : 0;
        workers.reserve(n_workers);
        try {
            for (size_t i = 0; i < n_workers; ++i) {
                workers.emplace_back(&impl::worker, this);
            }
        } catch (...) {
            stop();
            throw;
        }
    }

    ~impl() {
        stop();
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
        }
        work_ready.notify_all();
        for (auto & t : workers) {
            t.join();
        }
    }

    void worker() {
        std::unique_lock<std::mutex> lock(mutex);
        for (;;) {
            work_ready.wait(lock, [&] { return stopping || !queue.empty(); });
            if (stopping) {
                return;
            }
            const auto it = std::min_element(queue.begin(), queue.end(), [&](const task & a, const task & b) {
                const auto & x = jobs[a.job];
                const auto & y = jobs[b.job];
                return x.epoch != y.epoch ? x.epoch < y.epoch : x.layer < y.layer;
            });
            const task t = *it;
            queue.erase(it);
            auto & active = jobs[t.job];
            const int32_t il = active.layer;
            auto & layer = layers[il];
            const int32_t slot = layer.prefetch_loading < static_cast<size_t>(depth) ? layer.claim(t.expert, true) : -1;
            if (slot >= 0) {
                ++layer.prefetch_loading;
                ++totals.prefetch;
                lock.unlock();
                bool valid = true;
                try {
                    prefetch(il, { t.expert, slot });
                } catch (...) {
                    valid = false;
                    disable_prefetch(il);
                }
                lock.lock();
                layer.complete({ t.expert, slot }, valid);
                --layer.prefetch_loading;
                totals.failed += !valid;
                record_ready.notify_all();
            } else {
                layer.experts[t.expert].state = phase::absent;
            }
            --active.remaining;
        }
    }

    template<typename Predicate>
    void cancel_queued(Predicate obsolete) {
        queue.erase(std::remove_if(queue.begin(), queue.end(), [&](const task & t) {
            auto & active = jobs[t.job];
            if (!obsolete(active, t)) {
                return false;
            }
            auto & layer = layers[active.layer];
            layer.experts[t.expert].state = phase::absent;
            if (layer.plan.epoch == active.epoch) {
                auto & reads = layer.plan.reads;
                reads.erase(std::remove(reads.begin(), reads.end(), t.expert), reads.end());
            }
            --active.remaining;
            ++totals.cancelled;
            return true;
        }), queue.end());
    }

    void begin_batch() {
        std::lock_guard<std::mutex> lock(mutex);
        cancel_deferred();
        ++epoch;
        cancel_queued([&](const job & j, const task &) { return j.epoch < epoch; });
    }

    void release_all() {
        std::lock_guard<std::mutex> lock(mutex);
        cancel_deferred();
        for (auto & layer : layers) {
            layer.release();
        }
    }

    void disable_prefetch(int32_t il) {
        std::lock_guard<std::mutex> lock(mutex);
        auto & layer = layers.at(il);
        layer.prefetch_enabled = false;
        cancel_queued([&](const job & j, const task &) { return j.layer == il; });
        for (int32_t e : layer.hinted) {
            layer.set_hint(e, false);
        }
        layer.hinted.clear();
        layer.far_hints.clear();
        layer.near_hints.clear();
    }

    void arm(int32_t il, const std::vector<int32_t> & prediction, prediction_stage stage) {
        if (depth == 0 || il > last_window || layers[il].slots.empty() || !layers[il].prefetch_enabled) {
            return;
        }
        auto & target = layers[il];
        if (target.hints_epoch != epoch) {
            target.hints_epoch = epoch;
            target.near_hints.clear();
            target.far_hints.clear();
        }
        auto & plan = target.plan;
        const bool refine = stage == prediction_stage::near;
        const bool refresh = stage != prediction_stage::terminal;
        int32_t budget = depth;
        if (refine) {
            target.near_hints = prediction;
            if (plan.epoch != epoch || plan.refined) {
                return;
            }
            plan.refined = true;
            cancel_queued([&](const job & j, const task & t) {
                return j.layer == il && j.epoch == epoch &&
                    std::find(prediction.begin(), prediction.end(), t.expert) == prediction.end();
            });
            budget -= static_cast<int32_t>(plan.reads.size());
        } else if (refresh) {
            plan.epoch = epoch;
            plan.refined = false;
            plan.reads.clear();
            budget -= std::min(depth, 2);
        } else {
            target.far_hints = prediction;
            if (plan.epoch != epoch) {
                plan.epoch = epoch;
                plan.reads.clear();
            }
            budget -= static_cast<int32_t>(plan.reads.size());
        }
        if (!budget || prediction.empty()) {
            return;
        }
        auto free = jobs.end();
        for (auto it = jobs.begin(); it != jobs.end(); ++it) {
            if (refine && it->remaining && it->layer == il && it->epoch == epoch) {
                free = it;
                break;
            }
            if (!it->remaining && free == jobs.end()) {
                free = it;
            }
        }
        if (free == jobs.end()) {
            return;
        }
        if (!free->remaining) {
            *free = { il, epoch, 0 };
        }
        const size_t job_index = static_cast<size_t>(free - jobs.begin());
        for (int32_t e : prediction) {
            if (e < 0 || static_cast<size_t>(e) >= target.experts.size() ||
                    target.experts[e].state != phase::absent ||
                    std::find(plan.reads.begin(), plan.reads.end(), e) != plan.reads.end()) {
                continue;
            }
            target.experts[e].state = phase::queued;
            queue.push_back({ job_index, e });
            ++free->remaining;
            plan.reads.push_back(e);
            if (refine) {
                ++totals.near_queued;
            } else {
                ++totals.far_queued;
            }
            if (--budget == 0) {
                break;
            }
        }
        if (refresh) {
            target.far_hints = plan.reads;
        }
    }

    void demand_fill(int32_t il, window_layer & layer, std::unique_lock<std::mutex> & lock, const demand_load & load) {
        if (owned.empty()) {
            return;
        }
        totals.demand += owned.size();
        lock.unlock();
        try {
            load(il, owned);
        } catch (...) {
            lock.lock();
            for (const auto & f : owned) {
                layer.complete(f, false);
            }
            owned.clear();
            throw;
        }
        lock.lock();
        for (const auto & f : owned) {
            layer.complete(f, true);
        }
        owned.clear();
    }

    void cancel_deferred() {
        if (deferred.layer >= 0) {
            auto & layer = layers[deferred.layer];
            const auto fill = owned.front();
            layer.complete(fill, false);
            layer.experts[fill.expert].pinned = false;
            layer.selected.erase(std::find(layer.selected.begin(), layer.selected.end(), fill.expert));
            owned.clear();
            deferred.layer = -1;
        }
    }

    void arm_predictions(int32_t il, const std::vector<int32_t> & near, const std::vector<int32_t> & far) {
        if (!near.empty()) {
            arm(il + 1, near, far.empty() ? prediction_stage::terminal : prediction_stage::near);
        }
        if (!far.empty()) {
            arm(il + 2, far, il + 2 == last_window ? prediction_stage::terminal : prediction_stage::far);
        }
    }

    void finish_deferred(const demand_load & load) {
        std::unique_lock<std::mutex> lock(mutex);
        if (deferred.layer < 0) {
            throw std::logic_error("no deferred MoE window fill");
        }
        if (!load) {
            throw std::invalid_argument("missing deferred MoE window loader");
        }
        const int32_t il = std::exchange(deferred.layer, -1);
        auto & layer = layers[il];
        try {
            demand_fill(il, layer, lock, load);
            arm_predictions(il, deferred.near, deferred.far);
            for (int32_t slot : deferred.slots) {
                layer.slots[slot].age = ++layer.clock;
            }
        } catch (...) {
            // An HTP batch can still be using the pinned slots.
            work_ready.notify_all();
            throw;
        }
        lock.unlock();
        work_ready.notify_all();
    }

    void resolve(int32_t il, const int32_t * ids, size_t count, int32_t * slots,
                 const std::vector<int32_t> & near, const std::vector<int32_t> & far, const defer_load & defer) {
        auto & layer = layers.at(il);
        if (layer.slots.empty()) {
            throw std::logic_error("disabled MoE window");
        }
        for (size_t i = 0; i < count; ++i) {
            if (ids[i] < 0 || static_cast<size_t>(ids[i]) >= layer.experts.size()) {
                throw std::out_of_range("expert ID outside MoE window");
            }
        }

        std::unique_lock<std::mutex> lock(mutex);
        if (epoch == 0) {
            throw std::logic_error("MoE window batch was not started");
        }
        if (deferred.layer >= 0) {
            throw std::logic_error("previous MoE window fill is still deferred");
        }
        cancel_queued([&](const job & j, const task &) {
            return j.epoch < epoch || (j.epoch == epoch && j.layer <= il);
        });
        layer.apply_hints(epoch);
        owned.clear();
        joined.clear();

        for (size_t i = 0; i < count; ++i) {
            const int32_t id = ids[i];
            auto & e = layer.experts[id];
            if (!e.pinned) {
                e.pinned = true;
                layer.selected.push_back(id);
                if (e.state == phase::resident) {
                    layer.remove(e.slot);
                }
            }
        }
        try {
            for (int32_t id : layer.selected) {
                const auto & e = layer.experts[id];
                if (e.state == phase::resident) {
                    ++totals.hits;
                } else if (e.state == phase::loading) {
                    ++totals.joined;
                    joined.push_back(id);
                } else {
                    const int32_t s = layer.claim(id, false);
                    if (s < 0) {
                        throw std::runtime_error("selected experts exceed MoE window capacity");
                    }
                    owned.push_back({ id, s });
                }
            }
            if (defer && owned.size() == 1 && joined.empty() && layer.selected.size() > 1) {
                deferred.slots.resize(count);
                deferred.near = near;
                deferred.far = far;
                if (defer(owned.front())) {
                    for (size_t i = 0; i < count; ++i) {
                        slots[i] = layer.experts[ids[i]].slot;
                        deferred.slots[i] = slots[i];
                    }
                    deferred.layer = il;
                    return;
                }
            }
            demand_fill(il, layer, lock, demand);
            // Dispatch all owned reads first. Only wait for selected prefetches.
            for (int32_t id : joined) {
                auto & e = layer.experts[id];
                record_ready.wait(lock, [&] { return e.state != phase::loading; });
                if (e.state != phase::resident) {
                    const int32_t s = layer.claim(id, false);
                    if (s < 0) {
                        throw std::runtime_error("cannot retry MoE window read");
                    }
                    owned.push_back({ id, s });
                }
            }
            demand_fill(il, layer, lock, demand);
            arm_predictions(il, near, far);
            for (size_t i = 0; i < count; ++i) {
                const int32_t s = layer.experts[ids[i]].slot;
                slots[i] = s;
                layer.slots[s].age = ++layer.clock;
            }
        } catch (...) {
            for (const auto & f : owned) {
                layer.complete(f, false);
            }
            owned.clear();
            layer.release();
            work_ready.notify_all();
            throw;
        }
        lock.unlock();
        work_ready.notify_all();
    }
};

llama_moe_window::llama_moe_window(const std::vector<llama_moe_window_layer> & layers, int32_t depth, size_t threads,
        prefetch_load prefetch, demand_load demand) :
        pimpl(std::make_unique<impl>(layers, depth, threads, std::move(prefetch), std::move(demand))) {}

llama_moe_window::~llama_moe_window() = default;

void llama_moe_window::begin_batch() {
    pimpl->begin_batch();
}

void llama_moe_window::release_all() {
    pimpl->release_all();
}

void llama_moe_window::disable_prefetch(int32_t layer) {
    pimpl->disable_prefetch(layer);
}

void llama_moe_window::resolve(int32_t layer, const int32_t * ids, size_t count, int32_t * slots,
        const std::vector<int32_t> & near, const std::vector<int32_t> & far, const defer_load & defer) {
    pimpl->resolve(layer, ids, count, slots, near, far, defer);
}

void llama_moe_window::finish_deferred(const demand_load & load) {
    pimpl->finish_deferred(load);
}

llama_moe_window_stats llama_moe_window::stats() const {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    return pimpl->totals;
}
