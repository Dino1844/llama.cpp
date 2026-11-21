#pragma once

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <mutex>
#include <string>
#include <system_error>
#include <thread>
#include <utility>

struct htp_moe_protection {
    int64_t anchor_ns = 0;
    int64_t period_ns = 0;
    int64_t protect_ns = 0;
    int64_t batch_ns = 0;
    int64_t max_hold_ns = 0;
    int64_t expires_ns = 0;

    int64_t delay(int64_t now_ns) const {
        if (now_ns <= 0 || anchor_ns <= 0 || expires_ns <= now_ns ||
                period_ns <= 0 || protect_ns <= 0 || protect_ns >= period_ns ||
                batch_ns <= 0 || batch_ns > period_ns - protect_ns || max_hold_ns <= 0 || max_hold_ns >= period_ns) {
            return 0;
        }

        int64_t phase = (now_ns - anchor_ns) % period_ns;
        if (phase < 0) {
            phase += period_ns;
        }
        int64_t hold = 0;
        if (phase < protect_ns) {
            hold = protect_ns - phase;
        } else if (period_ns - phase < batch_ns) {
            hold = period_ns - phase + protect_ns;
        }

        const int64_t fresh_ns = expires_ns - now_ns;
        return hold <= max_hold_ns && hold <= fresh_ns && batch_ns <= fresh_ns - hold ? hold : 0;
    }
};

class htp_moe_pacer {
    using clock = std::chrono::steady_clock;

public:
    explicit htp_moe_pacer(const char * value = std::getenv("GGML_HEXAGON_MOE_PACING_FILE")) {
        if (value) {
            path = value;
        }
    }

    ~htp_moe_pacer() {
        finish();
        if (worker.joinable()) {
            {
                std::lock_guard<std::mutex> lock(mutex);
                stop = true;
            }
            changed.notify_one();
            worker.join();
        }
    }

    bool enabled() const { return !path.empty(); }

    // All calls except the send itself belong to the session's graph thread.
    bool submit(std::function<void()> send) {
        auto decision = clock::now();
        const int64_t hold_ns = delay(decision);
        if (!hold_ns) {
            return false;
        }

        std::lock_guard<std::mutex> lock(mutex);
        if (!worker.joinable()) {
            try {
                worker = std::thread([this] { run(); });
            } catch (const std::system_error &) {
                return false;
            }
        }
        if (active) {
            return false;
        }
        decision_at = decision;
        release_at = decision + std::chrono::nanoseconds(hold_ns);
        pending = std::move(send);
        active = true;
        expedite = false;
        changed.notify_one();
        return true;
    }

    // Keep intact routed packets on the graph thread to preserve queue order.
    int64_t pause() {
        auto decision = clock::now();
        const int64_t hold_ns = delay(decision);
        if (!hold_ns) {
            return 0;
        }
        std::this_thread::sleep_until(decision + std::chrono::nanoseconds(hold_ns));
        return std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now() - decision).count();
    }

    // Release the hold at read completion.
    int64_t finish() {
        if (!worker.joinable()) {
            return 0;
        }
        std::unique_lock<std::mutex> lock(mutex);
        expedite = true;
        changed.notify_one();
        changed.wait(lock, [&] { return !active; });
        return std::exchange(actual_delay_ns, 0);
    }

private:
    int64_t delay(clock::time_point & now) {
        if (!enabled()) {
            return 0;
        }
        if (now >= refresh_at) {
            refresh_at = now + std::chrono::milliseconds(50);
            refresh();
        }
        now = clock::now();
        const int64_t now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
        return protection.delay(now_ns);
    }

    void refresh() {
        protection = {};
        std::error_code error;
        if (!std::filesystem::is_regular_file(path, error) || std::filesystem::file_size(path, error) > 160 || error) {
            return;
        }
        std::ifstream file(path);
        htp_moe_protection next;
        if (file >> next.anchor_ns >> next.period_ns >> next.protect_ns >> next.batch_ns >> next.max_hold_ns >> next.expires_ns) {
            file >> std::ws;
            if (file.eof()) {
                protection = next;
            }
        }
    }

    void run() {
        std::unique_lock<std::mutex> lock(mutex);
        while (true) {
            changed.wait(lock, [&] { return active || stop; });
            if (stop) {
                return;
            }
            changed.wait_until(lock, release_at, [&] { return expedite; });
            auto send = std::move(pending);
            actual_delay_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now() - decision_at).count();
            lock.unlock();
            send();
            lock.lock();
            active = false;
            changed.notify_all();
        }
    }

    std::string path;
    htp_moe_protection protection;
    clock::time_point refresh_at{};

    std::mutex mutex;
    std::condition_variable changed;
    std::thread worker;
    std::function<void()> pending;
    clock::time_point decision_at;
    clock::time_point release_at;
    bool active = false;
    bool expedite = false;
    bool stop = false;
    int64_t actual_delay_ns = 0;
};
