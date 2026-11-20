#include "../ggml/src/ggml-hexagon/htp-moe-pacing.h"
#include "ggml.h"

#include <atomic>

static void test_protection() {
    htp_moe_protection p { 1000, 100, 10, 4, 12, 1500 };
    GGML_ASSERT(p.delay(995) == 0);
    GGML_ASSERT(p.delay(996) == 0);
    GGML_ASSERT(p.delay(998) == 12);
    GGML_ASSERT(p.delay(1000) == 10);
    GGML_ASSERT(p.delay(1009) == 1);
    GGML_ASSERT(p.delay(1010) == 0);
    GGML_ASSERT(p.delay(1098) == 12);
    GGML_ASSERT(p.delay(1500) == 0);

    p.max_hold_ns = 9;
    GGML_ASSERT(p.delay(1000) == 0);
    GGML_ASSERT(p.delay(1001) == 9);

    p.expires_ns = 1013;
    GGML_ASSERT(p.delay(1001) == 0);
    p.expires_ns = 1014;
    GGML_ASSERT(p.delay(1001) == 9);

    p.batch_ns = 91;
    GGML_ASSERT(p.delay(1001) == 0);
    p.period_ns = 0;
    GGML_ASSERT(p.delay(1001) == 0);
}

static void test_early_release() {
    using namespace std::chrono;
    const int64_t now = duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count();
    const auto path = std::filesystem::temp_directory_path() / ("ggml-htp-moe-pacing-" + std::to_string(now));
    GGML_ASSERT(std::filesystem::create_directory(path));
    const auto snapshot = path / "phase";
    {
        std::ofstream file(snapshot);
        file << now << " 60000000000 30000000000 500000 30000000000 " << now + 120000000000 << '\n';
        GGML_ASSERT(file.good());
    }

    std::atomic<int> sends {0};
    htp_moe_pacer pacer(snapshot.string().c_str());
    GGML_ASSERT(pacer.submit([&] { ++sends; }));
    GGML_ASSERT(!pacer.submit([&] { ++sends; }));
    pacer.finish();
    GGML_ASSERT(sends == 1 && pacer.finish() == 0);

    std::filesystem::remove(snapshot);
    std::filesystem::remove(path);
}

int main() {
    test_protection();
    test_early_release();
}
