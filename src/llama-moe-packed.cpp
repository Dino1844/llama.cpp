#include "llama-moe-packed.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

void llama_moe_packed_timing::add(const llama_moe_packed_timing & sample) {
    pread_us += sample.pread_us;
    crc_us += sample.crc_us;
    crc_cpu_us += sample.crc_cpu_us;
    max_pread_us = std::max(max_pread_us, sample.max_pread_us);
    max_crc_us = std::max(max_crc_us, sample.max_crc_us);
}

#if defined(__linux__)
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#if defined(__linux__) && defined(O_DIRECT)

#include "ggml.h"

#include <nlohmann/json.hpp>

#include <array>
#include <cerrno>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <exception>
#include <filesystem>
#include <limits>
#include <mutex>
#include <string>
#include <system_error>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#if defined(__aarch64__) && !defined(__AARCH64EB__) && (defined(__GNUC__) || defined(__clang__))
#define LLAMA_MOE_ARM_CRC32
#include <arm_acle.h>
#include <asm/hwcap.h>
#include <sys/auxv.h>
#endif

namespace {

constexpr size_t direct_alignment = 4096;
constexpr int read_flags = O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK;
constexpr uint32_t crc32_polynomial = 0xedb88320u;
constexpr const char * tiled_layout = "hexagon-tiled-v1";
constexpr const char * compact_layout = "hexagon-q3k-compact-v1";

using json = nlohmann::json;

json read_index(const std::filesystem::path & path) {
    const int fd = open(path.c_str(), read_flags);
    const auto close_file = [](FILE * file) { fclose(file); };
    std::unique_ptr<FILE, decltype(close_file)> file(fd >= 0 ? fdopen(fd, "rb") : nullptr, close_file);
    if (fd >= 0 && !file) {
        close(fd);
    }
    struct stat st = {};
    if (!file || fstat(fd, &st) != 0 || !S_ISREG(st.st_mode) || st.st_size <= 0 || st.st_size > 8 * 1024 * 1024) {
        throw std::runtime_error("cannot read packed expert index " + path.string() + "; run pack-hexagon-experts.py --verify");
    }
    std::string text(st.st_size, '\0');
    if (fread(text.data(), 1, text.size(), file.get()) != text.size()) {
        throw std::runtime_error("short read of packed expert index " + path.string());
    }
    return json::parse(text);
}

uint64_t packed_integer(const json & value) {
    if (!value.is_number_unsigned()) {
        throw std::runtime_error("invalid unsigned integer in packed expert manifest");
    }
    return value.get<uint64_t>();
}

size_t packed_add(size_t a, size_t b) {
    if (a > SIZE_MAX - b) {
        throw std::runtime_error("packed expert stride overflow");
    }
    return a + b;
}

struct packed_entry {
    std::string file;
    std::vector<std::string> tensors;
    std::vector<uint32_t> checksums;
};

struct packed_layer {
    ~packed_layer() {
        if (fd >= 0) {
            close(fd);
        }
    }

    int fd = -1;
    std::string path;
    size_t payload_size = 0;
    size_t stride = 0;
    std::vector<size_t> family_offsets;
    std::vector<uint32_t> checksums;
};

struct packed_staging {
    std::unique_ptr<void, decltype(&std::free)> buffer { nullptr, std::free };
    size_t capacity = 0;

    void reserve(size_t size) {
        if (size <= capacity) {
            return;
        }
        void * ptr = nullptr;
        const int rc = posix_memalign(&ptr, direct_alignment, size);
        if (rc != 0) {
            throw std::system_error(rc, std::generic_category(), "allocate packed expert staging");
        }
        buffer.reset(ptr);
        capacity = size;
    }

    uint8_t * data() const { return static_cast<uint8_t *>(buffer.get()); }
};

#ifdef LLAMA_MOE_ARM_CRC32
// Multiply reflected CRC32 residues modulo the IEEE polynomial.
uint32_t packed_crc32_multiply(uint32_t a, uint32_t b) {
    uint32_t product = 0;
    while (a) {
        if (a & 0x80000000u) {
            product ^= b;
        }
        a <<= 1;
        b = (b >> 1) ^ ((0u - (b & 1u)) & crc32_polynomial);
    }
    return product;
}

uint32_t packed_crc32_zero_bytes(size_t size) {
    uint32_t result = 0x80000000u;
    uint32_t power = 0x00800000u; // x^8 in reflected representation
    while (size) {
        if (size & 1u) {
            result = packed_crc32_multiply(result, power);
        }
        size >>= 1;
        if (size) {
            power = packed_crc32_multiply(power, power);
        }
    }
    return result;
}

__attribute__((target("+crc")))
uint32_t packed_crc32_arm(const uint8_t * data, size_t size) {
    const auto load_word = [](const uint8_t * ptr) {
        uint64_t word;
        std::memcpy(&word, ptr, sizeof(word));
        return word;
    };

    uint32_t crc = UINT32_MAX;
    if (size >= 32 * 1024) {
        const size_t lane_size = size / (3 * sizeof(uint64_t)) * sizeof(uint64_t);
        uint32_t crc1 = 0;
        uint32_t crc2 = 0;
        for (size_t i = 0; i < lane_size; i += sizeof(uint64_t)) {
            crc  = __crc32d(crc,  load_word(data + i));
            crc1 = __crc32d(crc1, load_word(data + lane_size + i));
            crc2 = __crc32d(crc2, load_word(data + 2 * lane_size + i));
        }
        // Account for the bytes after each lane before XORing its contribution.
        const uint32_t advance = packed_crc32_zero_bytes(lane_size);
        crc = packed_crc32_multiply(advance, crc) ^ crc1;
        crc = packed_crc32_multiply(advance, crc) ^ crc2;
        data += 3 * lane_size;
        size -= 3 * lane_size;
    }
    while (size >= sizeof(uint64_t)) {
        crc = __crc32d(crc, load_word(data));
        data += sizeof(uint64_t);
        size -= sizeof(uint64_t);
    }
    while (size--) {
        crc = __crc32b(crc, *data++);
    }
    return ~crc;
}
#endif

uint32_t packed_crc32(const uint8_t * data, size_t size) {
#ifdef LLAMA_MOE_ARM_CRC32
    static const bool has_crc32 = getauxval(AT_HWCAP) & HWCAP_CRC32;
    if (has_crc32) {
        return packed_crc32_arm(data, size);
    }
#endif
    static constexpr auto table = [] {
        std::array<uint32_t, 256> entries {};
        for (uint32_t i = 0; i < entries.size(); ++i) {
            uint32_t crc = i;
            for (int bit = 0; bit < 8; ++bit) {
                crc = (crc >> 1) ^ ((crc & 1) ? crc32_polynomial : 0);
            }
            entries[i] = crc;
        }
        return entries;
    }();

    uint32_t crc = UINT32_MAX;
    for (size_t i = 0; i < size; ++i) {
        crc = (crc >> 8) ^ table[(crc ^ data[i]) & 0xff];
    }
    return ~crc;
}

int64_t thread_time_us() {
    timespec ts {};
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    return int64_t(ts.tv_sec)*1000000 + ts.tv_nsec/1000;
}

void packed_read_expert(const packed_layer & layer, uint8_t * dest, int32_t expert, llama_moe_packed_timing * timing) {
    const off_t offset = static_cast<off_t>(static_cast<size_t>(expert) * layer.stride);
    const int64_t start = timing ? ggml_time_us() : 0;
    ssize_t count;
    do {
        count = pread(layer.fd, dest, layer.stride, offset);
    } while (count < 0 && errno == EINTR);
    if (count < 0) {
        const int error = errno;
        throw std::system_error(error, std::generic_category(), "packed expert read " + layer.path);
    }
    if (static_cast<size_t>(count) != layer.stride) {
        throw std::runtime_error("short packed expert read from " + layer.path);
    }
    const int64_t crc_start = timing ? ggml_time_us() : 0;
    const int64_t cpu_start = timing ? thread_time_us() : 0;
    const uint32_t checksum = packed_crc32(dest, layer.stride);
    if (timing) {
        timing->crc_cpu_us = thread_time_us() - cpu_start;
        timing->crc_us = ggml_time_us() - crc_start;
        timing->pread_us = crc_start - start;
        timing->max_crc_us = timing->crc_us;
        timing->max_pread_us = timing->pread_us;
    }
    if (checksum != layer.checksums[expert]) {
        throw std::runtime_error("packed expert checksum mismatch in " + layer.path + ", expert " + std::to_string(expert));
    }
}

} // namespace

struct llama_moe_packed::impl {
    std::filesystem::path directory;
    std::unordered_map<int32_t, packed_entry> entries;
    std::unordered_map<int32_t, std::unique_ptr<packed_layer>> layers;
    packed_staging staging;

    std::vector<std::thread> workers;
    std::mutex mutex;
    std::condition_variable ready;
    std::condition_variable complete;
    const packed_layer * active = nullptr;
    const std::vector<int32_t> * jobs = nullptr;
    completion on_complete;
    llama_moe_packed_timing * timing = nullptr;
    size_t next = 0;
    size_t remaining = 0;
    std::exception_ptr error;
    bool stopping = false;
    bool compact_q3k = false;

    impl(const char * directory, const llama_file_identity & source, size_t threads) : directory(directory) {
        if (threads == 0 || threads > 64) {
            throw std::invalid_argument("packed expert I/O needs 1-64 threads");
        }

        const auto path = this->directory / "index.json";
        const json index = read_index(path);
        const auto & format = index.at("layout");
        compact_q3k = format == compact_layout;
        if (packed_integer(index.at("version")) != 1 || (!compact_q3k && format != tiled_layout)) {
            throw std::runtime_error("unsupported packed expert index in " + path.string());
        }

        const auto & model = index.at("source");
        if (source[2] == 0 || !model.is_array() || model.size() != source.size() ||
                !std::equal(source.begin(), source.end(), model.begin(),
                    [](uint64_t expected, const json & value) { return expected == packed_integer(value); })) {
            throw std::runtime_error("packed expert index does not match the loaded GGUF; run pack-hexagon-experts.py --verify");
        }

        std::unordered_set<std::string> files;
        const auto & rows = index.at("layers");
        if (!rows.is_array() || rows.empty()) {
            throw std::runtime_error("invalid packed expert layer list in " + path.string());
        }
        for (const auto & row : rows) {
            const auto il = packed_integer(row.at("layer"));
            if (il > INT32_MAX) {
                throw std::runtime_error("invalid packed expert layer ID");
            }
            packed_entry entry;
            entry.file = row.at("file").get<std::string>();
            entry.tensors = row.at("tensors").get<std::vector<std::string>>();
            if (entry.file.empty() || entry.file == "." || entry.file == ".." ||
                    entry.file.find_first_of("/\\:") != std::string::npos || entry.file.find('\0') != std::string::npos ||
                    !files.insert(entry.file).second) {
                throw std::runtime_error("invalid or duplicate packed expert file name");
            }
            const auto & checksums = row.at("checksums");
            if (!checksums.is_array() || checksums.empty() || checksums.size() > INT32_MAX) {
                throw std::runtime_error("invalid packed expert checksums in " + entry.file);
            }
            entry.checksums.reserve(checksums.size());
            for (const auto & checksum : checksums) {
                const auto crc = packed_integer(checksum);
                if (crc > UINT32_MAX) {
                    throw std::runtime_error("invalid packed expert checksum in " + entry.file);
                }
                entry.checksums.push_back(crc);
            }
            if (!entries.emplace(il, std::move(entry)).second) {
                throw std::runtime_error("duplicate packed expert layer ID");
            }
        }

        workers.reserve(threads);
        try {
            for (size_t i = 0; i < threads; ++i) {
                workers.emplace_back(&impl::worker_loop, this, i);
            }
        } catch (...) {
            stop_workers();
            throw;
        }
    }

    ~impl() {
        stop_workers();
    }

    void stop_workers() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
        }
        ready.notify_all();
        for (auto & worker : workers) {
            worker.join();
        }
    }

    void worker_loop(size_t worker) {
        std::unique_lock<std::mutex> lock(mutex);
        for (;;) {
            ready.wait(lock, [&] { return stopping || (jobs && next < jobs->size()); });
            if (stopping) {
                return;
            }
            const size_t index = next++;
            const int32_t expert = (*jobs)[index];
            const packed_layer & layer = *active;
            auto * dest = staging.data() + worker * layer.stride;
            const bool measure = timing != nullptr;
            lock.unlock();

            std::exception_ptr failure;
            llama_moe_packed_timing sample;
            try {
                packed_read_expert(layer, dest, expert, measure ? &sample : nullptr);
            } catch (...) {
                failure = std::current_exception();
            }

            try {
                on_complete(index, failure ? nullptr : dest, layer.family_offsets);
            } catch (...) {
                if (!failure) {
                    failure = std::current_exception();
                }
            }

            lock.lock();
            if (failure && !error) {
                error = failure;
            }
            if (measure) {
                timing->add(sample);
            }
            if (--remaining == 0) {
                complete.notify_one();
            }
        }
    }

    void read_one(int32_t il, int32_t expert, const completion & on_complete, llama_moe_packed_timing * timing) const {
        const auto & layer = *layers.at(il);
        if (expert < 0 || static_cast<size_t>(expert) >= layer.checksums.size()) {
            throw std::out_of_range("invalid packed expert ID");
        }
        thread_local packed_staging scratch;
        scratch.reserve(layer.stride);
        packed_read_expert(layer, scratch.data(), expert, timing);
        on_complete(0, scratch.data(), layer.family_offsets);
    }

    const char * layout() const {
        return compact_q3k ? compact_layout : tiled_layout;
    }

    bool add_layer(int32_t il, const std::vector<llama_moe_packed_tensor> & tensors, bool bank_compact_q3k) {
        auto it = entries.find(il);
        if (it == entries.end()) {
            return false;
        }
        const auto & entry = it->second;
        if (tensors.empty() || entry.tensors.size() != tensors.size()) {
            throw std::runtime_error("incompatible packed layer " + std::to_string(il));
        }

        auto layer = std::make_unique<packed_layer>();
        layer->family_offsets.resize(tensors.size(), SIZE_MAX);
        const size_t expert_count = entry.checksums.size();
        for (const auto & name : entry.tensors) {
            const auto binding = std::find_if(tensors.begin(), tensors.end(), [&](const llama_moe_packed_tensor & t) {
                return t.tensor && name == t.tensor->name;
            });
            if (binding == tensors.end() || binding->tensor->ne[3] != 1 ||
                    binding->tensor->ne[2] != static_cast<int64_t>(expert_count) || binding->packed_size == 0) {
                throw std::runtime_error("incompatible packed expert tensor " + name);
            }
            if (binding->tensor->type == GGML_TYPE_Q3_K && bank_compact_q3k != compact_q3k) {
                throw std::runtime_error("packed Q3_K layout does not match the cache bank for " + name);
            }
            size_t & offset = layer->family_offsets[binding - tensors.begin()];
            if (offset != SIZE_MAX) {
                throw std::runtime_error("duplicate packed expert tensor " + name);
            }
            offset = layer->payload_size;
            layer->payload_size = packed_add(layer->payload_size, binding->packed_size);
        }
        layer->stride = packed_add(layer->payload_size, direct_alignment - 1) & ~(direct_alignment - 1);
        if (layer->stride > static_cast<size_t>(std::numeric_limits<ssize_t>::max()) ||
                expert_count > SIZE_MAX / layer->stride) {
            throw std::runtime_error("invalid packed expert stride in " + entry.file);
        }
        layer->path = (directory / entry.file).string();
        layer->fd = open(layer->path.c_str(), read_flags | O_DIRECT);
        if (layer->fd < 0 && (errno == EINVAL || errno == EOPNOTSUPP)) {
            layer->fd = open(layer->path.c_str(), read_flags);
        }
        if (layer->fd < 0) {
            const int error = errno;
            throw std::system_error(error, std::generic_category(), "open packed experts " + layer->path);
        }
        struct stat stat = {};
        if (fstat(layer->fd, &stat) != 0) {
            const int error = errno;
            throw std::system_error(error, std::generic_category(), "stat " + layer->path);
        }
        if (!S_ISREG(stat.st_mode) || stat.st_size < 0 ||
                static_cast<uint64_t>(stat.st_size) != layer->stride * expert_count) {
            throw std::runtime_error("incorrect packed expert file size: " + layer->path);
        }
        layer->checksums = std::move(it->second.checksums);
        layers.emplace(il, std::move(layer));
        entries.erase(it);
        return true;
    }

    void start(int32_t il, const std::vector<int32_t> & experts, llama_moe_packed_timing * timings,
               completion callback) {
        std::lock_guard<std::mutex> lock(mutex);
        if (jobs) {
            throw std::logic_error("packed expert read already active");
        }
        const packed_layer & layer = *layers.at(il);
        for (int32_t expert : experts) {
            if (expert < 0 || static_cast<size_t>(expert) >= layer.checksums.size()) {
                throw std::out_of_range("invalid packed expert ID");
            }
        }

        const size_t records = experts.empty() ? 0 : workers.size();
        if (std::max(records, experts.size()) > SIZE_MAX / layer.stride) {
            throw std::runtime_error("packed expert staging size overflow");
        }
        staging.reserve(records * layer.stride);

        active      = &layer;
        jobs        = &experts;
        on_complete = std::move(callback);
        timing      = timings;
        next        = 0;
        remaining   = experts.size();
        error       = nullptr;
        ready.notify_all();
    }

    size_t join() {
        std::unique_lock<std::mutex> lock(mutex);
        if (!jobs) {
            throw std::logic_error("no active packed expert read");
        }
        complete.wait(lock, [&] { return remaining == 0; });

        const size_t bytes = jobs->size() * active->payload_size;
        jobs = nullptr;
        active = nullptr;
        timing = nullptr;
        on_complete = {};
        if (error) {
            std::rethrow_exception(error);
        }
        return bytes;
    }

};

#else

struct llama_moe_packed::impl {
    impl(const char *, const llama_file_identity &, size_t) {
        throw std::runtime_error("packed experts require O_DIRECT support");
    }

    const char * layout() const { return ""; }
    void read_one(int32_t, int32_t, const completion &, llama_moe_packed_timing *) const {
        throw std::runtime_error("packed experts require O_DIRECT support");
    }
    bool add_layer(int32_t, const std::vector<llama_moe_packed_tensor> &, bool) { return false; }
    void start(int32_t, const std::vector<int32_t> &, llama_moe_packed_timing *, completion) {}
    size_t join() { return 0; }
};

#endif

llama_moe_packed::llama_moe_packed(const char * directory, const llama_file_identity & source, size_t threads) :
    pimpl(std::make_unique<impl>(directory, source, threads)) {}

llama_moe_packed::~llama_moe_packed() = default;

const char * llama_moe_packed::layout() const {
    return pimpl->layout();
}

bool llama_moe_packed::add_layer(int32_t layer, const std::vector<llama_moe_packed_tensor> & tensors, bool compact_q3k) {
    return pimpl->add_layer(layer, tensors, compact_q3k);
}

void llama_moe_packed::read_one(int32_t layer, int32_t expert, const completion & on_complete, llama_moe_packed_timing * timing) const {
    pimpl->read_one(layer, expert, on_complete, timing);
}

void llama_moe_packed::start_async(int32_t layer, const std::vector<int32_t> & experts, completion on_complete,
                                  llama_moe_packed_timing * timing) {
    if (!on_complete) {
        throw std::invalid_argument("packed expert completion callback is required");
    }
    pimpl->start(layer, experts, timing, std::move(on_complete));
}

size_t llama_moe_packed::join_async() {
    return pimpl->join();
}
