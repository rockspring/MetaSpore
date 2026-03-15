//
// Copyright 2022 DMetaSoul
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#include <common/logger.h>
#include <serving/gpu_utils.h>
#include <serving/ort_model.h>
#include <common/utils.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <filesystem>
#include <shared_mutex>
#include <thread>

#include <boost/core/demangle.hpp>
#include <fmt/format.h>
#include <fmt/ranges.h>

namespace metaspore::serving {

DECLARE_uint64(ort_intraop_thread_num);
DECLARE_uint64(ort_interop_thread_num);

// ---------------------------------------------------------------------------
// Global profiling signal version counters.
//
// Signal handlers only increment these counters (async-signal-safe).
// Each OrtModelContext tracks its own "last processed" version so every
// loaded model reacts to each signal independently.
// ---------------------------------------------------------------------------
static std::atomic<uint64_t> g_profiling_start_version{0};
static std::atomic<uint64_t> g_profiling_stop_version{0};

static void on_start_profiling(int) {
    g_profiling_start_version.fetch_add(1, std::memory_order_release);
}

static void on_stop_profiling(int) {
    g_profiling_stop_version.fetch_add(1, std::memory_order_release);
}

class OrtModelGlobal {
  public:
    OrtModelGlobal() : env_() { install_signal_handlers(); }

    Ort::Env env_;

  private:
    static void install_signal_handlers() {
        struct sigaction sa{};
        sigemptyset(&sa.sa_mask);
        sa.sa_flags = SA_RESTART;

        sa.sa_handler = on_start_profiling;
        sigaction(SIGUSR1, &sa, nullptr);

        sa.sa_handler = on_stop_profiling;
        sigaction(SIGUSR2, &sa, nullptr);

        spdlog::info("OrtModel: SIGUSR1=start profiling, SIGUSR2=stop profiling");
    }
};

static OrtModelGlobal &get_ort_model_global() {
    static OrtModelGlobal global;
    return global;
}

// ---------------------------------------------------------------------------
// OrtModelContext
// ---------------------------------------------------------------------------
class OrtModelContext {
  public:
    OrtModelContext() : run_options_(), session_options_(), session_(nullptr) {
        configure_session_options(session_options_);
    }

    static void configure_session_options(Ort::SessionOptions &opts) {
        opts.SetExecutionMode(ExecutionMode::ORT_PARALLEL);
        opts.SetInterOpNumThreads(FLAGS_ort_interop_thread_num);
        opts.SetIntraOpNumThreads(FLAGS_ort_intraop_thread_num);
        opts.DisableCpuMemArena();
        opts.DisableMemPattern();
    }

    // Build a fresh Ort::Session (optionally with profiling).
    // Must be called while holding an exclusive lock on session_mutex_ so that
    // no Run() is in progress during the swap.
    void rebuild_session_locked(const std::string &profile_prefix = "") {
        Ort::SessionOptions new_opts;
        configure_session_options(new_opts);
        if (!profile_prefix.empty()) {
            new_opts.EnableProfiling(profile_prefix.c_str());
        }
        if (GpuHelper::is_gpu_available()) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"
            OrtSessionOptionsAppendExecutionProvider_CUDA(new_opts, 0);
#pragma GCC diagnostic pop
        }
        auto file = std::filesystem::path(dir_path_) / "model.onnx";
        session_options_ = std::move(new_opts);
        session_ = Ort::Session(get_ort_model_global().env_, file.c_str(), session_options_);
    }

    // CAS 0→1: only one thread enters StartProfiling.
    // Acquires exclusive lock so no Run() overlaps with the session swap.
    void try_start_profiling() {
        int expected = 0;
        if (!profiling_state_.compare_exchange_strong(expected, 1,
                                                      std::memory_order_acq_rel,
                                                      std::memory_order_relaxed)) {
            spdlog::warn("OrtModel ({}): start profiling requested but already profiling",
                         dir_path_);
            return;
        }
        try {
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::system_clock::now().time_since_epoch())
                          .count();
            auto prefix = "/tmp" + dir_path_ + "/ort_profile_" + std::to_string(ms);
            std::filesystem::create_directories(std::filesystem::path(prefix).parent_path());
            {
                std::unique_lock lock(session_mutex_);
                rebuild_session_locked(prefix);
            }
            spdlog::info("OrtModel ({}): profiling started, prefix: {}", dir_path_, prefix);
        } catch (const std::exception &e) {
            profiling_state_.store(0, std::memory_order_release); // allow retry
            spdlog::error("OrtModel ({}): failed to start profiling: {}", dir_path_, e.what());
        }
    }

    // CAS 1→0: only one thread enters EndProfiling.
    // Acquires exclusive lock to guarantee no Run() is in progress when
    // EndProfiling() is called — ORT's Profiler::Start() throws if called
    // after EndProfiling() sets enabled_=false (observed with Loop nodes).
    // Immediately rebuilds a non-profiling session so subsequent inferences
    // never touch the disabled profiler.
    void try_stop_profiling() {
        int expected = 1;
        if (!profiling_state_.compare_exchange_strong(expected, 0,
                                                      std::memory_order_acq_rel,
                                                      std::memory_order_relaxed)) {
            spdlog::warn("OrtModel ({}): stop profiling requested but not profiling", dir_path_);
            return;
        }
        try {
            std::string profile_path;
            {
                std::unique_lock lock(session_mutex_);
                // EndProfiling() flushes the trace file; after this call
                // enabled_=false inside ORT. Any Run() on this session would fail.
                char *path = session_.EndProfiling(allocator_);
                profile_path = path;
                ::free(path);
                // Rebuild immediately so the session is usable again.
                rebuild_session_locked();
            }
            spdlog::info("OrtModel ({}): profiling stopped, file: {}", dir_path_, profile_path);
        } catch (const std::exception &e) {
            spdlog::error("OrtModel ({}): failed to stop profiling: {}", dir_path_, e.what());
        }
    }

    void start_monitor() {
        // Ignore signals that arrived before this model was loaded.
        last_start_version_ = g_profiling_start_version.load(std::memory_order_acquire);
        last_stop_version_ = g_profiling_stop_version.load(std::memory_order_acquire);

        monitor_running_.store(true, std::memory_order_release);
        monitor_thread_ = std::thread([this] {
            while (monitor_running_.load(std::memory_order_acquire)) {
                uint64_t sv = g_profiling_start_version.load(std::memory_order_acquire);
                if (sv > last_start_version_) {
                    last_start_version_ = sv;
                    try_start_profiling();
                }
                uint64_t ev = g_profiling_stop_version.load(std::memory_order_acquire);
                if (ev > last_stop_version_) {
                    last_stop_version_ = ev;
                    try_stop_profiling();
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        });
    }

    ~OrtModelContext() {
        monitor_running_.store(false, std::memory_order_release);
        if (monitor_thread_.joinable()) {
            monitor_thread_.join();
        }
        for (auto p : input_names_) {
            ::free((void *)p);
        }
        for (auto p : output_names_) {
            ::free((void *)p);
        }
    }

    // session_mutex_ guards session_ and session_options_.
    //
    // do_predict holds a shared (read) lock — many concurrent Run() calls allowed.
    // try_start/stop_profiling hold an exclusive (write) lock — ensures no Run()
    // is in progress during session swap or EndProfiling().
    mutable std::shared_mutex session_mutex_;

    Ort::RunOptions run_options_;
    Ort::SessionOptions session_options_;
    Ort::Session session_;
    Ort::AllocatorWithDefaultOptions allocator_;
    std::string dir_path_;
    std::vector<std::string> input_names_s_;
    std::vector<std::string> output_names_s_;
    std::vector<const char *> input_names_;
    std::vector<const char *> output_names_;

    // profiling_state_: 0 = idle, 1 = profiling
    std::atomic<int> profiling_state_{0};

    std::atomic<bool> monitor_running_{false};
    std::thread monitor_thread_;
    uint64_t last_start_version_{0};
    uint64_t last_stop_version_{0};
};

OrtModel::OrtModel() : context_(std::make_unique<OrtModelContext>()) {}

OrtModel::OrtModel(OrtModel &&) = default;

// To avoid std::unique_ptr requires complete type for OrtModelContext
OrtModel::~OrtModel() = default;

awaitable_status OrtModel::load(std::string dir_path) {
    auto &tp = Threadpools::get_background_threadpool();
    auto r = co_await boost::asio::co_spawn(
        tp,
        [this, &dir_path]() -> awaitable_status {
            auto dir = std::filesystem::path(dir_path);
            if (!std::filesystem::is_directory(dir)) {
                co_return absl::InvalidArgumentError(
                    fmt::format("{} is not a dir for OrtModel to load", dir_path));
            }
            auto file = dir / "model.onnx";
            if (!std::filesystem::is_regular_file(file)) {
                co_return absl::NotFoundError(
                    fmt::format("model.onnx doesn't exist under {}", dir_path));
            }
            if (GpuHelper::is_gpu_available()) {
                spdlog::info("Use cuda:0");
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"
                OrtSessionOptionsAppendExecutionProvider_CUDA(context_->session_options_, 0);
#pragma GCC diagnostic pop
            }
            context_->session_ =
                Ort::Session(get_ort_model_global().env_, file.c_str(), context_->session_options_);

            const size_t input_count = context_->session_.GetInputCount();
            context_->input_names_.reserve(input_count);
            context_->input_names_s_.reserve(input_count);
            for (size_t i = 0UL; i < input_count; ++i) {
                context_->input_names_.push_back(
                    context_->session_.GetInputName(i, context_->allocator_));
                context_->input_names_s_.push_back(context_->input_names_.back());
            }

            const size_t output_count = context_->session_.GetOutputCount();
            context_->output_names_.reserve(output_count);
            context_->output_names_s_.reserve(output_count);
            for (size_t i = 0UL; i < output_count; ++i) {
                context_->output_names_.push_back(
                    context_->session_.GetOutputName(i, context_->allocator_));
                context_->output_names_s_.push_back(context_->output_names_.back());
            }

            context_->dir_path_ = dir_path;
            spdlog::info("OrtModel loaded from {}, required inputs [{}], "
                         "producing outputs [{}]",
                         dir_path, fmt::join(context_->input_names_s_, ", "),
                         fmt::join(context_->output_names_s_, ", "));

            context_->start_monitor();
            co_return absl::OkStatus();
        },
        boost::asio::use_awaitable);
    co_return r;
}

awaitable_result<std::unique_ptr<OrtModelOutput>>
OrtModel::do_predict(std::unique_ptr<OrtModelInput> input) {
    // Shared lock: allows concurrent Run() calls.
    // Blocks only when try_stop_profiling holds the exclusive lock to call
    // EndProfiling() — prevents Run() from encountering enabled_=false.
    std::shared_lock lock(context_->session_mutex_);

    const size_t input_count = context_->session_.GetInputCount();
    std::vector<Ort::Value> inputs;
    inputs.reserve(input_count);
    for (const auto input_name : context_->input_names_) {
        if (auto it = input->inputs.find(input_name); it != input->inputs.end()) {
            inputs.push_back(std::move(it->second.value));
        } else {
            co_return absl::InvalidArgumentError(
                fmt::format("OrtModel cannot find input named {}", input_name));
        }
    }
    const size_t output_count = context_->session_.GetOutputCount();

    auto outs =
        context_->session_.Run(context_->run_options_, &context_->input_names_[0], &inputs[0],
                               input_count, &context_->output_names_[0], output_count);

    auto output = std::make_unique<OrtModelOutput>();
    for (size_t i = 0; i < output_count; ++i) {
        output->outputs.emplace(std::string(context_->output_names_[i]), std::move(outs[i]));
    }
    co_return output;
}

std::string OrtModel::info() const {
    return fmt::format("onnxruntime model loaded from {}/model.onnx", context_->dir_path_);
}

const std::vector<std::string> &OrtModel::input_names() const { return context_->input_names_s_; }

const std::vector<std::string> &OrtModel::output_names() const { return context_->output_names_s_; }

} // namespace metaspore::serving
