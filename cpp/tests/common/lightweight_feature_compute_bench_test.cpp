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

#include <chrono>
#include <algorithm>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <stdexcept>
#include <mutex>
#include <thread>
#include <vector>

#include <arrow/array.h>
#include <arrow/builder.h>
#include <arrow/record_batch.h>
#include <boost/asio/co_spawn.hpp>
#include <boost/asio/use_future.hpp>
#include <fmt/format.h>
#include <common/features/feature_compute_exec.h>
#include <common/features/lightweight_feature_compute_exec.h>
#include <common/features/lightweight_schema_parser.h>
#include <common/logger.h>
#include <common/features/schema_parser.h>
#include <common/test_utils.h>
#include <common/threadpool.h>
#include <common/types.h>
#include <common/utils.h>

using namespace metaspore;
using namespace metaspore::serving;

namespace {

class FeatureComputeExecuteLatencyStats {
  public:
    void observe_us(int64_t us) {
        if (us < 0) {
            return;
        }
        ensure_started();
        std::lock_guard<std::mutex> lk(mu_);
        samples_us_.push_back(us);
    }

  private:
    void ensure_started() {
        std::call_once(start_once_, [this]() {
            samples_us_.reserve(4096);
            std::thread([this]() { report_loop(); }).detach();
        });
    }

    void report_loop() {
        for (;;) {
            std::this_thread::sleep_for(std::chrono::minutes(1));
            std::vector<int64_t> snapshot;
            {
                std::lock_guard<std::mutex> lk(mu_);
                snapshot.swap(samples_us_);
            }

            if (snapshot.empty()) {
                spdlog::info("FeatureComputeBench::execute latency last_minute: count=0");
                continue;
            }

            std::sort(snapshot.begin(), snapshot.end());
            long double sum_us = 0;
            for (int64_t v : snapshot) {
                sum_us += static_cast<long double>(v);
            }
            const size_t n = snapshot.size();
            const long double avg_us = sum_us / static_cast<long double>(n);
            const size_t p99_index =
                std::min(n - 1, static_cast<size_t>((n * 99 + 99) / 100 - 1));
            const int64_t p99_us = snapshot[p99_index];
            const int64_t max_us = snapshot.back();

            spdlog::info(
                "FeatureComputeBench::execute latency last_minute: count={}, avg_ms={:.3f}, p99_ms={:.3f}, max_ms={:.3f}",
                n, static_cast<double>(avg_us) / 1000.0,
                static_cast<double>(p99_us) / 1000.0,
                static_cast<double>(max_us) / 1000.0);
        }
    }

    std::once_flag start_once_;
    std::mutex mu_;
    std::vector<int64_t> samples_us_;
};

FeatureComputeExecuteLatencyStats &GetFeatureComputeExecuteLatencyStats() {
    static FeatureComputeExecuteLatencyStats stats;
    return stats;
}

class FeatureComputeExecuteLatencyScope {
  public:
    FeatureComputeExecuteLatencyScope()
        : start_time_(std::chrono::steady_clock::now()) {}

    ~FeatureComputeExecuteLatencyScope() {
        const auto end_time = std::chrono::steady_clock::now();
        const auto elapsed_us =
            std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time_).count();
        GetFeatureComputeExecuteLatencyStats().observe_us(static_cast<int64_t>(elapsed_us));
    }

  private:
    std::chrono::steady_clock::time_point start_time_;
};

} // namespace

static std::string read_text_file(const std::string &path) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        throw std::runtime_error("failed to open file: " + path);
    }
    std::ostringstream oss;
    oss << ifs.rdbuf();
    return oss.str();
}

static std::set<std::string> collect_required_columns(const std::string &schema_source) {
    std::set<std::string> cols;
    std::istringstream is(schema_source);
    std::string line;
    while (std::getline(is, line)) {
        if (line.empty()) continue;
        std::istringstream ls(line);
        std::string token;
        while (std::getline(ls, token, '#')) {
            if (!token.empty()) cols.insert(token);
        }
    }
    return cols;
}

static std::string trim(const std::string &s) {
    size_t b = 0;
    while (b < s.size() && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
    size_t e = s.size();
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
    return s.substr(b, e - b);
}

static std::string unquote_json_string(const std::string &raw) {
    if (raw.size() >= 2 && raw.front() == '"' && raw.back() == '"') {
        std::string out;
        out.reserve(raw.size() - 2);
        for (size_t i = 1; i + 1 < raw.size(); ++i) {
            char c = raw[i];
            if (c == '\\' && i + 1 < raw.size() - 1) {
                char n = raw[++i];
                switch (n) {
                    case '"': out.push_back('"'); break;
                    case '\\': out.push_back('\\'); break;
                    case '/': out.push_back('/'); break;
                    case 'b': out.push_back('\b'); break;
                    case 'f': out.push_back('\f'); break;
                    case 'n': out.push_back('\n'); break;
                    case 'r': out.push_back('\r'); break;
                    case 't': out.push_back('\t'); break;
                    default:
                        // Keep unknown escape as-is.
                        out.push_back(n);
                        break;
                }
            } else {
                out.push_back(c);
            }
        }
        return out;
    }
    return raw;
}

// Parse flat top-level JSON object into string values.
// For non-string primitives, keep their literal text form.
static std::map<std::string, std::string> parse_flat_json_object(const std::string &json_text) {
    std::map<std::string, std::string> kv;
    std::istringstream is(json_text);
    std::string line;
    while (std::getline(is, line)) {
        line = trim(line);
        if (line.empty() || line == "{" || line == "}") continue;
        if (!line.empty() && line.back() == ',') line.pop_back();
        size_t colon = line.find(':');
        if (colon == std::string::npos) continue;
        std::string key = trim(line.substr(0, colon));
        std::string val = trim(line.substr(colon + 1));
        key = unquote_json_string(key);
        val = unquote_json_string(val);
        if (!key.empty()) kv.emplace(std::move(key), std::move(val));
    }
    return kv;
}

static std::shared_ptr<arrow::RecordBatch> make_batch(int64_t rows,
                                                      const std::string &schema_source,
                                                      const std::string &sample_json_path) {
    const std::string json_text = read_text_file(sample_json_path);
    const auto kv = parse_flat_json_object(json_text);

    const auto required_cols = collect_required_columns(schema_source);
    std::vector<std::shared_ptr<arrow::Array>> arrays;
    arrays.reserve(required_cols.size());
    arrow::FieldVector fields;
    fields.reserve(required_cols.size());
    for (const auto &col : required_cols) {
        std::string val;
        auto it = kv.find(col);
        if (it != kv.end()) val = it->second;
        arrow::StringBuilder builder;
        builder.Reserve(rows);
        for (int64_t r = 0; r < rows; ++r) {
            builder.Append(val);
        }
        auto array_result = builder.Finish();
        if (!array_result.ok()) {
            throw std::runtime_error(array_result.status().ToString());
        }
        arrays.push_back(*array_result);
        fields.push_back(arrow::field(col, arrow::utf8()));
    }
    auto schema = std::make_shared<arrow::Schema>(std::move(fields));
    return arrow::RecordBatch::Make(schema, rows, std::move(arrays));
}

static double bench_lightweight(const std::string &schema_source,
                                const std::shared_ptr<arrow::RecordBatch> &batch,
                                int warmup, int iters) {
    LightweightFeatureComputeExec compute;
    int feature_count = 0;
    std::istringstream is(schema_source);
    auto status = LightweightSchemaParser::parse_hash_and_combine(is, compute, feature_count);
    EXPECT_TRUE(status.ok()) << status.ToString();

    for (int i = 0; i < warmup; ++i) {
        FeatureComputeExecuteLatencyScope latency_scope;
        auto r = compute.execute(batch);
        EXPECT_TRUE(r.ok()) << r.status().ToString();
    }

    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        FeatureComputeExecuteLatencyScope latency_scope;
        auto r = compute.execute(batch);
        EXPECT_TRUE(r.ok()) << r.status().ToString();
    }
    auto end = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    return ms / iters;
}

static double bench_arrow_exec(const std::string &schema_source,
                               const std::shared_ptr<arrow::RecordBatch> &batch,
                               int warmup, int iters) {
    FeatureComputeExec exec;
    auto status = exec.add_source("t");
    EXPECT_TRUE(status.ok()) << status.ToString();
    int feature_count = 0;
    std::istringstream is(schema_source);
    status = FeatureSchemaParser::parse_hash_and_combine(is, exec, feature_count);
    EXPECT_TRUE(status.ok()) << status.ToString();

    auto run_once = [&]() -> metaspore::status {
        FeatureComputeExecuteLatencyScope latency_scope;
        auto fn = [&]() -> awaitable_result<std::shared_ptr<arrow::RecordBatch>> {
            ASSIGN_RESULT_OR_CO_RETURN_NOT_OK(auto ctx, exec.start_plan());
            CALL_AND_CO_RETURN_IF_STATUS_NOT_OK(exec.set_input_schema(ctx, "t", batch->schema()));
            CALL_AND_CO_RETURN_IF_STATUS_NOT_OK(exec.build_plan(ctx));
            CALL_AND_CO_RETURN_IF_STATUS_NOT_OK(exec.feed_input(ctx, "t", batch));
            Defer _([&] { (void)exec.finish_plan(ctx); });
            CO_ASSIGN_RESULT_OR_CO_RETURN_NOT_OK(auto result, exec.execute(ctx));
            co_return result;
        };
        auto &tp = Threadpools::get_background_threadpool();
        auto fut = boost::asio::co_spawn(tp, std::move(fn), boost::asio::use_future);
        auto result = fut.get();
        return result.status();
    };

    for (int i = 0; i < warmup; ++i) {
        auto s = run_once();
        EXPECT_TRUE(s.ok()) << s.ToString();
    }

    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        auto s = run_once();
        EXPECT_TRUE(s.ok()) << s.ToString();
    }
    auto end = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    return ms / iters;
}

TEST(LightweightFeatureComputeBenchTest, CompareWithArrowExec) {
    const std::string schema_path =
        "cpp/tests/common/testdata/lightweight_feature_compute_bench_schema.txt";
    const std::string sample_json_path =
        "cpp/tests/common/testdata/lightweight_feature_compute_bench_sample.json";
    const std::string schema_source = read_text_file(schema_path);
    const int rules = static_cast<int>(std::count(schema_source.begin(), schema_source.end(), '\n'));
    const int warmup = 2;
    const int iters = 5;
    fmt::print("schema_source_file={} rules={}\n", schema_path, rules);

    std::vector<int64_t> batch_sizes = {32, 64, 128, 1024, 4096};
    for (auto rows : batch_sizes) {
        auto batch = make_batch(rows, schema_source, sample_json_path);
        double lw_ms = bench_lightweight(schema_source, batch, warmup, iters);
        double arrow_ms = bench_arrow_exec(schema_source, batch, warmup, iters);
        fmt::print("rows {} cols {} rules {} | lightweight {:.3f} ms | arrow_exec {:.3f} ms\n",
                   rows, batch->num_columns(), rules, lw_ms, arrow_ms);
    }
}

int main(int argc, char **argv) { return run_all_tests(argc, argv); }
