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
#include <sstream>
#include <string>
#include <vector>

#include <arrow/array.h>
#include <arrow/builder.h>
#include <arrow/record_batch.h>
#include <boost/asio/co_spawn.hpp>
#include <boost/asio/use_future.hpp>
#include <fmt/format.h>
#include <common/features/feature_compute_exec.h>
#include <common/features/lightweight_feature_compute.h>
#include <common/features/schema_parser.h>
#include <common/test_utils.h>
#include <common/threadpool.h>
#include <common/types.h>
#include <common/utils.h>

using namespace metaspore;
using namespace metaspore::serving;

static std::string make_schema_source(int rules, int columns) {
    std::ostringstream oss;
    int single = std::min(rules, columns - 1);
    for (int i = 0; i < single; ++i) {
        oss << "c" << i << "\n";
    }
    for (int i = single; i < rules; ++i) {
        int a = i % columns;
        int b = (i + 1) % columns;
        oss << "c" << a << "#c" << b << "\n";
    }
    return oss.str();
}

static std::shared_ptr<arrow::RecordBatch> make_batch(int64_t rows, int64_t columns) {
    std::vector<std::shared_ptr<arrow::Array>> arrays;
    arrays.reserve(columns);
    arrow::FieldVector fields;
    fields.reserve(columns);
    for (int64_t c = 0; c < columns; ++c) {
        arrow::StringBuilder builder;
        builder.Reserve(rows);
        for (int64_t r = 0; r < rows; ++r) {
            std::string v = "vvvvvv" + std::to_string(c) + "_" + std::to_string(r);
            builder.Append(v);
        }
        auto array_result = builder.Finish();
        if (!array_result.ok()) {
            throw std::runtime_error(array_result.status().ToString());
        }
        arrays.push_back(*array_result);
        fields.push_back(arrow::field("c" + std::to_string(c), arrow::utf8()));
    }
    auto schema = std::make_shared<arrow::Schema>(std::move(fields));
    return arrow::RecordBatch::Make(schema, rows, std::move(arrays));
}

static double bench_lightweight(const std::string &schema_source,
                                const std::shared_ptr<arrow::RecordBatch> &batch,
                                int warmup, int iters) {
    LightweightFeatureCompute compute;
    int feature_count = 0;
    std::istringstream is(schema_source);
    auto status = compute.parse_schema(is, feature_count);
    EXPECT_TRUE(status.ok()) << status.ToString();

    for (int i = 0; i < warmup; ++i) {
        auto r = compute.execute(batch);
        EXPECT_TRUE(r.ok()) << r.status().ToString();
    }

    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
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
    const int rules = 341;
    const int columns = 340;
    const int warmup = 2;
    const int iters = 5;
    auto schema_source = make_schema_source(rules, columns);
    fmt::print("schema_source is\n{}\n", schema_source);

    std::vector<int64_t> batch_sizes = {32, 64, 128, 1024, 4096};
    for (auto rows : batch_sizes) {
        auto batch = make_batch(rows, columns);
        double lw_ms = bench_lightweight(schema_source, batch, warmup, iters);
        double arrow_ms = bench_arrow_exec(schema_source, batch, warmup, iters);
        fmt::print("rows {} cols {} rules {} | lightweight {:.3f} ms | arrow_exec {:.3f} ms\n",
                   rows, columns, rules, lw_ms, arrow_ms);
    }
}

int main(int argc, char **argv) { return run_all_tests(argc, argv); }
