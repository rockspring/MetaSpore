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

// Tests for LightweightFeatureComputeExec list<string> support.
// Each test builds the same RecordBatch, runs both the Arrow-based
// FeatureComputeExec and LightweightFeatureComputeExec, then asserts
// that every output column is equal.

#include <sstream>
#include <string>
#include <vector>

#include <arrow/api.h>
#include <arrow/compute/api.h>
#include <arrow/testing/gtest_util.h>
#include <boost/asio/co_spawn.hpp>
#include <boost/asio/use_future.hpp>

#include <common/features/feature_compute_exec.h>
#include <common/features/feature_compute_funcs.h>
#include <common/features/lightweight_feature_compute_exec.h>
#include <common/features/lightweight_schema_parser.h>
#include <common/features/schema_parser.h>
#include <common/test_utils.h>
#include <common/threadpool.h>
#include <common/types.h>
#include <common/utils.h>

using namespace metaspore;
using namespace metaspore::serving;
using namespace std::string_literals;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Run FeatureComputeExec (Arrow-based) synchronously and return the result.
static std::shared_ptr<arrow::RecordBatch>
run_arrow_exec(const std::string &schema_source,
               const std::shared_ptr<arrow::RecordBatch> &batch) {
    FeatureComputeExec exec;
    EXPECT_TRUE(exec.add_source("t").ok());
    int feature_count = 0;
    std::istringstream is(schema_source);
    EXPECT_TRUE(FeatureSchemaParser::parse_hash_and_combine(is, exec, feature_count).ok());

    std::shared_ptr<arrow::RecordBatch> result;
    auto fn = [&]() -> awaitable_result<std::shared_ptr<arrow::RecordBatch>> {
        ASSIGN_RESULT_OR_CO_RETURN_NOT_OK(auto ctx, exec.start_plan());
        CALL_AND_CO_RETURN_IF_STATUS_NOT_OK(exec.set_input_schema(ctx, "t", batch->schema()));
        CALL_AND_CO_RETURN_IF_STATUS_NOT_OK(exec.build_plan(ctx));
        CALL_AND_CO_RETURN_IF_STATUS_NOT_OK(exec.feed_input(ctx, "t", batch));
        Defer _([&] { (void)exec.finish_plan(ctx); });
        CO_ASSIGN_RESULT_OR_CO_RETURN_NOT_OK(auto rb, exec.execute(ctx));
        co_return rb;
    };
    auto fut = boost::asio::co_spawn(Threadpools::get_background_threadpool(),
                                     std::move(fn), boost::asio::use_future);
    auto res = fut.get();
    EXPECT_TRUE(res.ok()) << res.status().ToString();
    return *res;
}

// Run LightweightFeatureComputeExec and return the result.
static std::shared_ptr<arrow::RecordBatch>
run_lightweight(const std::string &schema_source,
                const std::shared_ptr<arrow::RecordBatch> &batch) {
    LightweightFeatureComputeExec compute;
    int feature_count = 0;
    std::istringstream is(schema_source);
    EXPECT_TRUE(
        LightweightSchemaParser::parse_hash_and_combine(is, compute, feature_count).ok());
    auto res = compute.execute(batch);
    EXPECT_TRUE(res.ok()) << res.status().ToString();
    return *res;
}

// Assert that two RecordBatches have the same number of columns and that
// every column pair is equal (using Arrow's Array::Equals).
static void assert_batches_equal(const std::shared_ptr<arrow::RecordBatch> &arrow_rb,
                                 const std::shared_ptr<arrow::RecordBatch> &lw_rb) {
    ASSERT_EQ(arrow_rb->num_columns(), lw_rb->num_columns());
    ASSERT_EQ(arrow_rb->num_rows(), lw_rb->num_rows());
    for (int c = 0; c < arrow_rb->num_columns(); ++c) {
        EXPECT_TRUE(arrow_rb->column(c)->Equals(lw_rb->column(c)))
            << "Column " << c << " differs.\n"
            << "Arrow:       " << arrow_rb->column(c)->ToString() << "\n"
            << "Lightweight: " << lw_rb->column(c)->ToString();
    }
}

// ---------------------------------------------------------------------------
// Batch builders
// ---------------------------------------------------------------------------

// Build a RecordBatch with one string column.
static std::shared_ptr<arrow::RecordBatch>
make_string_batch(const std::string &col_name,
                  const std::vector<std::string> &values,
                  const std::vector<bool> &nulls = {}) {
    arrow::StringBuilder builder;
    for (size_t i = 0; i < values.size(); ++i) {
        if (!nulls.empty() && nulls[i])
            EXPECT_TRUE(builder.AppendNull().ok());
        else
            EXPECT_TRUE(builder.Append(values[i]).ok());
    }
    EXPECT_OK_AND_ASSIGN(auto arr, builder.Finish());
    auto schema = arrow::schema({arrow::field(col_name, arrow::utf8())});
    return arrow::RecordBatch::Make(schema, (int64_t)values.size(), {arr});
}

// Build a RecordBatch with one list<string> column.
static std::shared_ptr<arrow::RecordBatch>
make_list_string_batch(const std::string &col_name,
                       const std::vector<std::vector<std::string>> &rows,
                       const std::vector<bool> &row_nulls = {}) {
    auto str_builder = std::make_shared<arrow::StringBuilder>();
    arrow::ListBuilder list_builder(arrow::default_memory_pool(), str_builder,
                                    std::make_shared<arrow::ListType>(arrow::utf8()));
    for (size_t i = 0; i < rows.size(); ++i) {
        if (!row_nulls.empty() && row_nulls[i]) {
            EXPECT_TRUE(list_builder.AppendNull().ok());
            continue;
        }
        EXPECT_TRUE(list_builder.Append().ok());
        for (const auto &s : rows[i])
            EXPECT_TRUE(str_builder->Append(s).ok());
    }
    EXPECT_OK_AND_ASSIGN(auto arr, list_builder.Finish());
    auto schema = arrow::schema({arrow::field(col_name, std::make_shared<arrow::ListType>(arrow::utf8()))});
    return arrow::RecordBatch::Make(schema, (int64_t)rows.size(), {arr});
}

// Merge two single-column batches into one two-column batch.
static std::shared_ptr<arrow::RecordBatch>
merge_batches(const std::shared_ptr<arrow::RecordBatch> &a,
              const std::shared_ptr<arrow::RecordBatch> &b) {
    EXPECT_EQ(a->num_rows(), b->num_rows());
    arrow::FieldVector fields = {a->schema()->field(0), b->schema()->field(0)};
    auto schema = std::make_shared<arrow::Schema>(fields);
    return arrow::RecordBatch::Make(schema, a->num_rows(),
                                    {a->column(0), b->column(0)});
}

// ---------------------------------------------------------------------------
// Case 1: single string column (existing behaviour, regression guard)
// ---------------------------------------------------------------------------
TEST(LightweightListStringTest, SingleStringColumn) {
    auto batch = make_string_batch("demand_pkgname",
                                   {"com.app.a", "com.app.b", "", "com.app.d"},
                                   {false, false, false, false});
    const std::string schema = "demand_pkgname\n";
    auto arrow_rb = run_arrow_exec(schema, batch);
    auto lw_rb    = run_lightweight(schema, batch);
    assert_batches_equal(arrow_rb, lw_rb);
}

// ---------------------------------------------------------------------------
// Case 2: single string column with nulls
// ---------------------------------------------------------------------------
TEST(LightweightListStringTest, SingleStringColumnWithNulls) {
    auto batch = make_string_batch("col_a",
                                   {"v1", "", "v3", "v4"},
                                   {false, false, true, false});
    const std::string schema = "col_a\n";
    auto arrow_rb = run_arrow_exec(schema, batch);
    auto lw_rb    = run_lightweight(schema, batch);
    assert_batches_equal(arrow_rb, lw_rb);
}

// ---------------------------------------------------------------------------
// Case 3: single list<string> column
// ---------------------------------------------------------------------------
TEST(LightweightListStringTest, SingleListStringColumn) {
    auto batch = make_list_string_batch(
        "duf_outer_shopee_pur_top5_list",
        {{"app1", "app2", "app3"}, {"app4"}, {"app5", "app6"}});
    const std::string schema = "duf_outer_shopee_pur_top5_list\n";
    auto arrow_rb = run_arrow_exec(schema, batch);
    auto lw_rb    = run_lightweight(schema, batch);
    assert_batches_equal(arrow_rb, lw_rb);
}

// ---------------------------------------------------------------------------
// Case 4: single list<string> column with null rows and empty list rows
// ---------------------------------------------------------------------------
TEST(LightweightListStringTest, SingleListStringColumnWithNulls) {
    auto batch = make_list_string_batch(
        "top5_list",
        {{"a", "b"}, {}, {"c"}, {"d", "e", "f"}, {}},
        {false, false, true, false, false});
    const std::string schema = "top5_list\n";
    auto arrow_rb = run_arrow_exec(schema, batch);
    auto lw_rb    = run_lightweight(schema, batch);
    assert_batches_equal(arrow_rb, lw_rb);
}

// ---------------------------------------------------------------------------
// Case 5: string # string cross (existing behaviour, regression guard)
// ---------------------------------------------------------------------------
TEST(LightweightListStringTest, StringCrossString) {
    auto ba = make_string_batch("cnt_30_60d", {"10", "20", "30"});
    auto bb = make_string_batch("demand_pkgname", {"com.a", "com.b", "com.c"});
    auto batch = merge_batches(ba, bb);
    const std::string schema = "cnt_30_60d#demand_pkgname\n";
    auto arrow_rb = run_arrow_exec(schema, batch);
    auto lw_rb    = run_lightweight(schema, batch);
    assert_batches_equal(arrow_rb, lw_rb);
}

// ---------------------------------------------------------------------------
// Case 6: string # string with nulls
// ---------------------------------------------------------------------------
TEST(LightweightListStringTest, StringCrossStringWithNulls) {
    auto ba = make_string_batch("col_a", {"v1", "", "v3"}, {false, false, true});
    auto bb = make_string_batch("col_b", {"x1", "x2", "x3"});
    auto batch = merge_batches(ba, bb);
    const std::string schema = "col_a#col_b\n";
    auto arrow_rb = run_arrow_exec(schema, batch);
    auto lw_rb    = run_lightweight(schema, batch);
    assert_batches_equal(arrow_rb, lw_rb);
}

// ---------------------------------------------------------------------------
// Case 7: list<string> # string cross
// ---------------------------------------------------------------------------
TEST(LightweightListStringTest, ListStringCrossString) {
    auto ba = make_list_string_batch(
        "duf_outer_shopee_pur_top5_list",
        {{"app1", "app2", "app3"}, {"app4", "app5"}, {"app6"}});
    auto bb = make_string_batch("demand_pkgname", {"com.a", "com.b", "com.c"});
    auto batch = merge_batches(ba, bb);
    const std::string schema = "duf_outer_shopee_pur_top5_list#demand_pkgname\n";
    auto arrow_rb = run_arrow_exec(schema, batch);
    auto lw_rb    = run_lightweight(schema, batch);
    assert_batches_equal(arrow_rb, lw_rb);
}

// ---------------------------------------------------------------------------
// Case 8: list<string> # string with null rows
// ---------------------------------------------------------------------------
TEST(LightweightListStringTest, ListStringCrossStringWithNulls) {
    auto ba = make_list_string_batch(
        "top5_list",
        {{"a", "b"}, {"c"}, {"d"}},
        {false, true, false});
    auto bb = make_string_batch("pkg", {"x", "y", ""});
    auto batch = merge_batches(ba, bb);
    const std::string schema = "top5_list#pkg\n";
    auto arrow_rb = run_arrow_exec(schema, batch);
    auto lw_rb    = run_lightweight(schema, batch);
    assert_batches_equal(arrow_rb, lw_rb);
}

// ---------------------------------------------------------------------------
// Case 9: string # list<string> cross (reversed order)
// ---------------------------------------------------------------------------
TEST(LightweightListStringTest, StringCrossListString) {
    auto ba = make_string_batch("demand_pkgname", {"com.a", "com.b", "com.c"});
    auto bb = make_list_string_batch(
        "top5_list",
        {{"app1", "app2"}, {"app3"}, {"app4", "app5", "app6"}});
    auto batch = merge_batches(ba, bb);
    const std::string schema = "demand_pkgname#top5_list\n";
    auto arrow_rb = run_arrow_exec(schema, batch);
    auto lw_rb    = run_lightweight(schema, batch);
    assert_batches_equal(arrow_rb, lw_rb);
}

// ---------------------------------------------------------------------------
// Case 10: list<string> # list<string> cross (cartesian product)
// ---------------------------------------------------------------------------
TEST(LightweightListStringTest, ListStringCrossListString) {
    auto ba = make_list_string_batch(
        "list_a",
        {{"a1", "a2"}, {"b1", "b2", "b3"}, {"c1"}});
    auto bb = make_list_string_batch(
        "list_b",
        {{"x1", "x2", "x3"}, {"y1"}, {"z1", "z2"}});
    auto batch = merge_batches(ba, bb);
    const std::string schema = "list_a#list_b\n";
    auto arrow_rb = run_arrow_exec(schema, batch);
    auto lw_rb    = run_lightweight(schema, batch);
    assert_batches_equal(arrow_rb, lw_rb);
}

// ---------------------------------------------------------------------------
// Case 11: list<string> # list<string> with null rows
// ---------------------------------------------------------------------------
TEST(LightweightListStringTest, ListStringCrossListStringWithNulls) {
    auto ba = make_list_string_batch(
        "list_a",
        {{"a1", "a2"}, {"b1"}, {"c1", "c2"}},
        {false, true, false});
    auto bb = make_list_string_batch(
        "list_b",
        {{"x1"}, {"y1", "y2"}, {}},
        {false, false, false});
    auto batch = merge_batches(ba, bb);
    const std::string schema = "list_a#list_b\n";
    auto arrow_rb = run_arrow_exec(schema, batch);
    auto lw_rb    = run_lightweight(schema, batch);
    assert_batches_equal(arrow_rb, lw_rb);
}

// ---------------------------------------------------------------------------
// Case 12: multiple specs in one schema (mixed types)
// ---------------------------------------------------------------------------
TEST(LightweightListStringTest, MultipleSpecsMixedTypes) {
    // Build a 3-column batch: string, list<string>, string
    auto ba = make_string_batch("pkg", {"com.a", "com.b", "com.c"});
    auto bb = make_list_string_batch("top5", {{"a1", "a2"}, {"b1"}, {"c1", "c2", "c3"}});
    auto bc = make_string_batch("cnt", {"10", "20", "30"});

    arrow::FieldVector fields = {ba->schema()->field(0),
                                 bb->schema()->field(0),
                                 bc->schema()->field(0)};
    auto schema = std::make_shared<arrow::Schema>(fields);
    auto batch = arrow::RecordBatch::Make(schema, 3,
                                          {ba->column(0), bb->column(0), bc->column(0)});

    // Three specs: single string, single list<string>, list<string>#string
    const std::string schema_src =
        "pkg\n"
        "top5\n"
        "top5#pkg\n";

    auto arrow_rb = run_arrow_exec(schema_src, batch);
    auto lw_rb    = run_lightweight(schema_src, batch);
    assert_batches_equal(arrow_rb, lw_rb);
}

int main(int argc, char **argv) { return run_all_tests(argc, argv); }
