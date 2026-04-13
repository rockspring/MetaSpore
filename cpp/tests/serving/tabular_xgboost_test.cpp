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

#include <arrow/api.h>
#include <arrow/builder.h>
#include <range/v3/all.hpp>

#include <serving/feature_extraction_model_input.h>
#include <serving/ort_model.h>
#include <serving/tabular_model.h>
#include <common/test_utils.h>
#include <common/types.h>

#include <thread>

using namespace ranges;
using namespace metaspore;
using namespace metaspore::serving;
using namespace std::string_literals;
using namespace arrow;

#define ASSERT_STATUS_OK_COROUTINE(status)                                                         \
    EXPECT_TRUE(status.ok()) << status;                                                            \
    if (!status.ok()) {                                                                            \
        co_return;                                                                                 \
    }

#define ASSERT_TRUE_COROUTINE(expr)                                                                \
    EXPECT_TRUE(expr);                                                                             \
    if (!expr) {                                                                                   \
        co_return;                                                                                 \
    }

TEST(TabularXGBoostModelTestSuite, TabularXGBoostModelTest) {
    boost::asio::co_spawn(
        Threadpools::get_background_threadpool(),
        []() -> awaitable<void> {
            TabularModel model;
            auto status = co_await model.load("xgboost_model");
            ASSERT_STATUS_OK_COROUTINE(status);

            // construct input record batch
            auto schema_fields =
                views::iota(0, 10) | views::transform([](int x) {
                    return arrow::field(fmt::format("field_{}", x), arrow::float32());
                }) |
                to<std::vector>();
            auto schema_obj = arrow::schema(std::move(schema_fields));
            std::vector<float> values = {0.6558618f,  0.13005558f, 0.03510657f, 0.23048967f,
                                         0.63329154f, 0.43201634f, 0.5795548f,  0.5384891f,
                                         0.9612295f,  0.39274803f};
            arrow::ArrayVector arrays;
            for (int i = 0; i < 10; i++) {
                arrow::FloatBuilder builder;
                ASSERT_STATUS_OK_COROUTINE(builder.Append(values[i]));
                auto result = builder.Finish();
                ASSERT_STATUS_OK_COROUTINE(result.status());
                arrays.push_back(std::move(result).ValueUnsafe());
            }
            auto rb = arrow::RecordBatch::Make(schema_obj, 1, std::move(arrays));
            fmt::print("Input: {}\n", rb->ToString());
            auto fe_input = std::make_unique<FeatureExtractionModelInput>();
            fe_input->feature_tables["input"] = rb;
            auto result = co_await model.predict(std::move(fe_input));
            ASSERT_STATUS_OK_COROUTINE(result.status());
            auto ort_output = dynamic_cast<OrtModelOutput *>(result->get());
            ASSERT_TRUE_COROUTINE(ort_output);
            for (const auto &[name, value] : ort_output->outputs) {
                auto shape = value.GetTensorTypeAndShapeInfo().GetShape();
                fmt::print("Ort Output Name {}, shape {}\n", name, fmt::join(shape, ","));
                TensorPrint::print_tensor<float>(value);
            }
        },
        boost::asio::detached);
}

int main(int argc, char **argv) { return run_all_tests(argc, argv); }
