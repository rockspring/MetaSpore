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

#include <sstream>
#include <stdexcept>
#include <common/arrow/arrow_helpers.h>
#include <common/features/schema_parser.h>
#include <common/threadpool.h>
#include <common/utils.h>
#include <metaspore/stack_trace_utils.h>
#include <metaspore/sparse_feature_extractor.h>
#include <boost/asio/use_future.hpp>
#include <spdlog/spdlog.h>

namespace metaspore {

DECLARE_bool(use_lightweight_feature_compute);

SparseFeatureExtractor::SparseFeatureExtractor(const std::string &source_table_name,
                                               const std::string &schema_source)
    : source_table_name_(source_table_name)
    , schema_source_(schema_source)
    , feature_count_(0)
    , use_lightweight_(FLAGS_use_lightweight_feature_compute) {
    std::istringstream stream(schema_source);
    status the_status;
    if (use_lightweight_) {
        the_status = lightweight_executor_.parse_schema(stream, feature_count_);
    } else {
        the_status = arrow_executor_.add_source(source_table_name_);
        if (the_status.ok()) {
            the_status = FeatureSchemaParser::parse_hash_and_combine(
                stream, arrow_executor_, feature_count_);
        }
    }
    check_construct(the_status);
}

std::tuple<std::vector<uint64_t>, std::vector<uint64_t>>
SparseFeatureExtractor::extract(std::shared_ptr<arrow::RecordBatch> batch) {
    std::shared_ptr<arrow::RecordBatch> result_batch;
    if (use_lightweight_) {
        auto result = lightweight_executor_.execute(batch);
        check_extract(result.status());
        result_batch = *result;
    } else {
        auto fn = [&]() -> awaitable_result<std::shared_ptr<arrow::RecordBatch>> {
            ASSIGN_RESULT_OR_CO_RETURN_NOT_OK(auto ctx, arrow_executor_.start_plan());
            CALL_AND_CO_RETURN_IF_STATUS_NOT_OK(
                arrow_executor_.set_input_schema(ctx, source_table_name_, batch->schema()));
            CALL_AND_CO_RETURN_IF_STATUS_NOT_OK(arrow_executor_.build_plan(ctx));
            CALL_AND_CO_RETURN_IF_STATUS_NOT_OK(
                arrow_executor_.feed_input(ctx, source_table_name_, batch));
            Defer _([&] { (void)arrow_executor_.finish_plan(ctx); });
            CO_ASSIGN_RESULT_OR_CO_RETURN_NOT_OK(auto result, arrow_executor_.execute(ctx));
            co_return result;
        };
        auto &tp = Threadpools::get_background_threadpool();
        auto fut = boost::asio::co_spawn(tp, std::move(fn), boost::asio::use_future);
        auto result = fut.get();
        check_extract(result.status());
        result_batch = *result;
    }
    std::vector<uint64_t> indices;
    std::vector<uint64_t> offsets;
    auto r = flatten_arrow_batch<std::vector>(result_batch, indices, offsets);
    check_extract(r.status());
    return std::make_tuple(std::move(indices), std::move(offsets));
}

void SparseFeatureExtractor::check_construct(const status &the_status) {
    check_status(the_status, "Fail to construct SparseFeatureExtractor for \"" + source_table_name_ + "\".");
}

void SparseFeatureExtractor::check_extract(const status &the_status) {
    check_status(the_status, "Fail to extract sparse features for \"" + source_table_name_ + "\".");
}

void SparseFeatureExtractor::check_status(const status &the_status, const std::string &message) {
    if (!the_status.ok()) {
        std::string serr;
        serr.append(message);
        serr.append(" ");
        serr.append(the_status.ToString());
        serr.append("\n\n");
        serr.append(GetStackTrace());
        spdlog::error(serr);
        throw std::runtime_error(serr);
    }
}

} // namespace metaspore
