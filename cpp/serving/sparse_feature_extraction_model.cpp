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
#include <common/features/feature_compute_exec.h>
#include <common/features/lightweight_feature_compute_exec.h>
#include <common/features/lightweight_schema_parser.h>
#include <common/features/schema_parser.h>
#include <serving/sparse_feature_extraction_model.h>
#include <common/threadpool.h>
#include <common/utils.h>

#include <boost/core/demangle.hpp>
#include <fmt/format.h>
#include <fmt/ranges.h>

#include <filesystem>
#include <algorithm>
#include <cctype>
#include <cstdlib>

namespace metaspore::serving {
DECLARE_bool(use_lightweight_feature_compute);

class SparseFeatureExtractionModelContext {
  public:
    bool use_lightweight{true};
    FeatureComputeExec arrow_exec;
    LightweightFeatureComputeExec lightweight_exec;
    std::vector<std::string> inputs_;
    std::vector<std::string> outputs_;
};

SparseFeatureExtractionModel::SparseFeatureExtractionModel() {
    context_ = std::make_unique<SparseFeatureExtractionModelContext>();
}

SparseFeatureExtractionModel::SparseFeatureExtractionModel(SparseFeatureExtractionModel &&) =
    default;

SparseFeatureExtractionModel::~SparseFeatureExtractionModel() = default;

awaitable_status SparseFeatureExtractionModel::load(std::string dir_path) {
    auto s = co_await boost::asio::co_spawn(
        Threadpools::get_background_threadpool(),
        [this, &dir_path]() -> awaitable_status {
            auto get_env_bool = [](const char *name, bool fallback) -> bool {
                const char *str = getenv(name);
                if (!str || !*str) {
                    return fallback;
                }
                std::string v(str);
                std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) {
                    return static_cast<char>(std::tolower(c));
                });
                if (v == "1" || v == "true" || v == "yes" || v == "on") {
                    return true;
                }
                if (v == "0" || v == "false" || v == "no" || v == "off") {
                    return false;
                }
                return fallback;
            };

            context_->use_lightweight = get_env_bool(
                "METASPORE_USE_LIGHTWEIGHT_FEATURE_COMPUTE",
                FLAGS_use_lightweight_feature_compute);
            std::filesystem::path p(dir_path);
            std::filesystem::path schema_file = p / "combine_schema.txt";
            if (!std::filesystem::is_regular_file(schema_file)) {
                co_return absl::NotFoundError(fmt::format(
                    "SparseFeatureExtractionModel cannot find {}", schema_file.string()));
            }
            if (context_->use_lightweight) {
                spdlog::info("use LightweightFeatureComputeExec");
                CALL_AND_CO_RETURN_IF_STATUS_NOT_OK(
                    LightweightSchemaParser::parse(schema_file.string(), context_->lightweight_exec));
                context_->inputs_ = context_->lightweight_exec.get_input_names();
            } else {
                spdlog::info("use FeatureComputeExec");
                CALL_AND_CO_RETURN_IF_STATUS_NOT_OK(
                    FeatureSchemaParser::parse(schema_file.string(), context_->arrow_exec));
                context_->inputs_ = context_->arrow_exec.get_input_names();
            }
            context_->outputs_.push_back(
                fmt::format("{}_fe", schema_file.parent_path().filename().string()));
            spdlog::info("SparseFeatureExtractionModel loaded from {}, required inputs [{}], "
                         "producing outputs [{}]",
                         dir_path, fmt::join(context_->inputs_, ", "),
                         fmt::join(context_->outputs_, ", "));
            co_return absl::OkStatus();
        },
        boost::asio::use_awaitable);
    co_return s;
}

awaitable_result<std::unique_ptr<SparseFeatureExtractionModelOutput>>
SparseFeatureExtractionModel::do_predict(std::unique_ptr<FeatureExtractionModelInput> input) {
    auto output = std::make_unique<SparseFeatureExtractionModelOutput>();
    if (context_->use_lightweight) {
        if (context_->inputs_.empty()) {
            co_return absl::InvalidArgumentError(
                "SparseFeatureExtractionModel has no input names for lightweight exec");
        }
        std::shared_ptr<arrow::RecordBatch> batch;
        const auto &expected = context_->inputs_[0];
        auto it = input->feature_tables.find(expected);
        if (it != input->feature_tables.end()) {
            batch = it->second;
        } else if (input->feature_tables.size() == 1) {
            batch = input->feature_tables.begin()->second;
        } else {
            co_return absl::NotFoundError(
                fmt::format("SparseFeatureExtractionModel cannot find input {}", expected));
        }
        ASSIGN_RESULT_OR_CO_RETURN_NOT_OK(auto result,
                                          context_->lightweight_exec.execute(batch));
        output->values = result;
    } else {
        ASSIGN_RESULT_OR_CO_RETURN_NOT_OK(auto ctx, context_->arrow_exec.start_plan());
        for (const auto &[name, batch] : input->feature_tables) {
            CALL_AND_CO_RETURN_IF_STATUS_NOT_OK(
                context_->arrow_exec.set_input_schema(ctx, name, batch->schema()));
        }

        CALL_AND_CO_RETURN_IF_STATUS_NOT_OK(context_->arrow_exec.build_plan(ctx));

        for (const auto &[name, batch] : input->feature_tables) {
            CALL_AND_CO_RETURN_IF_STATUS_NOT_OK(context_->arrow_exec.feed_input(ctx, name, batch));
        }

        Defer _([&] { (void)context_->arrow_exec.finish_plan(ctx); });

        CO_ASSIGN_RESULT_OR_CO_RETURN_NOT_OK(auto output_result,
                                             context_->arrow_exec.execute(ctx));
        output->values = output_result;
    }
    co_return output;
}

std::string SparseFeatureExtractionModel::info() const { return ""; }

const std::vector<std::string> &SparseFeatureExtractionModel::input_names() const {
    return context_->inputs_;
}

const std::vector<std::string> &SparseFeatureExtractionModel::output_names() const {
    return context_->outputs_;
}

} // namespace metaspore::serving
