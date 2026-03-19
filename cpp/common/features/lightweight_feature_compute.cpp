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

#include <common/features/lightweight_feature_compute.h>

#include <cctype>
#include <cstdlib>
#include <future>
#include <thread>
#include <unordered_map>

#include <arrow/array.h>
#include <arrow/builder.h>
#include <arrow/status.h>

#include <common/hash_utils.h>
#include <common/arrow/arrow_status.h>
#include <common/threadpool.h>

#include <boost/asio/post.hpp>

namespace metaspore {
DECLARE_uint64(lightweight_min_parallel_rules);
DECLARE_uint64(lightweight_max_workers);

static inline bool is_ident_char(char c) {
    return std::isalnum(static_cast<unsigned char>(c)) || c == '_';
}

static inline void trim_schema_line(std::string &line) {
    auto is_trim = [](char c) {
        return c == '#' || c == ' ' || c == '\t' || c == '\n' || c == '\r';
    };
    while (!line.empty() && is_trim(line.front())) {
        line.erase(line.begin());
    }
    while (!line.empty() && is_trim(line.back())) {
        line.pop_back();
    }
}

static bool parse_combine_line(const std::string &line, std::vector<std::string> &out_columns) {
    out_columns.clear();
    std::string token;
    for (size_t i = 0; i <= line.size(); ++i) {
        if (i == line.size() || line[i] == '#') {
            if (token.empty()) {
                return false;
            }
            for (char c : token) {
                if (!is_ident_char(c)) {
                    return false;
                }
            }
            out_columns.push_back(token);
            token.clear();
        } else {
            token.push_back(line[i]);
        }
    }
    return !out_columns.empty();
}

status LightweightFeatureCompute::parse_schema(std::istream &is, int &feature_count) {
    specs_.clear();
    feature_count = 0;

    std::string line;
    std::vector<std::string> columns;
    while (std::getline(is, line)) {
        trim_schema_line(line);
        if (line.empty()) {
            break;
        }

        if (!parse_combine_line(line, columns)) {
            return absl::InvalidArgumentError("Parsing combine rule failed " + line);
        }

        FeatureSpec spec;
        spec.columns = columns;
        spec.seeds.reserve(columns.size());
        for (const auto &name : columns) {
            spec.seeds.push_back(BKDRHashWithEqualPostfix(name.c_str(), name.size(), 0));
        }
        specs_.push_back(std::move(spec));
        feature_count++;
    }
    return absl::OkStatus();
}

static status ensure_string_array(const std::shared_ptr<arrow::Array> &array,
                                  std::shared_ptr<arrow::StringArray> &out) {
    out = std::dynamic_pointer_cast<arrow::StringArray>(array);
    if (!out) {
        return absl::InvalidArgumentError("LightweightFeatureCompute only supports string input");
    }
    return absl::OkStatus();
}

result<std::shared_ptr<arrow::RecordBatch>>
LightweightFeatureCompute::execute(const std::shared_ptr<arrow::RecordBatch> &batch) const {
    if (!batch) {
        return absl::InvalidArgumentError("LightweightFeatureCompute input batch is null");
    }

    const int64_t rows = batch->num_rows();
    const int64_t cols = batch->num_columns();
    std::vector<std::shared_ptr<arrow::Array>> output_columns(specs_.size());
    arrow::FieldVector fields(specs_.size());

    std::unordered_map<std::string, int> name_to_index;
    name_to_index.reserve((size_t)cols);
    for (int64_t i = 0; i < cols; ++i) {
        name_to_index.emplace(batch->schema()->field(i)->name(), (int)i);
    }

    std::vector<std::shared_ptr<arrow::StringArray>> column_cache((size_t)cols);
    std::vector<std::vector<int>> spec_indices;
    spec_indices.reserve(specs_.size());

    for (size_t spec_idx = 0; spec_idx < specs_.size(); ++spec_idx) {
        const auto &spec = specs_[spec_idx];
        if (spec.columns.empty()) {
            return absl::InvalidArgumentError("LightweightFeatureCompute feature spec is empty");
        }

        std::vector<int> indices;
        indices.reserve(spec.columns.size());
        for (const auto &name : spec.columns) {
            auto it = name_to_index.find(name);
            if (it == name_to_index.end()) {
                return absl::NotFoundError("LightweightFeatureCompute cannot find column " + name);
            }
            int col_index = it->second;
            if (!column_cache[(size_t)col_index]) {
                std::shared_ptr<arrow::StringArray> string_array;
                CALL_AND_RETURN_IF_STATUS_NOT_OK(
                    ensure_string_array(batch->column(col_index), string_array));
                column_cache[(size_t)col_index] = std::move(string_array);
            }
            indices.push_back(col_index);
        }
        spec_indices.push_back(std::move(indices));
    }

    auto compute_one = [&](size_t spec_idx) -> status {
        const auto &spec = specs_[spec_idx];
        const auto &indices = spec_indices[spec_idx];
        std::vector<std::shared_ptr<arrow::StringArray>> arrays;
        arrays.reserve(indices.size());
        for (int col_index : indices) {
            arrays.push_back(column_cache[(size_t)col_index]);
        }

        auto to_absl = [](const arrow::Status &s) -> status {
            return ArrowStatusToAbsl::arrow_status_to_absl(s);
        };

        if (spec.columns.size() == 1) {
            arrow::UInt64Builder builder;
            auto s = builder.Reserve(rows);
            if (!s.ok()) {
                return to_absl(s);
            }
            auto &arr = arrays[0];
            const uint64_t seed = spec.seeds[0];
            for (int64_t i = 0; i < rows; ++i) {
                if (arr->IsNull(i)) {
                    s = builder.AppendNull();
                    if (!s.ok()) {
                        return to_absl(s);
                    }
                    continue;
                }
                auto view = arr->GetView(i);
                if (view.empty()) {
                    s = builder.AppendNull();
                    if (!s.ok()) {
                        return to_absl(s);
                    }
                    continue;
                }
                uint64_t hash = BKDRHash(view.data(), view.length(), 0);
                uint64_t out = BKDRHashOneField(seed, hash);
                s = builder.Append(out);
                if (!s.ok()) {
                    return to_absl(s);
                }
            }
            auto array_result = builder.Finish();
            if (!array_result.ok()) {
                return to_absl(array_result.status());
            }
            auto array = *array_result;
            output_columns[spec_idx] = array;
            fields[spec_idx] = arrow::field("f" + std::to_string(spec_idx), arrow::uint64());
        } else {
            auto value_builder = std::make_shared<arrow::UInt64Builder>();
            arrow::ListBuilder builder(arrow::default_memory_pool(), value_builder,
                                       std::make_shared<arrow::ListType>(arrow::uint64()));
            auto s = builder.Reserve(rows);
            if (!s.ok()) {
                return to_absl(s);
            }
            s = value_builder->Reserve(rows);
            if (!s.ok()) {
                return to_absl(s);
            }

            for (int64_t i = 0; i < rows; ++i) {
                bool any_null = false;
                bool has_value = false;
                uint64_t combined = 0;

                for (size_t j = 0; j < arrays.size(); ++j) {
                    const auto &arr = arrays[j];
                    if (arr->IsNull(i)) {
                        any_null = true;
                        break;
                    }
                    auto view = arr->GetView(i);
                    if (view.empty()) {
                        any_null = true;
                        break;
                    }
                    uint64_t hash = BKDRHash(view.data(), view.length(), 0);
                    uint64_t value = BKDRHashOneField(spec.seeds[j], hash);
                    if (!has_value) {
                        combined = value;
                        has_value = true;
                    } else {
                        combined = BKDRHashConcatOneField(combined, value);
                    }
                }

                if (any_null || !has_value) {
                    s = builder.AppendNull();
                    if (!s.ok()) {
                        return to_absl(s);
                    }
                } else {
                    s = builder.Append();
                    if (!s.ok()) {
                        return to_absl(s);
                    }
                    s = value_builder->Append(combined);
                    if (!s.ok()) {
                        return to_absl(s);
                    }
                }
            }

            auto array_result = builder.Finish();
            if (!array_result.ok()) {
                return to_absl(array_result.status());
            }
            auto array = *array_result;
            output_columns[spec_idx] = array;
            fields[spec_idx] = arrow::field("f" + std::to_string(spec_idx),
                                            std::make_shared<arrow::ListType>(arrow::uint64()));
        }
        return absl::OkStatus();
    };

    auto get_env_u64 = [](const char *name, uint64_t fallback) -> uint64_t {
        const char *str = getenv(name);
        if (!str || !*str) {
            return fallback;
        }
        char *end = nullptr;
        unsigned long long v = std::strtoull(str, &end, 10);
        if (!end || *end != '\0') {
            return fallback;
        }
        return static_cast<uint64_t>(v);
    };

    const uint64_t min_parallel_rules = get_env_u64(
        "METASPORE_LIGHTWEIGHT_MIN_PARALLEL_RULES", FLAGS_lightweight_min_parallel_rules);
    size_t max_workers = static_cast<size_t>(get_env_u64(
        "METASPORE_LIGHTWEIGHT_MAX_WORKERS", FLAGS_lightweight_max_workers));
    if (max_workers == 0) {
        max_workers = static_cast<size_t>(FLAGS_background_thread_num);
    }
    if (max_workers == 0) {
        max_workers = std::max<size_t>(1, std::thread::hardware_concurrency());
    }
    size_t worker_count = 1;
    if (specs_.size() >= min_parallel_rules && max_workers > 1) {
        worker_count = std::min(max_workers, specs_.size());
    }

    if (worker_count == 1) {
        for (size_t i = 0; i < specs_.size(); ++i) {
            CALL_AND_RETURN_IF_STATUS_NOT_OK(compute_one(i));
        }
    } else {
        auto &tp = Threadpools::get_background_threadpool();
        size_t chunk = (specs_.size() + worker_count - 1) / worker_count;
        std::vector<std::promise<status>> promises(worker_count);
        std::vector<std::future<status>> futures;
        futures.reserve(worker_count);
        for (auto &p : promises) {
            futures.push_back(p.get_future());
        }

        for (size_t w = 0; w < worker_count; ++w) {
            size_t begin = w * chunk;
            size_t end = std::min(specs_.size(), begin + chunk);
            if (begin >= end) {
                promises[w].set_value(absl::OkStatus());
                continue;
            }
            boost::asio::post(tp, [begin, end, &promises, w, &compute_one]() {
                status s = absl::OkStatus();
                for (size_t i = begin; i < end; ++i) {
                    s = compute_one(i);
                    if (!s.ok()) {
                        break;
                    }
                }
                promises[w].set_value(s);
            });
        }

        for (auto &fut : futures) {
            status s = fut.get();
            if (!s.ok()) {
                return s;
            }
        }
    }

    auto schema = std::make_shared<arrow::Schema>(std::move(fields));
    return arrow::RecordBatch::Make(std::move(schema), rows, std::move(output_columns));
}

} // namespace metaspore
