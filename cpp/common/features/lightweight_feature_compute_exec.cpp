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

#include <common/features/lightweight_feature_compute_exec.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <future>
#include <numeric>
#include <thread>
#include <unordered_map>
#include <vector>

#include <arrow/array.h>
#include <arrow/builder.h>
#include <arrow/status.h>

#include <common/arrow/arrow_status.h>
#include <common/hash_utils.h>
#include <common/logger.h>
#include <common/threadpool.h>

#include <boost/asio/post.hpp>

namespace metaspore {

DECLARE_uint64(lightweight_feature_compute_min_parallel_rules);
DECLARE_uint64(lightweight_feature_compute_parallelism);
DECLARE_uint64(background_thread_num);

status LightweightFeatureComputeExec::add_source(const std::string &name) {
    if (name.empty()) {
        return absl::InvalidArgumentError(
            "LightweightFeatureComputeExec input name cannot be empty");
    }
    if (!input_names_.empty() && input_names_[0] == name) {
        return absl::OkStatus();
    }
    if (!input_names_.empty()) {
        return absl::AlreadyExistsError(
            "LightweightFeatureComputeExec only supports one input source");
    }
    input_names_.push_back(name);
    return absl::OkStatus();
}

status LightweightFeatureComputeExec::add_projection(
    std::vector<std::vector<std::string>> columns) {
    specs_.clear();
    specs_.reserve(columns.size());
    for (auto &cols : columns) {
        if (cols.empty()) {
            return absl::InvalidArgumentError(
                "LightweightFeatureComputeExec projection column list is empty");
        }
        FeatureSpec spec;
        spec.columns = std::move(cols);
        spec.seeds.reserve(spec.columns.size());
        for (const auto &name : spec.columns) {
            spec.seeds.push_back(BKDRHashWithEqualPostfix(name.c_str(), name.size(), 0));
        }
        specs_.push_back(std::move(spec));
    }
    return absl::OkStatus();
}

// Per-row accessor for a column: either a single hash value (string col)
// or a slice of hash values (list<string> col). Mirrors HashListAccessor but
// stores computed hashes inline so we don't need a separate uint64 array.
struct InlineHashList {
    // For string columns we store one value here and point begin/end into it.
    uint64_t scalar_val{0};
    // For list<string> columns we store values in a heap vector.
    std::vector<uint64_t> list_vals;
    bool is_null{false};

    size_t size() const {
        if (is_null) return 0;
        return list_vals.empty() ? 1 : list_vals.size();
    }
    uint64_t operator[](size_t i) const {
        return list_vals.empty() ? scalar_val : list_vals[i];
    }
};

// Build per-row InlineHashList for a string column.
static InlineHashList make_hash_list_string(const arrow::StringArray &arr, int64_t row,
                                            uint64_t seed) {
    InlineHashList h;
    if (arr.IsNull(row)) { h.is_null = true; return h; }
    auto view = arr.GetView(row);
    if (view.empty()) { h.is_null = true; return h; }
    h.scalar_val = BKDRHashOneField(seed, BKDRHash(view.data(), view.length(), 0));
    return h;
}

// Build per-row InlineHashList for a list<string> column.
static InlineHashList make_hash_list_list_string(const arrow::ListArray &arr, int64_t row,
                                                 uint64_t seed) {
    InlineHashList h;
    if (arr.IsNull(row)) { h.is_null = true; return h; }
    int32_t begin = arr.value_offset(row);
    int32_t end   = arr.value_offset(row + 1);
    if (begin == end) { h.is_null = true; return h; }
    const auto &values = static_cast<const arrow::StringArray &>(*arr.values());
    h.list_vals.reserve(end - begin);
    for (int32_t j = begin; j < end; ++j) {
        if (values.IsNull(j)) continue;
        auto view = values.GetView(j);
        if (view.empty()) continue;
        h.list_vals.push_back(BKDRHashOneField(seed, BKDRHash(view.data(), view.length(), 0)));
    }
    if (h.list_vals.empty()) h.is_null = true;
    return h;
}

// Thin wrapper so CartesianHashCombine's AppendFunc template param can be deduced.
template <typename AppendFunc>
static void cartesian_combine(const std::vector<InlineHashList> &lists, AppendFunc &&fn,
                               size_t total) {
    CartesianHashCombine<InlineHashList, std::vector, AppendFunc>::CombineOneFeature(
        lists, std::forward<AppendFunc>(fn), total);
}

result<std::shared_ptr<arrow::RecordBatch>>
LightweightFeatureComputeExec::execute(
    const std::shared_ptr<arrow::RecordBatch> &batch) const {
    if (!batch) {
        return absl::InvalidArgumentError("LightweightFeatureComputeExec input batch is null");
    }
    if (specs_.empty()) {
        return absl::InvalidArgumentError("LightweightFeatureComputeExec has no projections");
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

    // Cache raw Arrow arrays (base class) indexed by column position.
    std::vector<std::shared_ptr<arrow::Array>> column_cache((size_t)cols);
    std::vector<std::vector<int>> spec_indices;
    spec_indices.reserve(specs_.size());

    for (size_t spec_idx = 0; spec_idx < specs_.size(); ++spec_idx) {
        const auto &spec = specs_[spec_idx];
        if (spec.columns.empty()) {
            return absl::InvalidArgumentError(
                "LightweightFeatureComputeExec feature spec is empty");
        }

        std::vector<int> indices;
        indices.reserve(spec.columns.size());
        for (const auto &name : spec.columns) {
            auto it = name_to_index.find(name);
            if (it == name_to_index.end()) {
                return absl::NotFoundError(
                    "LightweightFeatureComputeExec cannot find column " + name);
            }
            int col_index = it->second;
            if (!column_cache[(size_t)col_index]) {
                column_cache[(size_t)col_index] = batch->column(col_index);
            }
            indices.push_back(col_index);
        }
        spec_indices.push_back(std::move(indices));
    }

    auto to_absl = [](const arrow::Status &s) -> status {
        return ArrowStatusToAbsl::arrow_status_to_absl(s);
    };

    auto compute_one = [&](size_t spec_idx) -> status {
        const auto &spec = specs_[spec_idx];
        const auto &indices = spec_indices[spec_idx];

        // Determine if any column is list<string>; if so output is list<uint64>.
        bool any_list = false;
        for (int col_index : indices) {
            if (column_cache[(size_t)col_index]->type_id() == arrow::Type::LIST)
                any_list = true;
        }

        if (!any_list && spec.columns.size() == 1) {
            // Fast path: single string column → uint64
            const auto &arr =
                static_cast<const arrow::StringArray &>(*column_cache[(size_t)indices[0]]);
            const uint64_t seed = spec.seeds[0];
            arrow::UInt64Builder builder;
            auto s = builder.Reserve(rows);
            if (!s.ok()) return to_absl(s);
            for (int64_t i = 0; i < rows; ++i) {
                if (arr.IsNull(i) || arr.GetView(i).empty()) {
                    s = builder.AppendNull();
                } else {
                    auto view = arr.GetView(i);
                    s = builder.Append(BKDRHashOneField(seed, BKDRHash(view.data(), view.length(), 0)));
                }
                if (!s.ok()) return to_absl(s);
            }
            auto array_result = builder.Finish();
            if (!array_result.ok()) return to_absl(array_result.status());
            output_columns[spec_idx] = *array_result;
            fields[spec_idx] = arrow::field("f" + std::to_string(spec_idx), arrow::uint64());
            return absl::OkStatus();
        }

        // General path: one or more columns, possibly list<string>.
        // Output is always list<uint64> (cartesian product of per-column hash lists).
        auto value_builder = std::make_shared<arrow::UInt64Builder>();
        arrow::ListBuilder builder(arrow::default_memory_pool(), value_builder,
                                   std::make_shared<arrow::ListType>(arrow::uint64()));
        auto s = builder.Reserve(rows);
        if (!s.ok()) return to_absl(s);

        for (int64_t i = 0; i < rows; ++i) {
            // Build per-column hash lists for this row.
            std::vector<InlineHashList> lists;
            lists.reserve(indices.size());
            bool any_null = false;
            for (size_t j = 0; j < indices.size(); ++j) {
                const auto &col = column_cache[(size_t)indices[j]];
                if (col->type_id() == arrow::Type::LIST) {
                    lists.push_back(make_hash_list_list_string(
                        static_cast<const arrow::ListArray &>(*col), i, spec.seeds[j]));
                } else {
                    lists.push_back(make_hash_list_string(
                        static_cast<const arrow::StringArray &>(*col), i, spec.seeds[j]));
                }
                if (lists.back().is_null) { any_null = true; break; }
            }

            if (any_null) {
                s = builder.AppendNull();
                if (!s.ok()) return to_absl(s);
                continue;
            }

            size_t total = std::accumulate(lists.begin(), lists.end(), size_t(1),
                                           [](size_t acc, const InlineHashList &l) {
                                               return acc * l.size();
                                           });
            if (total == 0) {
                s = builder.AppendNull();
                if (!s.ok()) return to_absl(s);
                continue;
            }

            s = builder.Append();
            if (!s.ok()) return to_absl(s);
            s = value_builder->Reserve(total);
            if (!s.ok()) return to_absl(s);

            cartesian_combine(lists, [&](uint64_t h) { (void)value_builder->Append(h); }, total);
        }

        auto array_result = builder.Finish();
        if (!array_result.ok()) return to_absl(array_result.status());
        output_columns[spec_idx] = *array_result;
        fields[spec_idx] = arrow::field("f" + std::to_string(spec_idx),
                                        std::make_shared<arrow::ListType>(arrow::uint64()));
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
        "METASPORE_LIGHTWEIGHT_FEATURE_COMPUTE_MIN_PARALLEL_RULES",
        FLAGS_lightweight_feature_compute_min_parallel_rules);
    size_t max_parallelism = static_cast<size_t>(get_env_u64(
        "METASPORE_LIGHTWEIGHT_FEATURE_COMPUTE_PARALLELISM",
        FLAGS_lightweight_feature_compute_parallelism));
    if (max_parallelism == 0) {
        max_parallelism = std::max<size_t>(1, std::thread::hardware_concurrency());
    }
    size_t partition_count = 1;
    if (specs_.size() >= min_parallel_rules && max_parallelism > 1) {
        partition_count = std::min(max_parallelism, specs_.size());
    }

    if (partition_count == 1) {
        for (size_t i = 0; i < specs_.size(); ++i) {
            CALL_AND_RETURN_IF_STATUS_NOT_OK(compute_one(i));
        }
    } else {
        auto &tp = Threadpools::get_fe_compute_threadpool();
        size_t chunk = (specs_.size() + partition_count - 1) / partition_count;
        std::vector<std::promise<status>> promises(partition_count);
        std::vector<std::future<status>> futures;
        futures.reserve(partition_count);
        for (auto &p : promises) {
            futures.push_back(p.get_future());
        }

        for (size_t w = 0; w < partition_count; ++w) {
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
    const int64_t out_cols = static_cast<int64_t>(output_columns.size());
    spdlog::debug(
        "LightweightFeatureComputeExec output schema built: rows={}, cols={}, schema={}",
        rows, out_cols, schema ? schema->ToString() : std::string("<null>"));

    auto out_batch =
        arrow::RecordBatch::Make(std::move(schema), rows, std::move(output_columns));
    spdlog::debug("LightweightFeatureComputeExec output RecordBatch: ptr={}, rows={}, cols={}",
                 static_cast<const void *>(out_batch.get()),
                 out_batch ? out_batch->num_rows() : -1,
                 out_batch ? out_batch->num_columns() : -1);
    if (out_batch) {
        const auto validate_status = out_batch->Validate();
        if (!validate_status.ok()) {
            spdlog::warn("LightweightFeatureComputeExec output RecordBatch validate failed: {}",
                         validate_status.ToString());
        }
    }
    return out_batch;
}

std::vector<std::string> LightweightFeatureComputeExec::get_input_names() const {
    return input_names_;
}

} // namespace metaspore
