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
#include <serving/converters.h>
#include <serving/dense_feature_extraction_model.h>
#include <serving/feature_extraction_model_input.h>
#include <serving/ort_model.h>
#include <serving/sparse_embedding_bag_model.h>
#include <serving/sparse_feature_extraction_model.h>
#include <serving/sparse_lookup_model.h>
#include <serving/tabular_model.h>
#include <serving/tensor_value_format.h>
#include <common/threadpool.h>

#include <arrow/record_batch.h>
#include <arrow/tensor.h>
#include <arrow/type.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string_view>
#include <thread>

#include <boost/algorithm/string.hpp>
#include <fmt/format.h>
#include <fmt/ranges.h>

namespace metaspore::serving {

namespace fs = std::filesystem;

struct SparseModelUnit {
    SparseFeatureExtractionModel fe_model;
    SparseLookupModel lookup_model;
    SparseEmbeddingBagModel emb_model;
    std::unique_ptr<Converter> fe_to_lookup_converter;
};

namespace {

constexpr const char *kTabularTraceDumpRootEnv = "METASPORE_TRACE_DUMP_ROOT";

class TabularTraceFileWriter {
  public:
    static TabularTraceFileWriter &instance() {
        static TabularTraceFileWriter writer;
        return writer;
    }

    bool enabled() const { return !root_dir_.empty(); }

    void dump(std::string_view category, std::string_view step, const std::string &content) {
        if (!enabled()) {
            return;
        }
        std::lock_guard<std::mutex> lk(mu_);
        const auto seq = seq_.fetch_add(1, std::memory_order_relaxed);
        const auto now = std::chrono::system_clock::now().time_since_epoch();
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
        const auto tid = std::hash<std::thread::id>{}(std::this_thread::get_id());
        const std::string file_name =
            fmt::format("{}_{}_{}_{}_{}.txt", ms, seq, tid, sanitize(category), sanitize(step));
        const fs::path p = root_dir_ / file_name;
        std::ofstream ofs(p, std::ios::out | std::ios::trunc);
        if (!ofs.is_open()) {
            spdlog::warn("Tabular trace failed to open dump file: {}", p.string());
            return;
        }
        ofs << content;
        ofs.flush();
    }

  private:
    TabularTraceFileWriter() : root_dir_() {
        const char *env = std::getenv(kTabularTraceDumpRootEnv);
        if (env == nullptr || *env == '\0') {
            return;
        }
        fs::path p(env);
        std::error_code ec;
        const bool path_exists = fs::exists(p, ec);
        if (ec) {
            spdlog::warn("Tabular trace cannot stat dump root from {}: {}", kTabularTraceDumpRootEnv,
                         ec.message());
            return;
        }
        if (!path_exists) {
            fs::create_directories(p, ec);
            if (ec) {
                spdlog::warn("Tabular trace cannot create dump root from {}: {}",
                             kTabularTraceDumpRootEnv, ec.message());
                return;
            }
        } else if (!fs::is_directory(p, ec)) {
            const std::string err = ec ? ec.message() : "path is not a directory";
            spdlog::warn("Tabular trace dump root from {} is invalid: {}", kTabularTraceDumpRootEnv,
                         err);
            return;
        }
        root_dir_ = p;
        spdlog::info("Tabular trace dump enabled, root={}", root_dir_.string());
    }

    static std::string sanitize(std::string_view s) {
        std::string out;
        out.reserve(s.size());
        for (const char c : s) {
            if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
                c == '_' || c == '-') {
                out.push_back(c);
            } else {
                out.push_back('_');
            }
        }
        if (out.empty()) {
            out = "unknown";
        }
        return out;
    }

    fs::path root_dir_;
    std::atomic<uint64_t> seq_{0};
    std::mutex mu_;
};

bool tabular_trace_enabled() { return TabularTraceFileWriter::instance().enabled(); }

std::string record_batch_values_full(const std::shared_ptr<arrow::RecordBatch> &batch) {
    if (!batch) {
        return "null";
    }
    return batch->ToString();
}

template <typename T>
std::string dump_arrow_tensor_shared_full(const std::shared_ptr<T> &t) {
    if (!t) {
        return "null";
    }
    const auto *raw = t->raw_data();
    const size_t sz = static_cast<size_t>(t->size());
    if (sz == 0) {
        return "[]";
    }
    switch (t->type_id()) {
    case arrow::Type::FLOAT:
        return format_tensor_data_full(reinterpret_cast<const float *>(raw), sz);
    case arrow::Type::DOUBLE:
        return format_tensor_data_full(reinterpret_cast<const double *>(raw), sz);
    case arrow::Type::INT8:
        return format_tensor_data_full(reinterpret_cast<const int8_t *>(raw), sz);
    case arrow::Type::UINT8:
        return format_tensor_data_full(reinterpret_cast<const uint8_t *>(raw), sz);
    case arrow::Type::INT16:
        return format_tensor_data_full(reinterpret_cast<const int16_t *>(raw), sz);
    case arrow::Type::UINT16:
        return format_tensor_data_full(reinterpret_cast<const uint16_t *>(raw), sz);
    case arrow::Type::INT32:
        return format_tensor_data_full(reinterpret_cast<const int32_t *>(raw), sz);
    case arrow::Type::UINT32:
        return format_tensor_data_full(reinterpret_cast<const uint32_t *>(raw), sz);
    case arrow::Type::INT64:
        return format_tensor_data_full(reinterpret_cast<const int64_t *>(raw), sz);
    case arrow::Type::UINT64:
        return format_tensor_data_full(reinterpret_cast<const uint64_t *>(raw), sz);
    default:
        return fmt::format("(tensor type {} value preview unsupported)", t->type()->ToString());
    }
}

std::string summarize_record_batch(const std::shared_ptr<arrow::RecordBatch> &batch) {
    if (!batch) {
        return "null";
    }
    std::vector<std::string> col_names;
    col_names.reserve(batch->num_columns());
    for (int i = 0; i < batch->num_columns(); ++i) {
        col_names.push_back(batch->column_name(i));
    }
    return fmt::format("rows={} cols=[{}]", batch->num_rows(), fmt::join(col_names, ", "));
}

void trace_fe_input(std::string_view step, const FeatureExtractionModelInput &in) {
    if (!tabular_trace_enabled()) {
        return;
    }
    std::ostringstream oss;
    oss << "FeatureExtractionModelInput\n";
    for (const auto &[k, v] : in.feature_tables) {
        oss << "table=\"" << k << "\" meta=" << summarize_record_batch(v) << "\n";
        oss << "values:\n" << record_batch_values_full(v) << "\n";
    }
    TabularTraceFileWriter::instance().dump("trace_fe_input", step, oss.str());
}

void trace_sparse_fe_output(std::string_view step, const SparseFeatureExtractionModelOutput &out) {
    if (!tabular_trace_enabled()) {
        return;
    }
    std::ostringstream oss;
    oss << "SparseFeatureExtractionModelOutput\n";
    oss << "meta=" << summarize_record_batch(out.values) << "\n";
    oss << "values:\n" << record_batch_values_full(out.values) << "\n";
    TabularTraceFileWriter::instance().dump("trace_sparse_fe_output", step, oss.str());
}

std::string summarize_shape(const std::vector<int64_t> &shape) { return fmt::format("{}", fmt::join(shape, ",")); }

std::string summarize_uint64_tensor(const std::shared_ptr<arrow::UInt64Tensor> &t) {
    if (!t) {
        return "null";
    }
    return fmt::format("shape=[{}] size={}", summarize_shape(t->shape()), t->size());
}

std::string summarize_float_tensor(const std::shared_ptr<arrow::FloatTensor> &t) {
    if (!t) {
        return "null";
    }
    return fmt::format("shape=[{}] size={}", summarize_shape(t->shape()), t->size());
}

void trace_sparse_lookup_input(std::string_view step, const SparseLookupModelInput &in) {
    if (!tabular_trace_enabled()) {
        return;
    }
    std::ostringstream oss;
    oss << "SparseLookupModelInput\n";
    oss << "batch_size=" << in.batch_size << " indices_meta=" << summarize_uint64_tensor(in.indices)
        << " offsets_meta=" << summarize_uint64_tensor(in.offsets) << "\n";
    oss << "indices_values=" << dump_arrow_tensor_shared_full(in.indices) << "\n";
    oss << "offsets_values=" << dump_arrow_tensor_shared_full(in.offsets) << "\n";
    TabularTraceFileWriter::instance().dump("trace_sparse_lookup_input", step, oss.str());
}

void trace_sparse_lookup_output(std::string_view step, const SparseLookupModelOutput &out) {
    if (!tabular_trace_enabled()) {
        return;
    }
    std::ostringstream oss;
    oss << "SparseLookupModelOutput\n";
    oss << "batch_size=" << out.batch_size << " keys.size=" << out.keys.size()
        << " indices_meta=" << summarize_uint64_tensor(out.indices)
        << " offsets_meta=" << summarize_uint64_tensor(out.offsets)
        << " values_meta=" << summarize_float_tensor(out.values) << "\n";
    oss << "keys_values=" << format_tensor_data_full(out.keys.data(), out.keys.size()) << "\n";
    oss << "indices_values=" << dump_arrow_tensor_shared_full(out.indices) << "\n";
    oss << "offsets_values=" << dump_arrow_tensor_shared_full(out.offsets) << "\n";
    oss << "values_values=" << dump_arrow_tensor_shared_full(out.values) << "\n";
    TabularTraceFileWriter::instance().dump("trace_sparse_lookup_output", step, oss.str());
}

void trace_dense_fe_output(std::string_view step, const DenseFeatureExtractionModelOutput &out) {
    if (!tabular_trace_enabled()) {
        return;
    }
    std::ostringstream oss;
    oss << "DenseFeatureExtractionModelOutput\n";
    for (const auto &[name, t] : out.feature_tensors) {
        oss << "name=\"" << name << "\" meta=" << summarize_float_tensor(t) << "\n";
        oss << "values=" << dump_arrow_tensor_shared_full(t) << "\n";
    }
    TabularTraceFileWriter::instance().dump("trace_dense_fe_output", step, oss.str());
}

void trace_ort_model_input(std::string_view step, const OrtModelInput &in) {
    if (!tabular_trace_enabled()) {
        return;
    }
    std::ostringstream oss;
    oss << "OrtModelInput\n";
    for (const auto &[name, holder] : in.inputs) {
        const Ort::Value &val = holder.value;
        if (!val.IsTensor()) {
            oss << "name=\"" << name << "\" non-tensor\n";
            continue;
        }
        Ort::TensorTypeAndShapeInfo info = val.GetTensorTypeAndShapeInfo();
        oss << "name=\"" << name << "\" shape=[" << summarize_shape(info.GetShape())
            << "] onnx_elem_type=" << static_cast<int>(info.GetElementType())
            << " element_count=" << info.GetElementCount() << "\n";
        oss << "values="
            << format_ort_tensor_data_full(val, info)
            << "\n";
    }
    TabularTraceFileWriter::instance().dump("trace_ort_model_input", step, oss.str());
}

void trace_ort_model_output(std::string_view step, const OrtModelOutput &out) {
    if (!tabular_trace_enabled()) {
        return;
    }
    std::ostringstream oss;
    oss << "OrtModelOutput\n";
    for (const auto &[name, val] : out.outputs) {
        if (!val.IsTensor()) {
            oss << "name=\"" << name << "\" non-tensor\n";
            continue;
        }
        Ort::TensorTypeAndShapeInfo info = val.GetTensorTypeAndShapeInfo();
        oss << "name=\"" << name << "\" shape=[" << summarize_shape(info.GetShape())
            << "] onnx_elem_type=" << static_cast<int>(info.GetElementType())
            << " element_count=" << info.GetElementCount() << "\n";
        oss << "values="
            << format_ort_tensor_data_full(val, info)
            << "\n";
    }
    TabularTraceFileWriter::instance().dump("trace_ort_model_output", step, oss.str());
}

} // namespace

class TabularModelContext {
  public:
    std::vector<SparseModelUnit> sparse_models;
    DenseFeatureExtractionModel dense_model;
    std::unique_ptr<Converter> dense_fe_to_ort_converter;
    OrtModel ort_model;
    // inputs of crt model is unique set of inputs of all Fe models
    std::vector<std::string> inputs_;
};

TabularModel::TabularModel() { context_ = std::make_unique<TabularModelContext>(); }

TabularModel::TabularModel(TabularModel &&) = default;

TabularModel::~TabularModel() = default;

awaitable_status TabularModel::load(std::string dir_path) {
    auto s = co_await boost::asio::co_spawn(
        Threadpools::get_background_threadpool(),
        [this, &dir_path]() -> awaitable_status {
            // load a ctr model
            // 1. find all subdirs prefixed with "sparse_" and load fe/lookup models in them
            fs::path root_dir(dir_path);
            if (!fs::is_directory(root_dir)) {
                co_return absl::NotFoundError(
                    fmt::format("TabularModel cannot find dir {}", dir_path));
            }
            bool dense_loaded = false;

            for (auto const &dir_entry : fs::directory_iterator{root_dir}) {
                if (!dir_entry.is_directory()) {
                    // find if dense schema exist and load it
                    if (dir_entry.path().filename() == "dense_schema.txt") {
                        CO_AWAIT_AND_CO_RETURN_IF_STATUS_NOT_OK(
                            context_->dense_model.load(root_dir.string()));
                        context_->dense_fe_to_ort_converter =
                            std::make_unique<DenseFEToOrtConverter>(
                                context_->dense_model.input_names());
                        std::copy(context_->dense_model.input_names().begin(),
                                  context_->dense_model.input_names().end(),
                                  std::back_inserter(context_->inputs_));
                    }
                    continue;
                }

                auto dir_name = dir_entry.path().filename();
                // begin to load components of sparse models
                if (boost::contains(dir_name.string(), "sparse")) {
                    SparseModelUnit unit;
                    int component_loaded = 0;
                    // load sparse model
                    for (auto const &sparse_dir_entry : fs::directory_iterator{dir_entry}) {
                        spdlog::info("find path {} under {}", sparse_dir_entry.path().string(),
                                     dir_entry.path().string());
                        if (!sparse_dir_entry.is_directory() &&
                            sparse_dir_entry.path().filename() == "combine_schema.txt") {
                            spdlog::info("Loading sparse fe model from {}",
                                         dir_entry.path().string());
                            // load sparse fe model
                            CO_AWAIT_AND_CO_RETURN_IF_STATUS_NOT_OK(
                                unit.fe_model.load(dir_entry.path().string()));
                            component_loaded ^= 0b1;
                            std::copy(unit.fe_model.input_names().begin(),
                                      unit.fe_model.input_names().end(),
                                      std::back_inserter(context_->inputs_));
                        } else if (sparse_dir_entry.path().filename().string() ==
                                   "embedding_table") {
                            spdlog::info("Loading sparse lookup model from {}",
                                         dir_entry.path().string());
                            // load lookup model
                            CO_AWAIT_AND_CO_RETURN_IF_STATUS_NOT_OK(
                                unit.lookup_model.load(dir_entry.path().string()));
                            unit.fe_to_lookup_converter =
                                std::make_unique<SparseFEToLookupConverter>();
                            component_loaded ^= 0b10;
                        } else if (!sparse_dir_entry.is_directory() &&
                                   sparse_dir_entry.path().filename() == "model.onnx") {
                            spdlog::info("Loading sparse embedding bag model from {}",
                                         dir_entry.path().string());
                            // load sparse emebdding bag model
                            CO_AWAIT_AND_CO_RETURN_IF_STATUS_NOT_OK(
                                unit.emb_model.load(dir_entry.path().string()));
                            component_loaded ^= 0b100;
                        }
                    }
                    if (component_loaded != 0b111) {
                        co_return absl::NotFoundError(
                            fmt::format("TabularModel with a sparse component under {} requires a "
                                        "combine_schema.txt file, an embedding_table dir and a "
                                        "model.onnx file to initialize, component loaded {:#b}",
                                        dir_name.string(), component_loaded));
                    }
                    context_->sparse_models.emplace_back(std::move(unit));
                } else if (boost::contains(dir_name.string(), "dense")) {
                    // load dense ort model
                    if (!dense_loaded) {
                        CO_AWAIT_AND_CO_RETURN_IF_STATUS_NOT_OK(
                            context_->ort_model.load(dir_entry.path()));
                        dense_loaded = true;
                    } else {
                        spdlog::error("TabularModel cannot support more than one dense model");
                        co_return absl::UnimplementedError(
                            "TabularModel cannot support more than one dense model");
                    }
                }
            }

            if (context_->sparse_models.empty() && !context_->dense_fe_to_ort_converter) {
                auto msg = fmt::format(
                    "TabularModel requires at least one fe model while loading from {}", dir_path);
                spdlog::error(msg);
                co_return absl::NotFoundError(msg);
            }

            if (!dense_loaded) {
                auto msg =
                    fmt::format("TabularModel requires an onnx model under {}/dense/", dir_path);
                spdlog::error(msg);
                co_return absl::NotFoundError(msg);
            }

            // get unique input names from all sparse/fe models as the inputs of TabularModel
            std::sort(context_->inputs_.begin(), context_->inputs_.end());
            auto last = std::unique(context_->inputs_.begin(), context_->inputs_.end());
            context_->inputs_.erase(last, context_->inputs_.end());

            spdlog::info("TabularModel loaded from {}, required inputs [{}], "
                         "producing outputs [{}]",
                         dir_path, fmt::join(context_->inputs_, ", "),
                         fmt::join(this->output_names(), ", "));
            co_return absl::OkStatus();
        },
        boost::asio::use_awaitable);
    co_return s;
}

std::unique_ptr<FeatureExtractionModelInput> get_input(ModelBase &fe_model,
                                                       const FeatureExtractionModelInput *input) {
    auto out = std::make_unique<FeatureExtractionModelInput>();
    for (const auto &name : fe_model.input_names()) {
        auto find = input->feature_tables.find(name);
        if (find == input->feature_tables.end()) {
            spdlog::error("Fe model required input {} not found");
            return nullptr;
        }
        out->feature_tables[name] = find->second;
    }
    return out;
}

awaitable_result<std::unique_ptr<OrtModelOutput>>
TabularModel::do_predict(std::unique_ptr<FeatureExtractionModelInput> input) {
    // firstly execute sparse fe and lookup
    // set all output to ort input
    auto *fe_input = input.get();
    trace_fe_input("entry", *fe_input);
    auto ort_in = std::make_unique<OrtModelInput>();
    for (size_t unit_i = 0; unit_i < context_->sparse_models.size(); ++unit_i) {
        auto &unit = context_->sparse_models[unit_i];
        auto sub_input = get_input(unit.fe_model, fe_input);
        if (!sub_input) {
            co_return absl::NotFoundError("Input not found for sparse fe model");
        }
        trace_fe_input(fmt::format("sparse[{}]/sparse_fe input", unit_i), *sub_input);
        CO_ASSIGN_RESULT_OR_CO_RETURN_NOT_OK(auto fe_result,
                                             unit.fe_model.do_predict(std::move(sub_input)));
        trace_sparse_fe_output(fmt::format("sparse[{}]/sparse_fe output", unit_i), *fe_result);
        auto lookup_in = std::make_unique<SparseLookupModelInput>();
        CALL_AND_CO_RETURN_IF_STATUS_NOT_OK(
            unit.fe_to_lookup_converter->convert_input(std::move(fe_result), lookup_in.get()));
        trace_sparse_lookup_input(fmt::format("sparse[{}]/lookup input", unit_i), *lookup_in);

        CO_ASSIGN_RESULT_OR_CO_RETURN_NOT_OK(auto lookup_result,
                                             unit.lookup_model.do_predict(std::move(lookup_in)));
        trace_sparse_lookup_output(fmt::format("sparse[{}]/lookup output", unit_i), *lookup_result);

        CO_ASSIGN_RESULT_OR_CO_RETURN_NOT_OK(auto emb_ort_result,
                                             unit.emb_model.do_predict(std::move(lookup_result)));
        trace_ort_model_output(fmt::format("sparse[{}]/embedding_bag output", unit_i),
                               *emb_ort_result);

        // merge embedding bag output to ort_in
        for (auto &[name, v] : emb_ort_result->outputs) {
            if (!ort_in->inputs.emplace(name, OrtModelInput::Value{.value = std::move(v)}).second) {
                co_return absl::AlreadyExistsError(
                    fmt::format("Sparse Embedding produced duplicated output {}", name));
            }
        }
    }

    // secondly execute dense fe if it exists
    // and set all output to ort input
    if (context_->dense_fe_to_ort_converter) {
        auto sub_input = get_input(context_->dense_model, fe_input);
        if (!sub_input) {
            co_return absl::NotFoundError("Input not found for dense fe model");
        }
        trace_fe_input("dense_fe input", *sub_input);
        CO_ASSIGN_RESULT_OR_CO_RETURN_NOT_OK(
            auto fe_result, context_->dense_model.do_predict(std::move(sub_input)));
        trace_dense_fe_output("dense_fe output", *fe_result);
        CO_RETURN_IF_STATUS_NOT_OK(
            context_->dense_fe_to_ort_converter->convert_input(std::move(fe_result), ort_in.get()));
    }

    // finally execute ort model prediction
    trace_ort_model_input("ort_model input", *ort_in);
    CO_ASSIGN_RESULT_OR_CO_RETURN_NOT_OK(auto final_result,
                                         context_->ort_model.do_predict(std::move(ort_in)));
    trace_ort_model_output("ort_model output", *final_result);
    co_return final_result;
}

std::string TabularModel::info() const { return ""; }

const std::vector<std::string> &TabularModel::input_names() const { return context_->inputs_; }

const std::vector<std::string> &TabularModel::output_names() const {
    return context_->ort_model.output_names();
}

} // namespace metaspore::serving