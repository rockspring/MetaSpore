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

#include <serving/feature_extraction_model_input.h>
#include <serving/grpc_model_runner.h>
#include <serving/metrics.h>
#include <serving/py_preprocessing_model.h>
#include <serving/ort_model.h>
#include <serving/tabular_model.h>

#include <chrono>

namespace metaspore::serving {

namespace {

inline double elapsed_ms(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - t0).count();
}

} // namespace

GrpcModelRunner::~GrpcModelRunner() = default;

awaitable_result<PredictReply> GrpcTabularModelRunner::predict(PredictRequest &request) {
    const auto &model_name = request.model_name();
    auto &metrics = Metrics::get_instance();

    // Stage 1: convert_input (grpc thread, synchronous)
    auto t0 = std::chrono::steady_clock::now();
    auto req   = std::make_unique<GrpcRequestOutput>(request);
    auto input = std::make_unique<FeatureExtractionModelInput>();
    auto conv_in_s = input_conveter->convert_input(std::move(req), input.get());
    metrics.record_duration(model_name, "convert_input", elapsed_ms(t0));
    CALL_AND_CO_RETURN_IF_STATUS_NOT_OK(conv_in_s);

    // Stage 2: model_compute (compute_threadpool via co_spawn, includes queue wait + inference)
    auto t1 = std::chrono::steady_clock::now();
    auto model_result = co_await model->predict(std::move(input));
    metrics.record_duration(model_name, "model_compute", elapsed_ms(t1));
    CALL_AND_CO_RETURN_IF_STATUS_NOT_OK(model_result);
    auto predict_result = std::move(*model_result);

    // Stage 3: convert_output (grpc thread, synchronous)
    auto t2 = std::chrono::steady_clock::now();
    PredictReply reply;
    auto reply_ptr  = std::make_unique<GrpcReplyInput>(reply);
    auto conv_out_s = output_conveter->convert_input(std::move(predict_result), reply_ptr.get());
    metrics.record_duration(model_name, "convert_output", elapsed_ms(t2));
    CALL_AND_CO_RETURN_IF_STATUS_NOT_OK(conv_out_s);

    auto *tabular_model = static_cast<TabularModel *>(model.get());
    (*reply.mutable_extras())["version"] = tabular_model->version();

    co_return reply;
}

awaitable_result<PredictReply> GrpcPreprocessingOrtModelRunner::predict(PredictRequest &request) {
    const auto &model_name = request.model_name();
    auto &metrics = Metrics::get_instance();

    // Stage 1: convert_input (grpc thread, synchronous)
    auto t0 = std::chrono::steady_clock::now();
    auto req   = std::make_unique<GrpcRequestOutput>(request);
    auto input = std::make_unique<PyPreprocessingModelInput>();
    auto conv_in_s = input_conveter->convert_input(std::move(req), input.get());
    metrics.record_duration(model_name, "convert_input", elapsed_ms(t0));
    CALL_AND_CO_RETURN_IF_STATUS_NOT_OK(conv_in_s);

    // Stage 2: model_compute (compute_threadpool via co_spawn, includes queue wait + inference)
    auto t1 = std::chrono::steady_clock::now();
    auto model_result = co_await model->predict(std::move(input));
    metrics.record_duration(model_name, "model_compute", elapsed_ms(t1));
    CALL_AND_CO_RETURN_IF_STATUS_NOT_OK(model_result);
    auto predict_result = std::move(*model_result);

    // Stage 3: convert_output (grpc thread, synchronous)
    auto t2 = std::chrono::steady_clock::now();
    PredictReply reply;
    auto reply_ptr  = std::make_unique<GrpcReplyInput>(reply);
    auto conv_out_s = output_conveter->convert_input(std::move(predict_result), reply_ptr.get());
    metrics.record_duration(model_name, "convert_output", elapsed_ms(t2));
    CALL_AND_CO_RETURN_IF_STATUS_NOT_OK(conv_out_s);

    co_return reply;
}

awaitable_result<PredictReply> GrpcOrtModelRunner::predict(PredictRequest &request) {
    const auto &model_name = request.model_name();
    auto &metrics = Metrics::get_instance();

    // Stage 1: convert_input (grpc thread, synchronous)
    auto t0 = std::chrono::steady_clock::now();
    auto req   = std::make_unique<GrpcRequestOutput>(request);
    auto input = std::make_unique<OrtModelInput>();
    auto conv_in_s = input_conveter->convert_input(std::move(req), input.get());
    metrics.record_duration(model_name, "convert_input", elapsed_ms(t0));
    CALL_AND_CO_RETURN_IF_STATUS_NOT_OK(conv_in_s);

    // Stage 2: model_compute (compute_threadpool via co_spawn, includes queue wait + inference)
    auto t1 = std::chrono::steady_clock::now();
    auto model_result = co_await model->predict(std::move(input));
    metrics.record_duration(model_name, "model_compute", elapsed_ms(t1));
    CALL_AND_CO_RETURN_IF_STATUS_NOT_OK(model_result);
    auto predict_result = std::move(*model_result);

    // Stage 3: convert_output (grpc thread, synchronous)
    auto t2 = std::chrono::steady_clock::now();
    PredictReply reply;
    auto reply_ptr  = std::make_unique<GrpcReplyInput>(reply);
    auto conv_out_s = output_conveter->convert_input(std::move(predict_result), reply_ptr.get());
    metrics.record_duration(model_name, "convert_output", elapsed_ms(t2));
    CALL_AND_CO_RETURN_IF_STATUS_NOT_OK(conv_out_s);

    co_return reply;
}

} // namespace metaspore::serving
