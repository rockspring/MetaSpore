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

#include <serving/metrics.h>
#include <common/logger.h>

#include <prometheus/counter.h>
#include <prometheus/exposer.h>
#include <prometheus/histogram.h>
#include <prometheus/registry.h>

#include <gflags/gflags.h>

#include <fmt/format.h>

namespace metaspore::serving {

DECLARE_string(grpc_listen_host);
DECLARE_uint32(metrics_port);

namespace {

// Histogram bucket boundaries in milliseconds.
// Sub-millisecond (0.5) through 5 seconds covers convert steps and heavy ORT inference.
const prometheus::Histogram::BucketBoundaries kDurationBuckets = {
    0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000
};

// Histogram bucket boundaries for request batch size.
const prometheus::Histogram::BucketBoundaries kBatchSizeBuckets = {
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048
};

} // namespace

Metrics &Metrics::get_instance() {
    static Metrics instance;
    return instance;
}

Metrics::Metrics() {
    registry_ = std::make_shared<prometheus::Registry>();

    requests_family_ = &prometheus::BuildCounter()
        .Name("metaspore_predict_requests_total")
        .Help("Total number of predict requests, labeled by model and status (ok/error)")
        .Register(*registry_);

    duration_family_ = &prometheus::BuildHistogram()
        .Name("metaspore_predict_duration_ms")
        .Help("Predict latency in milliseconds, labeled by model and stage "
              "(total/convert_input/model_compute/convert_output)")
        .Register(*registry_);

    batch_size_family_ = &prometheus::BuildHistogram()
        .Name("metaspore_predict_batch_size")
        .Help("Predict request batch size, labeled by model")
        .Register(*registry_);

    if (FLAGS_metrics_port == 0) {
        spdlog::info("Prometheus metrics disabled (--metrics_port=0)");
        return;
    }

    const std::string addr = fmt::format("{}:{}", FLAGS_grpc_listen_host, FLAGS_metrics_port);
    try {
        exposer_ = std::make_unique<prometheus::Exposer>(addr);
        exposer_->RegisterCollectable(registry_);
        spdlog::info("Prometheus metrics available at http://{}/metrics", addr);
    } catch (const std::exception &e) {
        spdlog::error("Failed to start metrics HTTP server on {}: {}", addr, e.what());
    }
}

Metrics::~Metrics() = default;

void Metrics::record_request(const std::string &model, bool ok) {
    if (!requests_family_) return;
    requests_family_->Add({{"model", model}, {"status", ok ? "ok" : "error"}}).Increment();
}

void Metrics::record_duration(const std::string &model, const std::string &stage,
                               double duration_ms) {
    if (!duration_family_) return;
    duration_family_->Add({{"model", model}, {"stage", stage}}, kDurationBuckets)
        .Observe(duration_ms);
}

void Metrics::record_batch_size(const std::string &model, int64_t batch_size) {
    if (!batch_size_family_ || batch_size <= 0) return;
    batch_size_family_->Add({{"model", model}}, kBatchSizeBuckets)
        .Observe(static_cast<double>(batch_size));
}

} // namespace metaspore::serving
