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

#pragma once

#include <prometheus/counter.h>
#include <prometheus/histogram.h>
#include <prometheus/registry.h>

#include <memory>
#include <optional>
#include <string>

// Forward declaration to avoid including exposer.h in the header
namespace prometheus { class Exposer; }

namespace metaspore::serving {

// Singleton that owns the Prometheus registry and HTTP exposer.
//
// Metrics exposed:
//   metaspore_predict_requests_total{model, status}  -- Counter  (QPS via rate())
//   metaspore_predict_duration_ms{model, stage}      -- Histogram
//
// Stages:
//   total          end-to-end latency (grpc_server.cpp)
//   convert_input  input conversion on grpc thread
//   model_compute  co_spawn → compute_threadpool (includes queue wait + inference)
//   convert_output output conversion on grpc thread
//
// HTTP endpoint: http://<grpc_listen_host>:<metrics_port>/metrics
class Metrics {
  public:
    static Metrics &get_instance();

    // Increment request counter. Call once per request after completion.
    void record_request(const std::string &model, bool ok);

    // Observe a stage duration (milliseconds, double precision).
    void record_duration(const std::string &model, const std::string &stage, double duration_ms);

  private:
    Metrics();
    ~Metrics();

    std::unique_ptr<prometheus::Exposer>       exposer_;
    std::shared_ptr<prometheus::Registry>      registry_;
    prometheus::Family<prometheus::Counter>   *requests_family_{nullptr};
    prometheus::Family<prometheus::Histogram> *duration_family_{nullptr};
};

} // namespace metaspore::serving
