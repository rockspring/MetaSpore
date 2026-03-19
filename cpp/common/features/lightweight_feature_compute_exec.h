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

#include <memory>
#include <string>
#include <vector>

#include <arrow/record_batch.h>

#include <common/types.h>

namespace metaspore {

class LightweightFeatureComputeExec {
  public:
    LightweightFeatureComputeExec() = default;

    status add_source(const std::string &name);

    status add_projection(std::vector<std::vector<std::string>> columns);

    result<std::shared_ptr<arrow::RecordBatch>>
    execute(const std::shared_ptr<arrow::RecordBatch> &batch) const;

    std::vector<std::string> get_input_names() const;

  private:
    struct FeatureSpec {
        std::vector<std::string> columns;
        std::vector<uint64_t> seeds;
    };

    std::vector<std::string> input_names_;
    std::vector<FeatureSpec> specs_;
};

} // namespace metaspore
