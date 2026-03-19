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

#include <istream>
#include <memory>
#include <string>
#include <vector>

#include <arrow/record_batch.h>

#include <common/types.h>

namespace metaspore {

class LightweightFeatureCompute {
  public:
    LightweightFeatureCompute() = default;

    status parse_schema(std::istream &is, int &feature_count);

    result<std::shared_ptr<arrow::RecordBatch>>
    execute(const std::shared_ptr<arrow::RecordBatch> &batch) const;

  private:
    struct FeatureSpec {
        std::vector<std::string> columns;
        std::vector<uint64_t> seeds;
    };

    std::vector<FeatureSpec> specs_;
};

} // namespace metaspore
