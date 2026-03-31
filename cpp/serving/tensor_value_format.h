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

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <type_traits>

#include <fmt/format.h>
#include <onnxruntime/onnxruntime_cxx_api.h>

namespace metaspore::serving {

inline constexpr size_t kDefaultTensorValuePreviewMaxElems = 256;

template <typename T>
std::string format_tensor_elem_preview(const T &v) {
    if constexpr (std::is_floating_point_v<T>) {
        return fmt::format("{:.6g}", v);
    } else if constexpr (std::is_same_v<T, bool>) {
        return v ? "true" : "false";
    } else if constexpr (std::is_same_v<T, uint8_t>) {
        return fmt::format("{}", static_cast<unsigned int>(v));
    } else if constexpr (std::is_same_v<T, int8_t>) {
        return fmt::format("{}", static_cast<int>(v));
    } else {
        return fmt::format("{}", v);
    }
}

template <typename T>
std::string format_tensor_data_preview(const T *data, size_t element_count,
                                       size_t max_elems = kDefaultTensorValuePreviewMaxElems) {
    const size_t n = std::min(element_count, max_elems);
    std::string s = "[";
    for (size_t i = 0; i < n; ++i) {
        if (i != 0) {
            s += ", ";
        }
        s += format_tensor_elem_preview(data[i]);
    }
    s += "]";
    if (element_count > max_elems) {
        s += fmt::format(" ... ({} more elements)", element_count - max_elems);
    }
    return s;
}

inline std::string format_ort_tensor_data_preview(
    const Ort::Value &val, const Ort::TensorTypeAndShapeInfo &info,
    size_t max_elems = kDefaultTensorValuePreviewMaxElems) {
    const size_t count = info.GetElementCount();
    if (count == 0) {
        return "[]";
    }
    switch (info.GetElementType()) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        return format_tensor_data_preview(val.GetTensorData<float>(), count, max_elems);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
        return format_tensor_data_preview(val.GetTensorData<double>(), count, max_elems);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
        return format_tensor_data_preview(val.GetTensorData<int8_t>(), count, max_elems);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
        return format_tensor_data_preview(val.GetTensorData<uint8_t>(), count, max_elems);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
        return format_tensor_data_preview(val.GetTensorData<int16_t>(), count, max_elems);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
        return format_tensor_data_preview(val.GetTensorData<uint16_t>(), count, max_elems);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
        return format_tensor_data_preview(val.GetTensorData<int32_t>(), count, max_elems);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
        return format_tensor_data_preview(val.GetTensorData<uint32_t>(), count, max_elems);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
        return format_tensor_data_preview(val.GetTensorData<int64_t>(), count, max_elems);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
        return format_tensor_data_preview(val.GetTensorData<uint64_t>(), count, max_elems);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
        return format_tensor_data_preview(val.GetTensorData<bool>(), count, max_elems);
    default:
        return fmt::format("(preview unsupported for onnx_elem_type={})",
                           static_cast<int>(info.GetElementType()));
    }
}

template <typename T>
std::string format_tensor_data_full(const T *data, size_t element_count) {
    return format_tensor_data_preview(data, element_count, element_count);
}

inline std::string format_ort_tensor_data_full(const Ort::Value &val,
                                               const Ort::TensorTypeAndShapeInfo &info) {
    return format_ort_tensor_data_preview(val, info, info.GetElementCount());
}

/** Full-tensor numeric scan for trace/debug (cost scales with element count). */
inline std::string format_ort_tensor_trace_stats(const Ort::Value &val,
                                                 const Ort::TensorTypeAndShapeInfo &info) {
    const size_t n = info.GetElementCount();
    if (n == 0) {
        return "empty tensor";
    }
    switch (info.GetElementType()) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
        const float *p = val.GetTensorData<float>();
        float mn = std::numeric_limits<float>::infinity();
        float mx = -std::numeric_limits<float>::infinity();
        double sum = 0;
        size_t nonzero = 0, nan_c = 0, inf_c = 0, valid = 0;
        for (size_t i = 0; i < n; ++i) {
            const float v = p[i];
            if (std::isnan(v)) {
                nan_c++;
                continue;
            }
            if (std::isinf(v)) {
                inf_c++;
                continue;
            }
            valid++;
            mn = std::min(mn, v);
            mx = std::max(mx, v);
            sum += v;
            if (v != 0.f) {
                nonzero++;
            }
        }
        if (valid == 0) {
            return fmt::format("float elem={} all_nan_or_inf nan={} inf={}", n, nan_c, inf_c);
        }
        return fmt::format(
            "float elem={} min={} max={} mean(valid)={} nonzero={} nan={} inf={}", n, mn, mx,
            sum / static_cast<double>(valid), nonzero, nan_c, inf_c);
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: {
        const double *p = val.GetTensorData<double>();
        double mn = std::numeric_limits<double>::infinity();
        double mx = -std::numeric_limits<double>::infinity();
        double sum = 0;
        size_t nonzero = 0, nan_c = 0, inf_c = 0, valid = 0;
        for (size_t i = 0; i < n; ++i) {
            const double v = p[i];
            if (std::isnan(v)) {
                nan_c++;
                continue;
            }
            if (std::isinf(v)) {
                inf_c++;
                continue;
            }
            valid++;
            mn = std::min(mn, v);
            mx = std::max(mx, v);
            sum += v;
            if (v != 0.) {
                nonzero++;
            }
        }
        if (valid == 0) {
            return fmt::format("double elem={} all_nan_or_inf nan={} inf={}", n, nan_c, inf_c);
        }
        return fmt::format(
            "double elem={} min={} max={} mean(valid)={} nonzero={} nan={} inf={}", n, mn, mx,
            sum / static_cast<double>(valid), nonzero, nan_c, inf_c);
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: {
        const int32_t *p = val.GetTensorData<int32_t>();
        int32_t mn = p[0], mx = p[0];
        int64_t sum = 0;
        size_t nonzero = 0;
        for (size_t i = 0; i < n; ++i) {
            const int32_t v = p[i];
            mn = std::min(mn, v);
            mx = std::max(mx, v);
            sum += v;
            if (v != 0) {
                nonzero++;
            }
        }
        return fmt::format("int32 elem={} min={} max={} mean={} nonzero={}", n, mn, mx,
                           static_cast<double>(sum) / static_cast<double>(n), nonzero);
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
        const int64_t *p = val.GetTensorData<int64_t>();
        int64_t mn = p[0], mx = p[0];
        long double sum = 0;
        size_t nonzero = 0;
        for (size_t i = 0; i < n; ++i) {
            const int64_t v = p[i];
            mn = std::min(mn, v);
            mx = std::max(mx, v);
            sum += v;
            if (v != 0) {
                nonzero++;
            }
        }
        return fmt::format("int64 elem={} min={} max={} mean={} nonzero={}", n, mn, mx,
                           static_cast<double>(sum / static_cast<long double>(n)), nonzero);
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32: {
        const uint32_t *p = val.GetTensorData<uint32_t>();
        uint32_t mn = p[0], mx = p[0];
        uint64_t sum = 0;
        size_t nonzero = 0;
        for (size_t i = 0; i < n; ++i) {
            const uint32_t v = p[i];
            mn = std::min(mn, v);
            mx = std::max(mx, v);
            sum += v;
            if (v != 0u) {
                nonzero++;
            }
        }
        return fmt::format("uint32 elem={} min={} max={} mean={} nonzero={}", n, mn, mx,
                           static_cast<double>(sum) / static_cast<double>(n), nonzero);
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64: {
        const uint64_t *p = val.GetTensorData<uint64_t>();
        uint64_t mn = p[0], mx = p[0];
        long double sum = 0;
        size_t nonzero = 0;
        for (size_t i = 0; i < n; ++i) {
            const uint64_t v = p[i];
            mn = std::min(mn, v);
            mx = std::max(mx, v);
            sum += v;
            if (v != 0u) {
                nonzero++;
            }
        }
        return fmt::format("uint64 elem={} min={} max={} mean={} nonzero={}", n, mn, mx,
                           static_cast<double>(sum / static_cast<long double>(n)), nonzero);
    }
    default:
        return fmt::format("elem={} onnx_elem_type={} (numeric stats not implemented)",
                          n, static_cast<int>(info.GetElementType()));
    }
}

} // namespace metaspore::serving
