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

#if __has_include(<agrpc/asioGrpc.hpp>)
#include <agrpc/asioGrpc.hpp>
#elif __has_include(<agrpc/asio_grpc.hpp>)
#include <agrpc/asio_grpc.hpp>
#else
#error "Neither <agrpc/asioGrpc.hpp> nor <agrpc/asio_grpc.hpp> is available"
#endif

namespace metaspore::serving::agrpcx {

template <typename... Args>
auto request(Args &&...args) {
    if constexpr (requires { agrpc::request(std::forward<Args>(args)...); }) {
        return agrpc::request(std::forward<Args>(args)...);
    } else {
        return agrpc::b::request(std::forward<Args>(args)...);
    }
}

template <typename... Args>
auto finish(Args &&...args) {
    if constexpr (requires { agrpc::finish(std::forward<Args>(args)...); }) {
        return agrpc::finish(std::forward<Args>(args)...);
    } else {
        return agrpc::b::finish(std::forward<Args>(args)...);
    }
}

} // namespace metaspore::serving::agrpcx
