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

#include <stdlib.h>
#include <atomic>
#include <cstdio>
#include <future>
#include <memory>
#include <thread>
#include <boost/asio/post.hpp>
#include <boost/asio/thread_pool.hpp>
#include <gflags/gflags.h>
#include <pthread.h>

namespace metaspore {

DECLARE_uint64(compute_thread_num);
DECLARE_uint64(fe_compute_thread_num);
DECLARE_uint64(background_thread_num);

using threadpool = boost::asio::thread_pool;

inline void set_pool_thread_names(threadpool &tp, const char *prefix, int num_threads) {
    if (num_threads <= 0)
        return;

    struct State {
        std::atomic<int> index{0};
        std::atomic<int> barrier{0};
        std::promise<void> done;
        int total;
    };
    auto state = std::make_shared<State>();
    state->total = num_threads;
    auto future = state->done.get_future();

    for (int i = 0; i < num_threads; ++i) {
        boost::asio::post(tp, [state, prefix]() {
            int idx = state->index.fetch_add(1);
            char name[16];
            std::snprintf(name, sizeof(name), "%.11s_%d", prefix, idx);
#ifdef __linux__
            pthread_setname_np(pthread_self(), name);
#elif defined(__APPLE__)
            pthread_setname_np(name);
#endif
            // Spin-barrier: block this thread until all pool threads are named,
            // preventing one thread from completing early and stealing another's task.
            int n = state->barrier.fetch_add(1, std::memory_order_acq_rel) + 1;
            if (n == state->total) {
                state->done.set_value();
            } else {
                while (state->barrier.load(std::memory_order_acquire) < state->total)
                    std::this_thread::yield();
            }
        });
    }

    future.wait();
}

class Threadpools {
  public:
    static threadpool &get_compute_threadpool() {
        static int n = get_compute_thread_num();
        static threadpool tp(n);
        static bool named = (set_pool_thread_names(tp, "ms_compute", n), true);
        (void)named;
        return tp;
    }

    static threadpool &get_fe_compute_threadpool() {
        static int n = get_fe_compute_thread_num();
        static threadpool tp(n);
        static bool named = (set_pool_thread_names(tp, "ms_fe_comp", n), true);
        (void)named;
        return tp;
    }

    static threadpool &get_background_threadpool() {
        static int n = get_background_thread_num();
        static threadpool tp(n);
        static bool named = (set_pool_thread_names(tp, "ms_bg", n), true);
        (void)named;
        return tp;
    }

  private:
    static int get_compute_thread_num() {
        const char* str = getenv("METASPORE_COMPUTE_THREAD_NUM");
        if (str)
            return std::stoi(str);
        return FLAGS_compute_thread_num;
    }

    static int get_fe_compute_thread_num() {
        const char* str = getenv("METASPORE_FE_COMPUTE_THREAD_NUM");
        if (str)
            return std::stoi(str);
        return FLAGS_fe_compute_thread_num;
    }

    static int get_background_thread_num() {
        const char* str = getenv("METASPORE_BACKGROUND_THREAD_NUM");
        if (str)
            return std::stoi(str);
        return FLAGS_background_thread_num;
    }
};

} // namespace metaspore
