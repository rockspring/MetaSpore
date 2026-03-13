# Serving 线程模型分析

## 概览

MetaSpore Serving 使用两类线程池处理请求：

| 线程池 | 数量 | 职责 |
|--------|------|------|
| `grpc_server_threads` | `--grpc_server_threads`（默认 4） | gRPC IO、协程调度、请求收发 |
| `compute_threadpool` | `METASPORE_COMPUTE_THREAD_NUM` 或 `--compute_thread_num` | 模型推理计算 |

两个线程池**完全独立**，推理慢不影响接收新请求，但超出 compute 线程数后新请求开始排队。

---

## 一、请求接收层：`register_predict_request_handler`

```cpp
// grpc_server.cpp - GrpcServer::run()
for (int i = 0; i < grpc_server_thread_count; i++) {
    grpc_server_threads.emplace_back([&, i] {
        auto &grpc_context = grpc_server_contexts[i];
        register_predict_request_handler(grpc_context, predict_service, server_shutdown);
        grpc_context.run();   // 单线程事件循环，阻塞于此
    });
}
```

每个 `grpc_server_threads[i]` 拥有独立的 `grpc_context`，运行各自的事件循环，互不干扰。

`register_predict_request_handler` 内部通过 `bind_executor(grpc_context, handler)` 将请求处理协程绑定到对应的 `grpc_context`，**同一个 `grpc_context` 上的协程串行调度**。

### grpc 线程内的协程调度

```
grpc_server_threads[i] 事件循环：
  请求A 到达 → 开始处理 → co_await predict() → 挂起，释放线程
  请求B 到达 → 开始处理 → co_await predict() → 挂起，释放线程
  请求A 完成 → 恢复，发送响应
  请求B 完成 → 恢复，发送响应
```

**协程挂起期间线程不阻塞**，可继续处理其他请求的 IO 事件，这是协程相比线程的核心优势。

---

## 二、请求处理层：`GrpcTabularModelRunner::predict`

```cpp
awaitable_result<PredictReply> GrpcTabularModelRunner::predict(PredictRequest &request) {
    // ① 在 grpc 线程上同步执行
    auto req   = std::make_unique<GrpcRequestOutput>(request);
    auto input = std::make_unique<FeatureExtractionModelInput>();
    CALL_AND_CO_RETURN_IF_STATUS_NOT_OK(input_conveter->convert_input(...));

    // ② co_spawn 切换到 compute_threadpool，grpc 线程挂起
    CO_ASSIGN_RESULT_OR_CO_RETURN_NOT_OK(auto predict_result, model->predict(std::move(input)));

    // ③ compute 线程完成后，结果投递回 grpc 线程
    PredictReply reply;
    CALL_AND_CO_RETURN_IF_STATUS_NOT_OK(output_conveter->convert_input(...));
    co_return reply;
}
```

| 步骤 | 执行线程 | 说明 |
|------|---------|------|
| `convert_input`（输入转换） | grpc 线程 | 同步执行，若慢则阻塞事件循环 |
| `model->predict()` | compute 线程 | 通过 `co_spawn` 切换，grpc 线程释放 |
| `convert_input`（输出转换） | grpc 线程 | 同步执行 |

> **注意**：输入/输出转换在 grpc 线程上同步执行，若这两步耗时较长，会直接影响该线程上所有请求的调度。

---

## 三、线程切换核心：`ModelBaseCRTP::predict`

```cpp
awaitable_result<...> ModelBaseCRTP::predict(...) {
    auto &tp = Threadpools::get_compute_threadpool();
    auto result = co_await boost::asio::co_spawn(
        tp,                          // 目标 executor
        [&]() -> awaitable_result<OutputType> {
            co_return co_await static_cast<Model*>(this)->do_predict(...);
        },
        boost::asio::use_awaitable   // 完成后结果投回调用方 executor
    );
    co_return result;
}
```

`co_spawn` + `use_awaitable` 的线程切换行为：

```
grpc_server_threads[i]                   compute_threadpool[j]
        │
        │  co_await co_spawn(tp, lambda)
        ├─────────────────────────────────► lambda 入队 compute_threadpool
        │  挂起，释放 grpc 线程
        │  [此时可处理其他请求 IO 事件]           lambda 被 compute 线程拾取执行
        │                                         do_predict() 执行中...
        │                                         完成，result 投递回 grpc_context
        │ ◄───────────────────────────────────────
        │  恢复，co_return result
```

`co_spawn` 的 awaitable `await_ready()` 始终返回 `false`，**一定发生线程切换**。

---

## 四、推理流水线：`TabularModel::do_predict`

此方法运行在 `compute_threadpool[j]` 上，内部流水线如下：

```
compute_threadpool[j]                    Arrow ExecPlan 线程
        │
        ├─[1] Sparse FE: start/build/feed  同步准备
        │
        │    co_await channel.async_receive ──────► Arrow 计算 sparse features
        │    若 channel 为空则挂起，让出线程               SinkConsumer::Consume
        │                                               → channel.try_send()
        │ ◄────────────────────────────────────────
        │    恢复（若 Arrow 已完成则不挂起）
        │
        ├─[2] Sparse Lookup: hashmap 查找   同步阻塞（内存映射哈希表）
        │
        ├─[3] Embedding Bag: session_.Run() 同步阻塞（OnnxRuntime 推理）
        │
        ├─[4] Dense FE: start/build/feed   同步准备
        │
        │    co_await channel.async_receive ──────► Arrow 计算 dense features
        │    若 channel 为空则挂起，让出线程
        │ ◄────────────────────────────────────────
        │
        └─[5] Final ORT: session_.Run()     同步阻塞（OnnxRuntime 推理）
```

### 各阶段线程状态

| 阶段 | 执行方式 | compute 线程状态 |
|------|---------|----------------|
| Sparse FE（Arrow ExecPlan） | 异步，`co_await channel` | Arrow 完成前**让出**，完成后立即恢复 |
| Sparse Lookup（hashmap 查找） | 同步 | **阻塞** |
| Embedding Bag ORT | 同步 `session_.Run()` | **阻塞** |
| Dense FE（Arrow ExecPlan） | 异步，`co_await channel` | Arrow 完成前**让出**，完成后立即恢复 |
| Final ORT | 同步 `session_.Run()` | **阻塞** |

---

## 五、`co_await` 是否放弃线程

`co_await expr` 是否放弃线程取决于 `expr.await_ready()` 的返回值：

```
await_ready() == true   → 直接取结果，不挂起，不放弃线程
await_ready() == false  → 挂起协程，放弃线程，等 await_resume() 被调用
```

| 场景 | await_ready | 是否放弃线程 |
|------|-------------|------------|
| `co_await co_spawn(other_executor, ...)` | 始终 false | **是**，切换到目标 executor |
| `co_await channel.async_receive()`，channel 有数据 | true | 否，直接取值继续执行 |
| `co_await channel.async_receive()`，channel 为空 | false | **是**，等待 Arrow push 数据 |
| `co_await sub_coroutine()`（同 executor） | false | 不放弃线程，切换执行子协程 |
| `session_.Run()`（非协程） | 不适用 | **阻塞**线程（无法让出） |

### 同 executor 下的 co_await

```cpp
// TabularModel::do_predict 中
CO_ASSIGN_RESULT_OR_CO_RETURN_NOT_OK(auto fe_result, unit.fe_model.do_predict(...));
// 展开为：
auto &&__r__ = co_await unit.fe_model.do_predict(...);
```

`do_predict()` 返回 `boost::asio::awaitable<T>`，其 `await_ready()` 始终返回 `false`，**协程会挂起**。但子协程立即在**同一个 compute 线程**上调度执行，父协程完成后立即恢复，从 CPU 角度看线程持续忙碌，本质等同于普通函数调用。

---

## 六、完整请求处理时序

```
grpc_server_threads[i]        compute_threadpool[j]       Arrow ExecPlan 线程
        │
        │  接收 gRPC 请求
        │  convert_input()              （同步，grpc 线程）
        │
        │  co_await co_spawn(tp, ...)
        ├──────────────────────────►
        │  grpc 线程释放
        │  [处理其他请求]                do_predict() 开始
        │                               Sparse FE 准备
        │                               co_await channel ──────────►
        │                               compute 线程释放               Arrow 计算
        │                               [可处理其他 compute 任务]      channel.try_send
        │                               compute 线程恢复 ◄─────────────
        │                               Sparse Lookup (阻塞)
        │                               Embedding Bag ORT (阻塞)
        │                               Dense FE 准备
        │                               co_await channel ──────────►
        │                               compute 线程释放               Arrow 计算
        │                               compute 线程恢复 ◄─────────────
        │                               Final ORT (阻塞)
        │                               完成，投递结果回 grpc_context
        │ ◄──────────────────────────
        │  grpc 线程恢复
        │  convert_output()             （同步，grpc 线程）
        │  发送 gRPC 响应
```

---

## 七、性能瓶颈与慢请求

### 并发上限

```
可同时接收请求数  = grpc_server_threads  （默认 4）
可同时执行推理数  = compute_threadpool 线程数
```

### 慢请求的影响

| 慢的位置 | 影响范围 | 严重程度 |
|---------|---------|---------|
| `convert_input/output`（grpc 线程） | 阻塞事件循环，新请求无法调度 | 严重 |
| `session_.Run()`（compute 线程） | 占用 compute 线程，超出线程数后队列堆积 | 一般 |
| Arrow FE（channel 等待） | compute 线程短暂让出，影响较小 | 轻微 |

### 慢请求日志

服务通过 `--predict_slow_log_threshold_ms`（默认 50ms）记录慢请求，日志包含：

```
[warn] Slow predict request: model=<name> batch_count=<n> elapsed=<ms>ms
```

`elapsed` 为端到端耗时，包含在 compute_threadpool 的**排队等待时间**，能反映真实请求延迟。
发送 `SIGUSR1` 信号可动态开启/关闭 CPU Profiler 进行性能分析：

```bash
kill -USR1 <pid>   # 开始采样
kill -USR1 <pid>   # 停止采样，输出到 --profiler_output 指定路径
```
