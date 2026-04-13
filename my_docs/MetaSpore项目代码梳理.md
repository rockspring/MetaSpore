# MetaSpore 项目代码梳理

> 作者：代码梳理文档  
> 项目版本：1.2.0  
> 开源方：DMetaSoul（元灵数智）  
> 仓库：https://github.com/meta-soul/MetaSpore

---

## 一、项目概述

MetaSpore 是一个**一站式端到端机器学习开发平台**，覆盖从数据预处理、模型训练、离线实验，到在线预测、实验分桶（A/B Test）的完整机器学习生命周期。

核心定位：

- 兼容 PyTorch 生态的分布式深度学习训练框架，原生支持大规模稀疏特征
- 与 PySpark 无缝集成，可直接从数据湖/数仓读取训练数据
- 高性能在线推理服务，支持神经网络、决策树、Spark ML、SKLearn 等多种模型
- 在离线统一特征抽取框架，自动生成线上特征读取逻辑，消除特征一致性问题
- 在线算法应用框架，提供模型预测、实验分桶、参数热加载等能力

---

## 二、整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                        离线训练层                            │
│  PySpark 数据 → Python 训练框架 → 分布式参数服务器 → 模型存储  │
└─────────────────────────────────────────────────────────────┘
                              ↓ 模型导出
┌─────────────────────────────────────────────────────────────┐
│                        在线服务层                            │
│  gRPC 请求 → 特征提取 → 稀疏查找 → ORT 推理 → gRPC 响应      │
└─────────────────────────────────────────────────────────────┘
                              ↓ 实验管理
┌─────────────────────────────────────────────────────────────┐
│                     Java 在线应用层                          │
│  实验分桶 / A/B Test / 特征服务 / 推荐服务                    │
└─────────────────────────────────────────────────────────────┘
```

项目由三大部分组成：
1. **C++ 核心引擎**（`cpp/`）：分布式训练框架 + 高性能在线推理服务
2. **Python 训练框架**（`python/`）：基于 PyTorch 的分布式训练 API
3. **Java 在线服务**（`java/`）：在线特征提取、实验分桶、推荐服务

---

## 三、目录结构总览

```
MetaSpore/
├── cpp/
│   ├── common/          # 通用工具库（Arrow、哈希表、特征计算）
│   ├── metaspore/       # 分布式训练 C++ 核心（PS 框架、张量存储）
│   ├── serving/         # 在线推理服务（gRPC、模型管理、ORT）
│   └── tests/           # C++ 单元测试
├── python/
│   ├── metaspore/       # Python 训练框架（分布式训练、算法库）
│   ├── metasporecli/    # 命令行工具
│   └── metasporeflow/   # 工作流框架
├── java/
│   └── online-service/  # Java 在线服务框架
├── protos/              # Protocol Buffer 定义
├── thrift/              # Thrift 定义
├── cmake/               # CMake 构建配置
├── demo/                # 端到端示例（推荐、CTR、搜索等）
├── docker/              # Docker 镜像
├── kubernetes/          # K8s 部署配置
└── tutorials/           # 教程
```

---

## 四、C++ 核心引擎

### 4.1 通用工具库（`cpp/common/`）

这是整个 C++ 层的基础设施，被训练框架和推理服务共同依赖。

#### 核心类型系统（`types.h`）

基于 Boost.ASIO 协程和 Abseil 状态库，定义了全局通用类型：

```cpp
template <typename T> using awaitable = boost::asio::awaitable<T>;
template <typename T> using result    = absl::StatusOr<T>;
using status = absl::Status;
```

配套宏（`RETURN_IF_STATUS_NOT_OK`、`CO_AWAIT_AND_CO_RETURN_IF_STATUS_NOT_OK` 等）统一了同步/异步代码的错误处理风格。

#### 高性能哈希表（`hashmap/`）

这是稀疏特征存储的核心数据结构，专为大规模 Embedding 表设计：

| 文件 | 功能 |
|------|------|
| `array_hash_map.h` | 核心哈希表模板，支持序列化/反序列化 |
| `memory_mapped_array_hash_map.h` | 内存映射版本，支持超大规模稀疏表 |
| `perfect_hash_index_builder.h` | 完美哈希索引，优化只读查询性能 |
| `hash_uniquifier.h` | 哈希去重工具 |

`ArrayHashMap<TKey, TValue>` 是一个链式哈希表，内部用四个数组（keys、values、next、first）实现，支持 `value_count_per_key` 来存储定长向量值（即 Embedding 向量）。

#### 特征计算引擎（`features/`）

基于 Apache Arrow 的列式特征计算框架：

- `FeatureComputeExec`：构建特征计算执行计划，支持多源 Join 和 Projection
- `FeatureComputeFuncs`：特征计算函数库（Hash、Combine 等）
- `SchemaParser`：解析特征 Schema 定义

#### Arrow 序列化（`arrow/`）

封装 Apache Arrow 的 RecordBatch 和 Tensor 序列化/反序列化，用于 gRPC 传输和跨进程数据交换。

---

### 4.2 分布式训练框架（`cpp/metaspore/`）

这是 MetaSpore 分布式训练的 C++ 核心，实现了一套完整的参数服务器（Parameter Server）框架。

#### 分布式通信层

```
ActorProcess（协调中心）
    ├── MessageTransport（消息传输）
    │       └── ZeroMQTransport（ZeroMQ 实现）
    ├── NodeManager（节点管理 + 屏障同步）
    └── PSAgent（参数服务器代理）
```

- **`ActorProcess`**：分布式系统的协调中心，管理节点注册、屏障同步、消息路由
- **`PSAgent`**：参数服务器代理基类，处理 Push/Pull 请求，支持 `SendRequest`、`BroadcastRequest` 等操作
- **`NodeManager`**：管理 Worker/Server/Coordinator 节点的生命周期
- **`MessageTransport`**：消息传输抽象，底层使用 ZeroMQ 实现

#### 张量存储层

| 类 | 用途 |
|----|------|
| `DenseTensor` | 稠密张量（全连接层权重），支持 Push/Pull/Load/Save |
| `SparseTensor` | 稀疏张量（Embedding 表），支持按 key 查找和更新 |
| `DenseTensorPartition` | 稠密张量分区，用于分布式存储 |
| `SparseTensorPartition` | 稀疏张量分区，底层使用 `ArrayHashMap` |
| `TensorPartitionStore` | 张量分区存储管理 |

`SparseTensor` 提供了丰富的操作接口：
- `Push/Pull`：按 key 推送/拉取 Embedding 向量
- `PushPartition/PullPartition`：分区级别的批量操作
- `Load/Save/Export/ImportFrom`：模型持久化
- `PruneSmall/PruneOld`：稀疏表剪枝（按值大小或访问时间）

#### 文件系统抽象

- `Filesys`：文件系统抽象基类
- `LocalFilesys`：本地文件系统
- `S3SdkFilesys`：AWS S3 文件系统

#### Python 绑定

通过 pybind11 将 C++ 核心暴露给 Python：
- `ms_ps_python_bindings`：PS 框架绑定
- `tensor_store_python_bindings`：张量存储绑定
- `feature_extraction_python_bindings`：特征提取绑定

---

### 4.3 在线推理服务（`cpp/serving/`）

基于 gRPC + ONNX Runtime 的高性能在线推理服务。

#### 服务架构

```
gRPC 请求
    ↓
GrpcServer（异步 gRPC 服务器）
    ↓
ModelManager（单例，管理所有已加载模型）
    ↓
GrpcModelRunner（模型运行器，串联多个模型）
    ↓
具体模型（OrtModel / SparseLookupModel / FeatureExtractionModel）
    ↓
gRPC 响应
```

#### 模型基类设计

采用 CRTP（奇异递归模板模式）实现零开销多态：

```cpp
// 基类：定义接口
class ModelBase {
    virtual awaitable_status load(std::string dir_path) = 0;
    virtual awaitable_result<std::unique_ptr<ModelInputOutput>>
        predict(std::unique_ptr<ModelInputOutput> input) = 0;
};

// CRTP 模板：自动派发到具体类型
template <typename Model>
class ModelBaseCRTP : public ModelBase {
    // 自动处理类型检查和线程池调度
};
```

#### 支持的模型类型

| 模型类 | 功能 |
|--------|------|
| `OrtModel` | ONNX Runtime 模型推理（支持 CPU/GPU） |
| `SparseLookupModel` | 稀疏 Embedding 查找 |
| `SparseEmbeddingBagModel` | 稀疏 EmbeddingBag 聚合 |
| `TabularModel` | 表格数据模型 |
| `SparseFeatureExtractionModel` | 稀疏特征提取 |
| `DenseFeatureExtractionModel` | 稠密特征提取 |
| `PyPreprocessingModel` | Python 预处理模型 |
| `PyPreprocessingOrtModel` | Python 预处理 + ORT 组合模型 |

#### 数据转换器

`Converter` 体系负责在不同模型之间转换数据格式：

```
GrpcRequestOutput
    → GrpcRequestToFEConverter → FeatureExtractionModelInput
    → SparseFEToLookupConverter → SparseLookupModelInput
    → OrtToGrpcReplyConverter → GrpcReply
```

#### gRPC 接口定义（`protos/metaspore.proto`）

```protobuf
service Predict {
  rpc Predict(PredictRequest) returns (PredictReply) {}
}

message PredictRequest {
  string              model_name = 1;
  map<string, string> parameters = 3;
  map<string, bytes>  payload    = 5;
}

message PredictReply {
  map<string, bytes>  payload = 1;
  map<string, string> extras  = 3;
}
```

---

## 五、Python 训练框架

### 5.1 核心训练 API（`python/metaspore/`）

#### 分布式训练流程

```python
# 典型使用方式
agent = Agent(...)
module = MyModel()
model = Model(agent, module, experiment_name="my_exp")
trainer = DistributedTrainer(model, updater=SGDTensorUpdater(lr=0.01))
await trainer.train(dataset)
```

核心类：

| 类 | 职责 |
|----|------|
| `Agent` | 封装 C++ PSAgent，管理分布式通信 |
| `Model` | 封装 PyTorch Module + 分布式张量 |
| `DistributedTrainer` | 分布式训练器，协调前向/反向传播和参数同步 |
| `DistributedTensor` | Python 侧分布式张量封装 |

#### 张量初始化器（`initializer.py`）

- `ZeroTensorInitializer`：全零初始化
- `OneTensorInitializer`：全一初始化
- `DefaultTensorInitializer`：默认初始化策略

#### 张量更新器（`updater.py`）

- `SGDTensorUpdater`：随机梯度下降
- `EMATensorUpdater`：指数移动平均
- 支持自定义更新器

#### Embedding 层（`embedding.py`）

`EmbeddingOperator` 是稀疏特征学习的核心，将离散 ID 映射到稠密向量，底层对接 C++ `SparseTensor`。

### 5.2 算法库（`python/metaspore/algos/`）

内置了丰富的推荐系统算法：

**排序模型（CTR 预估）**

| 算法 | 文件 |
|------|------|
| Wide & Deep | `widedeep_net.py` |
| DeepFM | `deepfm_net.py` |
| xDeepFM | `xdeepfm_net.py` |
| DCN | `dcn_net.py` |
| DCN v2 | `dcn_v2_net.py` |
| AutoInt | `autoint_net.py` |
| PNN | `pnn_net.py` |
| FFM | `ffm_net.py` |
| FwFM | `fwfm_net.py` |

**召回模型**

- `twotower/`：双塔模型（向量召回）
- `item_cf_retrieval.py`：ItemCF 协同过滤

**其他**

- `sequential/`：序列模型（行为序列建模）
- `multitask/`：多任务学习
- `graph/`：图神经网络
- `feature/`：特征工程工具
- `pipeline/`：训练管道
- `tuner/`：超参数调优

### 5.3 PySpark 集成（`spark.py`）

提供 PySpark RDD/DataFrame 到训练数据的无缝转换，支持直接从数据湖读取训练样本。

---

## 六、Java 在线服务框架（`java/online-service/`）

Java 层负责在线业务逻辑，通过 gRPC 调用 C++ 推理服务。

| 模块 | 功能 |
|------|------|
| `serving/` | 在线模型服务，封装 gRPC 调用 |
| `feature-extract/` | 在线特征提取，对接特征存储 |
| `experiment-pipeline/` | 实验分桶和 A/B Test 流量管理 |
| `feature-service-sdk/` | 特征服务 SDK |
| `feature-source/` | 特征数据源适配 |
| `recommend-service/` | 推荐服务示例 |

---

## 七、构建系统

### CMake 配置

项目使用 CMake 3.20+ 构建，C++20 标准，主要构建目标：

| 目标 | 类型 | 内容 |
|------|------|------|
| `metaspore-common` | 静态库 | 通用工具、Arrow、特征计算、哈希表 |
| `metaspore_shared` | 共享库 | 分布式训练框架 + Python 绑定 |
| `metaspore-serving` | 静态库 | 在线推理服务 |
| `metaspore-serving-bin` | 可执行文件 | 推理服务二进制 |

构建选项：

```cmake
option(BUILD_TRAIN_PKG   "构建训练包"     ON)
option(BUILD_SERVING_BIN "构建服务二进制"  ON)
option(ENABLE_GPU        "启用 GPU 支持"  ON)
option(ENABLE_TESTS      "构建测试"       OFF)
```

编译优化：`-funroll-loops -march=core-avx2`（AVX2 向量化指令集）

### 依赖管理

通过 vcpkg 管理 C++ 依赖：

| 依赖 | 用途 |
|------|------|
| Boost.ASIO | 异步 IO 和 C++20 协程 |
| gRPC + Protobuf | RPC 框架 |
| Apache Arrow | 列式数据格式 |
| ONNX Runtime | 模型推理引擎 |
| ZeroMQ | 分布式消息队列 |
| Thrift | 序列化框架 |
| abseil | Google 基础库（StatusOr） |
| fmt / spdlog | 格式化和日志 |
| xtensor | 张量计算 |
| mimalloc | 高性能内存分配器 |
| range-v3 | 范围库 |

---

## 八、关键设计模式

### CRTP（奇异递归模板模式）

`ModelBaseCRTP` 和 `ConverterCRTP` 使用 CRTP 实现编译期多态，避免虚函数开销，同时自动处理类型检查：

```cpp
template <typename Model>
class ModelBaseCRTP : public ModelBase {
    // 自动将 predict() 调度到 Model::do_predict()
    // 并在线程池中异步执行
};
```

### 协程异步（Boost.ASIO Coroutines）

整个推理服务基于 C++20 协程构建，`awaitable<T>` 贯穿所有异步操作，避免回调地狱：

```cpp
awaitable_result<std::unique_ptr<ModelInputOutput>>
OrtModel::do_predict(std::unique_ptr<OrtModelInput> input) {
    // co_await 异步操作
    co_return result;
}
```

### Actor 模型

分布式训练层采用 Actor 模型，`ActorProcess` 作为协调中心，`PSAgent` 作为各节点的消息处理单元，通过 ZeroMQ 异步通信。

### 单例 + 读写锁

`ModelManager` 使用单例模式管理所有已加载模型，内部用 `std::shared_mutex` 实现读写分离，支持并发预测和动态加载。

---

## 九、性能优化亮点

1. **内存映射哈希表**：`MemoryMappedArrayHashMap` 支持超大规模稀疏 Embedding 表，无需全量加载到内存
2. **完美哈希**：`PerfectHashIndexBuilder` 为只读场景构建完美哈希索引，O(1) 查询无冲突
3. **AVX2 向量化**：编译时启用 AVX2 指令集，加速数值计算
4. **mimalloc**：替换系统 malloc，降低内存分配延迟
5. **线程池分离**：计算线程池和后台 IO 线程池分离，避免相互阻塞
6. **协程异步**：全链路异步 IO，最大化吞吐量
7. **稀疏表剪枝**：`PruneSmall/PruneOld` 定期清理低价值 Embedding，控制内存占用

---

## 十、端到端示例（`demo/`）

项目提供了多个完整的行业应用示例：

| 示例 | 场景 |
|------|------|
| `movielens/` | 电影推荐（召回 + 排序完整链路） |
| `ctr/` | CTR 点击率预估 |
| `twotower/` | 双塔向量召回 |
| `sequential/` | 序列行为建模 |
| `multitask/` | 多任务学习 |
| `graph/` | 图神经网络推荐 |
| `multimodal/` | 多模态推荐 |
| `ecommerce/` | 电商推荐场景 |
| `search/` | 搜索排序 |
| `text_classification/` | 文本分类 |
| `riskmodels/` | 风控模型 |

---

## 十一、总结

MetaSpore 是一个架构清晰、工程成熟的机器学习平台。其核心设计思路是：

- **C++ 负责性能**：分布式通信、张量存储、特征计算、推理服务全部用 C++ 实现，追求极致性能
- **Python 负责易用性**：通过 pybind11 将 C++ 能力暴露给 Python，提供友好的训练 API 和丰富的算法库
- **Java 负责在线业务**：在线特征服务、实验分桶等业务逻辑用 Java 实现，便于与现有后端系统集成
- **统一特征框架**：离线训练和在线推理共用同一套特征计算逻辑，从根本上解决特征一致性问题

对于想深入了解工业级推荐系统工程实现的开发者，MetaSpore 是一个非常值得研究的参考项目。
