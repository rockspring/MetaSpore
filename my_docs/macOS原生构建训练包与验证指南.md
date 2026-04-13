# macOS 原生构建 MetaSpore 训练安装包与验证指南

本文档说明在 **Apple 芯片（arm64）或 Intel（x86_64）** 的 macOS 上，从安装系统与开发工具开始，到用 **vcpkg 清单模式 + 仓库自带 overlay-ports** 完成 CMake 构建、产出 Python wheel、安装并跑通 **`sparse_wdl_export_test`** 训练示例的完整流程。

> 说明：`vcpkg.json` 中的 `builtin-baseline` 与仓库根目录下的 `vcpkg-configuration.json`（`./overlay-ports`）组合已在 Ubuntu 等平台验证；macOS 上同样**不要**随意改动该组合，只需保证本地 **vcpkg 仓库检出到 baseline 提交**（见下文第 4 节）。

---

## 1. 适用环境与假设

- macOS（本文以 **arm64 + Homebrew 在 `/opt/homebrew`** 为主；Intel Mac 请将 triplet 改为 `x64-osx`，工具路径一般为 `/usr/local`）。
- 已安装 **Xcode 命令行工具**（提供 `clang`、`git`、`make` 等）：
  ```bash
  xcode-select --install
  ```
- 网络可访问 GitHub（clone vcpkg、拉取 port 源码）。

---

## 2. 安装 Homebrew 与通用构建工具

若未安装 [Homebrew](https://brew.sh/)，按官网说明安装后执行：

```bash
brew update
brew install cmake ninja git
```

**vcpkg 在编译部分依赖时**可能会要求系统提供 `autoconf` / `automake` / `libtool` 等（例如 `libb2`）。若后续 CMake 或 vcpkg 报错提示缺少这些程序，请安装：

```bash
brew install autoconf autoconf-archive automake libtool
```

确认 `ninja` 在 `PATH` 中（Apple Silicon 示例）：

```bash
export PATH="/opt/homebrew/bin:$PATH"
which cmake ninja
```

---

## 3. Conda 环境与 Python 3.12

使用 **Python 3.12** 的 Conda 环境（名称自定，这里以 `ms312` 为例）：

```bash
conda create -n ms312 python=3.12 -y
conda activate ms312
```

构建阶段 CMake 会通过 Python 探测 **PyArrow** 的头文件与动态库路径，请提前安装：

```bash
pip install -U pip
pip install numpy pyarrow
```

训练示例依赖 **PyTorch**、**PySpark**（以及常见 ML 栈），建议在验证训练前安装（版本可按项目要求微调）：

```bash
pip install torch pyspark
```

> **PySpark 与 Python 版本一致**：本地 Spark worker 会使用 `python3`。若 driver 为 Conda 的 3.12，而系统默认 `python3` 为其它版本，可能出现 “Python in worker has different version” 错误。请固定为当前环境的解释器，例如：
>
> ```bash
> export PYSPARK_PYTHON="$(which python)"
> export PYSPARK_DRIVER_PYTHON="$(which python)"
> ```

---

## 4. 安装 vcpkg 并对齐 `vcpkg.json` 的 baseline

仓库根目录 `vcpkg.json` 内含 **`builtin-baseline`**（提交哈希）。**必须**让本地 `$VCPKG_ROOT` 指向的 vcpkg 仓库处于该提交（或兼容历史），否则 manifest 模式可能报无法检出 `versions/baseline.json` 等错误。

```bash
export VCPKG_ROOT="$HOME/vcpkg"

if [ ! -d "$VCPKG_ROOT/.git" ]; then
  git clone https://github.com/microsoft/vcpkg.git "$VCPKG_ROOT"
fi

"$VCPKG_ROOT/bootstrap-vcpkg.sh"
```

从 **MetaSpore 仓库根目录** 读取 baseline 并检出（将 `META_SPORE_HOME` 换成你的克隆路径）：

```bash
export META_SPORE_HOME="/path/to/MetaSpore"
export META_SPORE_HOME="$(pwd)"
BASELINE="$(python3 -c "import json; print(json.load(open('$META_SPORE_HOME/vcpkg.json'))['builtin-baseline'])")"
git -C "$VCPKG_ROOT" fetch origin
git -C "$VCPKG_ROOT" checkout "$BASELINE"
```

若 vcpkg 是 **浅克隆** 且 `checkout` 失败，可先 **`git -C "$VCPKG_ROOT" fetch --unshallow origin`**（或重新完整 clone），再执行 `checkout "$BASELINE"`。

每次拉取 MetaSpore 新代码后，若 `builtin-baseline` 变更，需对 vcpkg 仓库重复执行 **fetch + checkout**。

---

## 5. vcpkg triplet 与 CMake 前置环境变量

**Apple Silicon（arm64）**：

```bash
export VCPKG_DEFAULT_TRIPLET=arm64-osx
```

**Intel Mac（x86_64）**：

```bash
export VCPKG_DEFAULT_TRIPLET=x64-osx
```

统一设置（请将 `META_SPORE_HOME` 改为本机路径）：

```bash
export META_SPORE_HOME="/path/to/MetaSpore"
export VCPKG_ROOT="$HOME/vcpkg"
export PATH="/opt/homebrew/bin:$PATH"   # Intel 可改为 /usr/local/bin
```

建议在配置前清空构建目录，避免旧的 CMake 缓存与 triplet 不一致：

```bash
rm -rf "$META_SPORE_HOME/build-vcpkg-verify"
```

---

## 6. CMake 配置与编译（对齐训练包场景）

参考 `docker/ubuntu24.04/Dockerfile_training_build`，训练包构建通常会关闭测试与 Serving 二进制，仅保留训练相关目标。镜像中使用 `RelWithDebInfo`，本地调试可用 `Debug`；以下为与 Docker **选项对齐**的示例（**Release 类型可改为 `RelWithDebInfo`**）：

```bash
cmake -S "$META_SPORE_HOME" -B "$META_SPORE_HOME/build-vcpkg-verify" \
  -G Ninja \
  -DCMAKE_MAKE_PROGRAM="$(which ninja)" \
  -DCMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DPython_EXECUTABLE=$(which python3) \
  -DENABLE_TESTS=OFF \
  -DBUILD_SERVING_BIN=OFF \
  -DBUILD_TRAIN_PKG=ON \
  -DONNX_DISABLE_STATIC_REGISTRATION=ON
```

说明：

- **首次配置**时，vcpkg 会根据 `vcpkg.json` + `vcpkg-configuration.json` 自动安装依赖，耗时较长。
- **Overlay**：仓库内 `vcpkg-configuration.json` 已指定 `"./overlay-ports"`，无需额外传参；请保持从 **`META_SPORE_HOME` 作为 `-S` 源码目录** 调用 CMake，以便 vcpkg 读到该配置。

仅构建 Python wheel（与 Dockerfile 中 `python_wheel` 目标一致）：

```bash
cmake --build "$META_SPORE_HOME/build-vcpkg-verify" --target python_wheel --parallel
```

若需完整默认目标，可省略 `--target python_wheel`。

构建成功后，`build-vcpkg-verify` 目录下会出现类似：

`metaspore-1.2.0-cp312-cp312-macosx_*_arm64.whl`（具体 `macosx` 次版本号随系统/SDK 变化，以实际文件名为准）。

---

## 7. 安装 wheel 并确认 Python 可导入

```bash
conda activate ms312
pip install --force-reinstall "$META_SPORE_HOME/build-vcpkg-verify"/metaspore-*.whl
python -c "import metaspore as ms; print('metaspore', ms.__file__)"
```

> 若 `pip` 提示找不到 wheel 文件，请用 `ls "$META_SPORE_HOME/build-vcpkg-verify"/*.whl` 确认精确文件名后再安装。

---

## 8. 准备 `sparse_wdl_export_test` 的数据与目录

脚本 `python/tests/sparse_wdl_export_test.py` 内 **`S3_ROOT_DIR = './'`**，因此必须在「工作目录」下具备如下相对路径：

- `schema/wdl/column_name_demo.txt`
- `schema/wdl/combine_schema_demo.txt`
- `data/day_0_0.001_train_head1000.csv`
- `data/day_0_0.001_test_head10.csv`

推荐在临时目录搭一套最小文件集（与 `my_docs/training_release_手动训练验证手册.md` 思路一致）：

```bash
VERIFY_ROOT="/private/tmp/metaspore-mac-verify"
rm -rf "$VERIFY_ROOT"
mkdir -p "$VERIFY_ROOT/schema/wdl" "$VERIFY_ROOT/data"

cp "$META_SPORE_HOME/cpp/tests/schema/wdl/column_name_demo.txt" \
   "$META_SPORE_HOME/cpp/tests/schema/wdl/combine_schema_demo.txt" \
   "$VERIFY_ROOT/schema/wdl/"
```

生成小规模合成数据（40 列、制表符分隔；与测试脚本一致）：

```bash
python3 << 'PY'
import random, os
random.seed(42)
base = "/private/tmp/metaspore-mac-verify/data"
os.makedirs(base, exist_ok=True)

def row():
    label = str(random.randint(0, 1))
    ints = [str(random.randint(0, 100)) for _ in range(13)]
    cats = [str(random.randint(0, 9999)) for _ in range(26)]
    return "\t".join([label] + ints + cats)

for name, n in [
    ("day_0_0.001_train_head1000.csv", 1000),
    ("day_0_0.001_test_head10.csv", 10),
]:
    with open(os.path.join(base, name), "w") as f:
        for _ in range(n):
            f.write(row() + "\n")
PY
```

---

## 9. 运行训练验证脚本

```bash
conda activate ms312
export PYSPARK_PYTHON="$(which python)"
export PYSPARK_DRIVER_PYTHON="$(which python)"

cd /private/tmp/metaspore-mac-verify
python "$META_SPORE_HOME/python/tests/sparse_wdl_export_test.py"
```

预期现象：脚本会拉起本地 Spark、训练若干 step、打印评估指标，并对测试集执行 `transform`、`show()`。进程 **退出码为 0** 即表示本机训练链路基本打通。

---

## 10. 常见问题与排查

| 现象 | 处理方向 |
|------|----------|
| vcpkg 报错无法 checkout `builtin-baseline` | 对 `$VCPKG_ROOT` 执行 `git fetch`，并 **checkout 到 `vcpkg.json` 中的提交** |
| CMake 找不到 Ninja | `brew install ninja`，并将 `/opt/homebrew/bin` 加入 `PATH`，或设置 `-DCMAKE_MAKE_PROGRAM` |
| 某 port 需要 autoconf/automake/libtool | `brew install autoconf autoconf-archive automake libtool` 后重试构建 |
| 配置阶段提示找不到 `pyarrow` | 在 **当前 CMake 使用的同一 Python** 环境中 `pip install pyarrow numpy` |
| 打 wheel 时报 `_metaspore.so` 不存在 | 仓库需包含对 macOS 的修复：wheel 步骤应使用 `_metaspore.dylib`（由 `CMAKE_SHARED_LIBRARY_SUFFIX` 决定），请更新到已修复的 `cmake/python_wheel.cmake` |
| `torch.load` / `Weights only load failed`（PyTorch 2.6+） | 训练代码中对 cloudpickle 序列化的 module 需 `weights_only=False`；请使用已修复的 `python/metaspore/estimator.py` |
| `socket.gaierror` 解析本机 hostname | 使用已修复的 `python/metaspore/network_utils.py`（或在本机 `/etc/hosts` 配置 hostname） |
| `metaspore` 无 `PyTorchEstimator` | 通常表示 **未安装 pyspark**；安装并保证能 `import pyspark` 后重新打开 Python；`__init__.py` 仅在 pyspark 可用时导出 Estimator |
| PySpark worker Python 版本与 driver 不一致 | 设置 `PYSPARK_PYTHON` 与 `PYSPARK_DRIVER_PYTHON` 为同一 Conda `python` |

---

## 11. 与 Docker 构建的对应关系（便于对照）

| 项目 | Docker（`Dockerfile_training_build`） | 本指南（macOS） |
|------|----------------------------------------|-----------------|
| vcpkg 依赖 | 使用事先安装的 triplet 目录 | manifest + `VCPKG_ROOT` toolchain 自动安装 |
| CMake 类型 | `RelWithDebInfo` | 推荐 `RelWithDebInfo`，本地可选用 `Debug` |
| 训练相关选项 | `ENABLE_TESTS=OFF`, `BUILD_SERVING_BIN=OFF`, `BUILD_TRAIN_PKG=ON` | 同上 |
| 产物 | `python_wheel` | `cmake --build ... --target python_wheel` |

---

## 12. 参考文件

- `vcpkg.json` — 依赖与 `builtin-baseline`
- `vcpkg-configuration.json` — `overlay-ports` 路径
- `docker/ubuntu24.04/Dockerfile_training_build` — CMake 选项参考
- `my_docs/training_release_手动训练验证手册.md` — 容器内验证思路（数据准备可复用）
- `python/tests/sparse_wdl_export_test.py` — 端到端训练与导出测试脚本

按上述步骤，即可在 macOS 上完成 **工具链安装 → vcpkg 对齐 → 构建 wheel → pip 安装 → 本地训练脚本验证** 的完整闭环。
