# MetaSpore Training Release 容器手动训练验证手册

本文档说明如何**手动进入** `dmetasoul/metaspore-training-release:v1.0.0` 容器，并一步一步验证训练与导出流程。

## 1. 前提条件

- 本地已构建镜像：`dmetasoul/metaspore-training-release:v1.0.0`
- 本机已安装 Docker Desktop 并可运行容器
- 代码仓库路径（示例）：
  - `/System/Volumes/Data/data/code/public/ai/MetaSpore`
- 如果 Docker Desktop 未共享该路径，请改用已共享路径（例如 `/private/tmp`）做挂载

可先检查镜像：

```bash
docker images | grep "dmetasoul/metaspore-training-release"
```

---

## 2. 在宿主机准备验证用目录与数据

> 说明：以下用 `/private/tmp/metaspore-verify` 作为容器挂载目录，避免 macOS 路径共享问题。

```bash
rm -rf /private/tmp/metaspore-verify
mkdir -p /private/tmp/metaspore-verify/tests/schema/wdl
mkdir -p /private/tmp/metaspore-verify/tests/data
```

拷贝测试脚本与 schema 文件：

```bash
cp /System/Volumes/Data/data/code/public/ai/MetaSpore/python/tests/sparse_wdl_export_test.py \
   /private/tmp/metaspore-verify/tests/

cp /System/Volumes/Data/data/code/public/ai/MetaSpore/cpp/tests/schema/wdl/column_name_demo.txt \
   /System/Volumes/Data/data/code/public/ai/MetaSpore/cpp/tests/schema/wdl/combine_schema_demo.txt \
   /private/tmp/metaspore-verify/tests/schema/wdl/
```

生成一份小规模合成训练/测试数据（40 列，`\t` 分隔）：

```bash
python3 << 'PY'
import random, os
random.seed(42)
base='/private/tmp/metaspore-verify/tests/data'
os.makedirs(base, exist_ok=True)

def row():
    label=str(random.randint(0,1))
    ints=[str(random.randint(0,100)) for _ in range(13)]
    cats=[str(random.randint(0,9999)) for _ in range(26)]
    return '\t'.join([label]+ints+cats)

for name,n in [('day_0_0.001_train_head1000.csv',1000),('day_0_0.001_test_head10.csv',10)]:
    with open(os.path.join(base,name),'w') as f:
        for _ in range(n):
            f.write(row()+'\n')
print('prepared')
PY
```

---

## 3. 手动进入容器

```bash
docker run --rm -it \
  -v /private/tmp/metaspore-verify/tests:/workspace \
  -w /workspace \
  -e TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
  dmetasoul/metaspore-training-release:v1.0.0 \
  bash
```

进入后可先检查版本：

```bash
java -version
cat /opt/spark/RELEASE
python -V
python -m pip show pyspark | sed -n '1,3p'
```

---

## 4. 容器内执行训练验证（手动）

安装导出依赖（如果镜像里还没有）：

```bash
python -m pip install onnxscript
```

准备输出目录：

```bash
rm -rf output
mkdir -p output/model_out output/model_export
```

执行训练脚本：

```bash
spark-submit --master "local[2]" --driver-memory 4g sparse_wdl_export_test.py
```

---

## 5. 验证输出结果

训练结束后，检查模型输出和导出目录：

```bash
echo "model_out:"
ls -la output/model_out | head

echo "model_export:"
ls -la output/model_export
```

预期 `output/model_export` 至少包含：

- `_dense`
- `sparse__sparse`
- `sparse_lr_layer`

可以进一步确认 ONNX 文件存在：

```bash
find output/model_export -name "*.onnx" -maxdepth 3
```

---

## 6. 常见问题与处理

- **路径无法挂载（mounts denied）**
  - 现象：`The path ... is not shared from the host`
  - 处理：改用 Docker Desktop 已共享路径（推荐 `/private/tmp`），或在 Docker Desktop 里添加共享目录

- **`one of schema and column_names must be specified`**
  - 说明：`read_s3_csv` 新版本要求传 `column_names` 或 `schema`
  - 处理：使用已修复后的测试脚本（仓库中的 `python/tests/sparse_wdl_export_test.py`）

- **`torch.load` 的 weights_only 报错**
  - 处理：运行容器时增加环境变量  
    `-e TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`

- **`bkdr_hash ... large_string` 报错**
  - 处理：使用仓库中已修复的 `python/metaspore/embedding.py`（已做 Arrow string 类型兼容）

- **ONNX 导出 ScriptModule 报错**
  - 处理：确保导出代码中 `torch.onnx.export(..., dynamo=False, ...)`（`embedding.py` 与 `model.py` 均已处理）

---

## 7. 一键方式（可选）

如果不需要交互式进入容器，也可以直接一条命令执行：

```bash
docker run --rm \
  -v /private/tmp/metaspore-verify/tests:/workspace \
  -w /workspace \
  -e TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
  dmetasoul/metaspore-training-release:v1.0.0 \
  bash -ec 'python -m pip install -q onnxscript && rm -rf output && mkdir -p output/model_out output/model_export && spark-submit --master "local[2]" --driver-memory 4g sparse_wdl_export_test.py && ls -la output/model_export'
```

