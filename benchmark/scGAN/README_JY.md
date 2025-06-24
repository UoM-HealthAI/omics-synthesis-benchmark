# scGAN 工作流程详解

## 概述

本文档详细解释了 scGAN 数据预处理流程，特别是 `python main.py --param parameters_tabula_muris.json --process` 命令的执行过程。

## 📋 命令执行流程详解

### 🔍 1. 命令行参数解析

```bash
python main.py --param parameters_tabula_muris.json --process
```

**参数说明：**
- `--param`: 指定参数文件路径 `parameters_tabula_muris.json`
- `--process`: 启动数据预处理模式（设置为 `True`）

**代码位置：** `main.py` 第25-31行
```python
parser.add_argument('--param', required=True, help='Path to the parameters json file')
parser.add_argument('--process', required=False, default=False, action='store_true',
                   help='process the raw file and generate TF records for training')
```

### 🗂️ 2. 读取参数文件

**代码位置：** `main.py` 第61-66行
```python
with open(a.param, 'r') as fp:
    parameters = json.load(fp)

all_exp_dir = parameters['exp_param']['experiments_dir']  # 输出目录
GPU_NB = parameters['exp_param']['GPU']                   # GPU配置  
experiments = parameters['experiments']                   # 实验配置
```

**功能：**
- 解析 JSON 参数文件
- 提取实验目录、GPU配置、实验参数等信息

### 📁 3. 创建实验目录结构

**代码位置：** `main.py` 第77-91行
```python
if a.process:
    try:
        os.makedirs(exp_dir)  # 创建实验目录
    except OSError:
        raise OSError('实验目录已存在错误')
    
    # 复制原始数据文件到实验目录
    copyfile(raw_input, os.path.join(exp_dir, raw_file_name))
    
    # 在实验目录创建 parameters.json 文件
    with open(os.path.join(exp_dir, 'parameters.json'), 'w') as fp:
        fp.write(json.dumps(exp_param, sort_keys=True, indent=4))
```

**功能：**
- 创建实验输出目录
- 复制原始 h5ad 数据文件到实验目录
- 保存实验参数到 parameters.json

### 🔄 4. 启动数据预处理

**代码位置：** `main.py` 第93-94行
```python
if a.process:
    process_files(exp_folders)  # 调用预处理函数
```

**功能：**
- 调用 `preprocessing/write_tfrecords.py` 中的 `process_files` 函数
- 启动并行数据预处理流程

## 📊 数据预处理详细流程

### 5.1 GeneMatrix 数据加载 

**代码位置：** `preprocessing/process_raw.py`
```python
sc_data = GeneMatrix(job_path)    # 创建数据矩阵对象
sc_data.apply_preprocessing()     # 应用预处理步骤
```

**GeneMatrix 类功能：**
- 读取 h5ad 格式的单细胞数据
- 解析实验参数
- 应用质量控制和过滤
- 数据标准化和分割

### 5.2 数据预处理步骤

1. **读取 h5ad 文件**: 从 `raw_input` 路径加载 Tabula Muris 数据
2. **基因过滤**: 移除表达细胞数 < `min_cells` 的基因
3. **细胞过滤**: 移除表达基因数 < `min_genes` 的细胞  
4. **数据标准化**: 应用参数指定的标准化方法（如 `normalize_per_cell_LS_200`）
5. **数据分割**: 分为训练集/验证集/测试集

### 5.3 TFRecords 文件创建

**代码位置：** `preprocessing/write_tfrecords.py`
```python
worker_path = join(job_path, 'TF_records')  # 创建 TF_records 目录
os.makedirs(worker_path, exist_ok=True)

with TFRFWriter(worker_path, categories=cat) as writer:
    for line in sc_data.sc_raw:          # 遍历每个细胞
        sc_genes, d = process_line(line)  # 处理单个细胞数据
        writer.write_numpy(sc_genes, d.barcode, d.count_no,
                          d.genes_no, d.dset, d.cluster)  # 写入TFRecord
```

**TFRFWriter 类功能：**
- 将数据分片为多个 TFRecord 文件（默认10个训练文件）
- 压缩存储（GZIP格式）
- 保存稀疏矩阵格式数据

## 📂 输出文件结构

执行完成后，会在实验目录下生成以下文件结构：

```
experiment_output_dir/
├── parameters.json           # 实验参数文件
├── all_converted.h5ad       # 复制的原始数据文件  
└── TF_records/              # TensorFlow Records 目录
    ├── train-0.tfrecords    # 训练数据分片0
    ├── train-1.tfrecords    # 训练数据分片1
    ├── train-2.tfrecords    # 训练数据分片2
    ├── ...                  # 更多训练分片
    ├── train-9.tfrecords    # 训练数据分片9
    ├── validate.tfrecords   # 验证数据
    └── test.tfrecords       # 测试数据
```

## 🎯 关键数据转换

### 稀疏矩阵转换

每个细胞的基因表达数据都会转换为稀疏格式以节省存储空间：

```python
# 提取非零值位置和数值
idx, vals = to_sparse(scg_line)  
feat_map['indices'] = tf.train.Feature(int64_list=tf.train.Int64List(value=idx))
feat_map['values'] = tf.train.Feature(float_list=tf.train.FloatList(value=vals))
```

### 元数据保存

每个 TFRecord 包含以下信息：

```python
feat_map['barcode'] = # 细胞条形码（唯一标识符）
feat_map['genes_no'] = # 该细胞表达的基因数量  
feat_map['count_no'] = # 该细胞的总表达计数
feat_map['cluster_int'] = # 细胞类型标签（整数编码）
feat_map['cluster_1hot'] = # 细胞类型的one-hot编码
```

## ⚡ 并行处理

数据预处理支持多进程并行处理：

```python
# 使用多进程池并行处理多个实验
pool = mp.Pool()
results = pool.imap_unordered(read_and_serialize, exp_folders)
```

这大大加快了大型数据集的处理速度。

## 📋 参数文件示例

典型的 `parameters_tabula_muris.json` 文件结构：

```json
{
  "exp_param": {
    "experiments_dir": "/path/to/output",
    "GPU": [0]
  },
  "experiments": {
    "experiment_name": {
      "input_ds": {
        "raw_input": "data/tabula_muris/all_converted.h5ad",
        "scale": "normalize_per_cell_LS_200",
        "filtering": {
          "min_cells": 3,
          "min_genes": 10
        },
        "split": {
          "test_cells": 100,
          "valid_cells": 100
        }
      }
    }
  }
}
```

## ✅ 执行成功标志

执行成功后，你会看到：

1. **控制台输出**: `done with writing for: /path/to/experiment`
2. **文件生成**: TF_records 目录下包含所有必要的 .tfrecords 文件
3. **无错误**: 没有 Python 异常或错误信息

## 🔧 常见问题排查

### 问题1: 内存不足
**症状**: `MemoryError` 或进程被杀死
**解决**: 减小批处理大小或增加系统内存

### 问题2: 文件路径错误
**症状**: `FileNotFoundError`
**解决**: 检查参数文件中的 `raw_input` 路径是否正确

### 问题3: 权限错误
**症状**: `PermissionError`
**解决**: 确保对输出目录有写权限

### 问题4: TensorFlow 版本兼容性
**症状**: TensorFlow 相关错误
**解决**: 确保使用兼容的 TensorFlow 版本（推荐 2.6.0）

## 🚀 后续步骤

数据预处理完成后，可以进行：

1. **模型训练**: `python main.py --param parameters_tabula_muris.json --train`
2. **数据生成**: `python main.py --param parameters_tabula_muris.json --generate`
3. **结果分析**: 使用生成的 TFRecords 进行模型训练和评估

## 📚 相关文件

- `main.py`: 主入口脚本
- `preprocessing/process_raw.py`: GeneMatrix 类定义
- `preprocessing/write_tfrecords.py`: TFRecords 写入逻辑
- `estimators/run_exp.py`: 实验运行逻辑

---

*本文档由 Claude Code 生成，用于详细解释 scGAN 数据预处理工作流程。*