# ACTIVA.py 代码分析文档

## 概述

ACTIVA是一个用于单细胞RNA测序数据的条件生成模型，结合了IntroVAE（内省变分自编码器）和细胞类型分类器来生成高质量的单细胞数据。该模型通过对抗训练和条件约束来确保生成数据的真实性和生物学意义。

## 1. 数据读取与处理

### 1.1 支持的数据格式
- **主要格式**: Scanpy格式 (`.h5ad`文件) - 默认选项
- **备选格式**: CSV格式 (`.h5` + `.csv`标签文件)

### 1.2 预设数据集
代码支持4个预设的数据集例子：

1. **PBMC数据集** (`--example_data pbmc`)
   - 路径: `/home/ubuntu/scGAN_ProcessedData/MADE_BY_scGAN/68kPBMCs_7kTest.h5ad`
   - 细胞类型数: 10个
   - 约68K个PBMC细胞

2. **小鼠脑数据集** (`--example_data '20k brain'`)
   - 路径: `/home/ubuntu/scGAN_ProcessedData/MADE_BY_scGAN/20Kneurons_2KTest.h5`
   - 细胞类型数: 8个
   - 约20K个神经元

3. **COVID数据集** (`--example_data covid`) - 默认数据集
   - 路径: `/home/ubuntu/COVID_Data/NeuroCOVID/TrainSplitData/NeuroCOVID_preprocessed_splitted.h5ad`
   - 细胞类型数: 16个
   - 约78K个NeuroCOVID数据
   - 数据来源: https://doi.org/10.1016/j.immuni.2020.12.011

4. **Tabula Muris数据集** (`--example_data tabula_muris`)
   - 路径: `/ceph/hpc/home/eujinyuanw/omics_analysis/benchmark/ACTIVA/data/tabula_muris_split.h5ad`
   - 细胞类型数: 12个 (Bladder, Heart_and_Aorta, Kidney, 等)
   - 约57K个来自12种组织类型的小鼠细胞

### 1.3 数据加载过程
```python
train_data_loader, valid_data_loader = Scanpy_IO(
    data_path,
    test_no_valid=True,
    batchSize=opt.batchSize,  # 默认128
    workers=opt.workers,      # 默认24
    log=False,
    verbose=1
)
```

### 1.4 数据预处理
- 通过`Scanpy_IO`函数创建PyTorch DataLoader
- 自动确定输入特征数量（基因数量）
- 自动获取细胞类型标签信息
- 支持训练/验证集分割

## 2. 模型架构

### 2.1 ACTIVA生成模型
```python
model = ACTIVA(
    latent_dim=opt.zdim,        # 潜在空间维度，默认128
    input_size=inp_size,        # 输入特征数（基因数量）
    threshold=0                 # 计数矩阵阈值
).to(device)
```

**组件**:
- **编码器（Encoder）**: 将输入数据编码到潜在空间
- **解码器（Decoder）**: 从潜在空间重构数据
- **基于IntroVAE架构**: 支持对抗训练和内省机制

### 2.2 分类器模型
```python
cf_model = Classifier(
    output_dim=number_of_classes,  # 细胞类型数量
    input_size=inp_size           # 输入特征数
).to(device)
```

**作用**:
- 预测细胞类型
- 为生成过程提供条件约束
- 确保生成数据的生物学一致性

## 3. 训练流程

### 3.1 三阶段训练策略

训练过程分为三个阶段：

#### 阶段1: 分类器预训练 (前`num_cf`个epochs，默认5个)
- **目标**: 训练细胞类型分类器
- **损失函数**: 交叉熵损失
- **优化器**: Adam优化器，学习率0.0001
- **学习率衰减**: 每1000步衰减，衰减率0.95

#### 阶段2: VAE预训练 (前`num_vae`个epochs，默认10个)
- **目标**: 预训练变分自编码器部分
- **损失函数**: 重构损失 + KL散度损失
- **优化器**: 分别为编码器和解码器使用Adam优化器，学习率0.0002

#### 阶段3: 完整ACTIVA训练 (剩余epochs，默认总共200个)
- **目标**: 训练完整的条件生成模型
- **结合**: IntroVAE对抗训练 + 分类器条件约束

### 3.2 仅分类器训练模式
```bash
python ACTIVA.py --classifierOnly True --classifierEpochs 10
```
- 可选择只训练分类器而不训练生成模型
- 用于评估分类器性能或快速原型验证

## 4. 损失函数详解

### 4.1 分类器训练损失
```python
loss = cf_criterion(pred_cluster.squeeze(), true_labels)
```
- **类型**: 交叉熵损失
- **作用**: 优化细胞类型分类准确性

### 4.2 VAE训练损失
```python
loss_rec = model.reconstruction_loss(rec, real, True)
loss_kl = model.kl_loss(real_mu, real_logvar).mean()
loss = loss_rec + loss_kl
```
- **重构损失**: 确保重构数据与原始数据相似
- **KL散度损失**: 正则化潜在空间分布

### 4.3 完整ACTIVA训练损失

#### 编码器损失
```python
lossE = opt.alpha_1 * loss_margin + opt.alpha_2 * (loss_classification + loss_rec)
```

其中：
- **边际损失（Margin Loss）**: 
  ```python
  loss_margin = lossE_real_kl + \
                (F.relu(opt.m_plus-lossE_rec_kl) + \
                 F.relu(opt.m_plus-lossE_fake_kl)) * 0.5 * opt.weight_neg
  ```
- **分类损失**: 确保重构数据保持原始细胞类型特征
- **重构损失**: 保持数据重构质量

#### 解码器损失
```python
lossG = opt.alpha_1 * 0.5 * (lossG_rec_kl + lossG_fake_kl)
```

### 4.4 关键超参数
- `m_plus = 150.0`: 边际参数，控制对抗训练强度
- `alpha_1 = weight_kl = 1.0`: KL损失权重
- `alpha_2 = weight_rec = 0.05`: 重构和分类损失权重
- `weight_neg = 0.5`: 负样本权重

## 5. 条件生成机制

### 5.1 条件约束步骤
```python
# 获取真实数据的分类结果
cf_model.eval()
real_classification = cf_model(batch.to(device))

# 获取重构数据的分类结果
rec_classification = cf_model(rec)

# 计算分类一致性损失
loss_classification = model.classification_loss(rec_classification, real_classification)
```

### 5.2 生成过程
1. **编码**: 真实数据 → 潜在表示
2. **解码**: 潜在表示 → 重构数据
3. **分类约束**: 确保重构数据保持原始细胞类型
4. **对抗训练**: 通过边际损失提高生成质量

## 6. 模型保存与评估

### 6.1 模型检查点保存
- **ACTIVA模型**: 每`print_frequency`个epochs保存 (默认25)
- **分类器模型**: 每`cf_print_frequency`个epochs保存 (默认5)
- **保存格式**: 包含模型权重、优化器状态等

### 6.2 模型评估
- **分类器评估**: 在验证集上评估分类准确性
- **生成质量**: 通过重构损失和分类一致性评估

## 7. 使用的训练文件

### 7.1 核心依赖模块
- `ACTIVA`: 主要的生成模型实现
- `ACTIVA.utils`: 工具函数（模型保存/加载等）
- `ACTINN`: 分类器实现和数据加载
- `cpu_patch`: CPU运行补丁

### 7.2 外部依赖
- `torch`: PyTorch深度学习框架
- `scanpy`: 单细胞数据处理
- `tensorboardX`: 训练监控和可视化
- `numpy`: 数值计算
- `tqdm`: 进度条显示

## 8. 运行示例

### 8.1 基本运行
```bash
python ACTIVA.py --example_data tabula_muris --nEpochs 200
```

### 8.2 自定义参数运行
```bash
python ACTIVA.py \
    --example_data tabula_muris \
    --nEpochs 200 \
    --batchSize 64 \
    --lr_e 0.0001 \
    --lr_g 0.0001 \
    --zdim 256 \
    --tensorboard
```

### 8.3 GPU/CPU选择
- 默认使用GPU（如果可用）
- 使用`--cpu`强制使用CPU训练

## 9. 输出文件

### 9.1 模型检查点
- 保存在当前目录或指定路径
- 包含完整的模型状态用于后续推理

### 9.2 TensorBoard日志
- 保存在`--outf`指定目录（默认`./withL2-TensorBoard-z128/`）
- 包含训练损失、学习率等监控信息

## 10. 总结

ACTIVA是一个复杂的条件生成模型，专门设计用于单细胞RNA测序数据的生成和分析。它通过结合变分自编码器、对抗训练和细胞类型分类器，能够生成既保持原始数据统计特性又符合生物学约束的高质量单细胞数据。该模型特别适用于数据增强、罕见细胞类型生成和跨数据集转换等任务。