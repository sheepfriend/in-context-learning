# 实验配置指南

## 实验设置

`run_experiments.py` 脚本已配置为运行以下实验：

### 实验参数

| 参数 | 取值 |
|------|------|
| **V** (图的数量) | 5, 20 |
| **num_training_examples** | 256, 512, 1024, 2048, 4096 |
| **模型类型** | `gpt2` (标准), `lowrank_gpt2` (低秩) |
| **每个配置运行次数** | 3 |

### 总实验数量

```
总数 = 2 (V值) × 5 (样本数) × 2 (模型) × 3 (重复) = 60 个实验
```

### 实验矩阵

#### V = 5

| num_examples | gpt2 (标准) | lowrank_gpt2 | 每个配置重复 |
|--------------|-------------|--------------|--------------|
| 256          | ✓           | ✓            | 3次          |
| 512          | ✓           | ✓            | 3次          |
| 1024         | ✓           | ✓            | 3次          |
| 2048         | ✓           | ✓            | 3次          |
| 4096         | ✓           | ✓            | 3次          |

#### V = 20

| num_examples | gpt2 (标准) | lowrank_gpt2 | 每个配置重复 |
|--------------|-------------|--------------|--------------|
| 256          | ✓           | ✓            | 3次          |
| 512          | ✓           | ✓            | 3次          |
| 1024         | ✓           | ✓            | 3次          |
| 2048         | ✓           | ✓            | 3次          |
| 4096         | ✓           | ✓            | 3次          |

## 运行方式

### 方法 1: 直接运行所有实验

```bash
python run_experiments.py
```

这将按顺序运行所有60个实验。

### 方法 2: 使用后台运行

```bash
nohup python run_experiments.py > experiments.log 2>&1 &
```

实验将在后台运行，所有输出保存到 `experiments.log`。

### 方法 3: 使用 screen 或 tmux

```bash
# 使用 screen
screen -S experiments
python run_experiments.py

# 使用 tmux
tmux new -s experiments
python run_experiments.py
```

## 输出格式

每个实验的命名格式：
```
table_connectivity_{model_tag}_V{V}_N{num_examples}_run{run_idx}
```

示例：
- `table_connectivity_standard_V5_N256_run1` - 标准模型，V=5，256个样本，第1次运行
- `table_connectivity_lowrank_V20_N4096_run3` - 低秩模型，V=20，4096个样本，第3次运行

## 输出目录结构

```
../models/
├── table_connectivity/                    # 标准模型输出
│   ├── table_connectivity_standard_V5_N256_run1/
│   ├── table_connectivity_standard_V5_N256_run2/
│   ├── ...
│   └── table_connectivity_standard_V20_N4096_run3/
│
└── table_connectivity_lowrank/            # 低秩模型输出
    ├── table_connectivity_lowrank_V5_N256_run1/
    ├── table_connectivity_lowrank_V5_N256_run2/
    ├── ...
    └── table_connectivity_lowrank_V20_N4096_run3/
```

## 实验执行顺序

实验按以下顺序执行：

1. **V=5, num_examples=256**
   - gpt2: run1, run2, run3
   - lowrank_gpt2: run1, run2, run3

2. **V=5, num_examples=512**
   - gpt2: run1, run2, run3
   - lowrank_gpt2: run1, run2, run3

3. ... (依此类推)

10. **V=20, num_examples=4096**
    - gpt2: run1, run2, run3
    - lowrank_gpt2: run1, run2, run3

## 配置详情

### 标准 GPT2 模型配置

基于 `src/conf/table_connectivity.yaml`:
- family: gpt2
- n_embd: 256
- n_layer: 12
- n_head: 8
- 默认位置编码（每个位置独立）

### Low-Rank GPT2 模型配置

基于 `src/conf/table_connectivity_lowrank.yaml`:
- family: lowrank_gpt2
- n_embd: 256
- n_layer: 12
- n_head: 8
- **低秩位置编码**：共享C+1长度的模式，最后3个独立
- **自定义attention masks**：
  - 前2层：block diagonal pattern
  - 剩余层：sparse connectivity
- n_positions动态调整：V*(C+1)+3

## 自定义实验参数

如需修改实验参数，编辑 `run_experiments.py` 中的以下行：

```python
# 实验参数
V_VALUES = [5, 20]                          # 修改V值
NUM_EXAMPLES = [256, 512, 1024, 2048, 4096] # 修改训练样本数
MODEL_TYPES = ["gpt2", "lowrank_gpt2"]      # 修改模型类型
NUM_RUNS = 3                                 # 修改每个配置的重复次数
```

## 监控实验进度

### 实时查看日志

```bash
tail -f experiments.log
```

### 查看完成的实验数量

```bash
grep "Successfully completed" experiments.log | wc -l
```

### 查看失败的实验

```bash
grep "Failed on run" experiments.log
```

## 预期运行时间

假设每个实验运行时间约为 X 分钟：
- 单个实验：~X 分钟
- 总时间：~60X 分钟

（实际时间取决于硬件配置和训练步数）

## 结果分析

实验完成后，可以使用 `src/eval.py` 或 Jupyter notebook 分析结果：

```python
import pandas as pd
import matplotlib.pyplot as plt

# 加载结果并比较不同配置的性能
# ...
```

## 注意事项

1. **磁盘空间**：确保有足够的磁盘空间存储60个模型检查点
2. **GPU内存**：两种模型对GPU内存的需求可能不同
3. **临时文件**：脚本会自动清理临时配置文件
4. **失败处理**：如果某个实验失败，脚本会继续运行后续实验
5. **中断恢复**：目前不支持断点续传，如需此功能请修改脚本

## 常见问题

**Q: 如何只运行特定V值的实验？**
```python
V_VALUES = [20]  # 只运行V=20的实验
```

**Q: 如何只测试一种模型？**
```python
MODEL_TYPES = ["lowrank_gpt2"]  # 只测试低秩模型
```

**Q: 如何增加每个配置的运行次数？**
```python
NUM_RUNS = 5  # 每个配置运行5次
```

**Q: 实验失败怎么办？**

查看日志文件了解失败原因，然后可以手动运行失败的配置：
```bash
python src/train.py --config src/conf/table_connectivity.yaml
```

