# MatrixChainTransformer - Custom Model for Matrix Chain Task

## 概述

`MatrixChainTransformer` 是专门为 Matrix Chain 任务设计的自定义 Transformer 架构。

## 架构设计

### 输入格式
- 输入: `[M_1, M_2, ..., M_L]`，其中每个 `M_i` 是块对角矩阵：
  ```
  M_i = [X_i,  0,   0  ]
        [ 0,  Y_i,  0  ]
        [ 0,   0,  Z_i ]
  ```
- 每个块的大小为 `3n × 3n`（其中 `n` 是子矩阵的大小）

### 架构流程

1. **Stage 1: 双路Transformer**
   - Transformer 1: 处理原始序列 `[M_1, ..., M_L]`
   - Transformer 2: 处理转置序列 `[M_1^T, ..., M_L^T]`
   - 两个Transformer各有一层，参数不共享

2. **Stage 2: 融合处理**
   - 将两路输出拼接: `[output_1, output_2]`
   - 通过两个Transformer（无位置编码）继续处理
   - Transformer 3: 处理拼接后的序列
   - Transformer 4: 处理转置后的序列
   - 两个Transformer各有一层，参数不共享

3. **Stage 3: 预测头**
   - MLP层降维
   - 输出最后一个 `M_i` 的 `Y` 和 `Z` 预测

### 训练方式

与标准的next token prediction不同，这个模型采用特殊的训练流程：

1. **第一步：预测Y**
   - 将最后一个 `M_L` 的 `Y` 和 `Z` 部分都mask（置零）
   - **注意**：Z也要mask，因为Z是从Y计算得出的（Z=YB），在Y未知时Z也应未知
   - 模型预测 `Y`
   - 计算 `Y` 的MSE loss

2. **第二步：预测Z**
   - 使用真实的 `Y` 值
   - 将最后一个 `M_L` 的 `Z` 部分mask（置零）
   - 模型预测 `Z`
   - 计算 `Z` 的MSE loss

3. **总Loss**
   - `Loss = (Y_loss + Z_loss) / 2`
   - 只在最后一个 `M_i` 的 `Y` 和 `Z` 上计算loss

## 使用方法

### 1. 配置文件

创建配置文件 `src/conf/matrix_chain_custom.yaml`：

```yaml
inherit: 
    - wandb.yaml

model:
    family: matrix_chain_transformer
    n_dims: 12  # 3*n (for n=4)
    n_embd: 128
    n_head: 4
    L: 3  # Number of M_i blocks
    n: 4  # Matrix size

training:
    task: matrix_chain
    data: matrix_chain
    task_kwargs: {"L": 3, "n": 4, "m": 4, "p": 4, "q": 4}
    batch_size: 64
    learning_rate: 0.0003
    train_steps: 10000
    curriculum:
        dims:
            start: 12
            end: 12
            inc: 0
            interval: 2000
        points:
            start: 36
            end: 36
            inc: 0
            interval: 2000

out_dir: ../models/matrix_chain_custom

wandb:
    name: "matrix_chain_custom_transformer"
```

### 2. 训练模型

```bash
cd src
python train.py --config conf/matrix_chain_custom.yaml
```

### 3. 代码中使用

```python
from models import MatrixChainTransformer

# 创建模型
model = MatrixChainTransformer(
    n_dims=12,      # 3*n
    n_embd=128,     # embedding dimension
    n_head=4,       # number of attention heads
    L=3,            # number of M_i blocks
    n=4             # matrix size
)

# 前向传播
output = model(xs, ys)

# 提取预测
last_block_start = (L - 1) * 3 * n
y_start = last_block_start + n
y_end = last_block_start + 2 * n
z_start = last_block_start + 2 * n
z_end = last_block_start + 3 * n

y_pred = output[:, y_start:y_end, n:2*n]
z_pred = output[:, z_start:z_end, 2*n:3*n]
```

## 模型参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `n_dims` | 输入维度（应为3*n） | 必须指定 |
| `n_embd` | Embedding维度 | 128 |
| `n_head` | 注意力头数 | 4 |
| `L` | M_i块的数量 | 3 |
| `n` | 子矩阵大小 | 4 |

## 关键特性

1. **专门设计**: 针对Matrix Chain任务的特定结构设计
2. **对称处理**: 同时处理原始和转置序列，捕获矩阵的对称性
3. **无位置编码**: Stage 2的Transformer不使用位置编码，让模型更关注内容而非位置
4. **两步预测**: 先预测Y，再基于真实Y预测Z，符合任务的因果结构
5. **聚焦最后块**: 只对最后一个M_i计算loss，更符合实际应用场景

## 测试

运行测试脚本验证模型：

```bash
python test_matrix_chain_custom_model.py
python test_train_matrix_chain_custom.py
```

## 与标准GPT2模型的对比

| 特性 | MatrixChainTransformer | 标准GPT2 |
|------|------------------------|----------|
| 架构 | 自定义多阶段 | 标准Transformer |
| 位置编码 | 部分使用 | 全局使用 |
| 训练方式 | 两步mask预测 | Next token prediction |
| Loss计算 | 只在最后M_i | 在所有位置 |
| 适用场景 | Matrix Chain专用 | 通用 |

## 注意事项

1. 确保 `n_dims = 3 * n`
2. 确保输入shape为 `(batch_size, L*3*n, 3*n)`
3. Task应使用 `matrix_chain`，Data应使用 `matrix_chain`
4. 模型会自动检测并使用特殊的训练流程（在`train.py`中）

