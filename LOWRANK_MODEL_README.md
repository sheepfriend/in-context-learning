# Low-Rank Transformer Model

这个文档说明了新实现的 `LowRankTransformerModel` 类的功能和使用方法。

## 概述

`LowRankTransformerModel` 是基于 `TransformerModel` 的改进版本，具有以下两个主要特性：

1. **Low-Rank Positional Embeddings（低秩位置编码）**
2. **Layer-specific Custom Attention Masks（分层自定义注意力掩码）**

## 1. Low-Rank Positional Embeddings

### 设计原理

标准的positional embedding为每个位置学习独立的embedding向量。在序列长度为 `V*(C+1)+3` 的情况下，这需要大量参数。

Low-rank设计利用了序列的结构化特性：
- **前 V*(C+1) 个位置**：这些位置可以看作V个重复的块，每个块长度为C+1
  - 我们只学习**一个**长度为C+1的共享embedding
  - 将这个embedding重复V次，覆盖前V*(C+1)个位置
- **最后 3 个位置**：学习3个独立的embedding

### 参数减少

以配置 V=20, C=3, n_embd=64 为例：
- **标准方案**：83 × 64 = 5,312 个参数（每个位置独立）
- **Low-rank方案**：(4 × 64) + (3 × 64) = 448 个参数
- **减少**：4,864 个参数（减少约91.6%）

对于更大的模型（如n_embd=256）：
- **标准方案**：83 × 256 = 21,248 个参数
- **Low-rank方案**：(4 × 256) + (3 × 256) = 1,792 个参数
- **减少**：19,456 个参数（减少约91.6%）

### 实现

```python
# Low-rank positional embeddings
self.pos_emb_shared = nn.Parameter(torch.randn(C + 1, n_embd) * 0.02)  # 共享的C+1长度embedding
self.pos_emb_last3 = nn.Parameter(torch.randn(3, n_embd) * 0.02)      # 最后3个独立embedding

# 生成完整的positional embeddings
pos_emb_repeated = self.pos_emb_shared.repeat(V, 1)  # 重复V次
pos_emb = torch.cat([pos_emb_repeated, self.pos_emb_last3], dim=0)  # 拼接
```

## 2. Custom Attention Masks

### 设计原理

注意力掩码在不同层有不同的结构，以实现特定的信息流动模式。

#### 前2层：Block Diagonal Pattern

**目的**：让每个C+1长度的块内部进行充分的局部信息交换。

**规则**：
- V*(C+1)个位置被分成V个块，每块长度为C+1
- 每个块内的tokens可以互相attend
- 不同块之间**不能**互相attend
- 最后3个位置只能attend自己

**示例**（V=20, C=3）：
- Block 0: 位置 [0,1,2,3] 互相attend
- Block 1: 位置 [4,5,6,7] 互相attend
- ...
- Block 19: 位置 [76,77,78,79] 互相attend
- 位置 80 只能attend自己
- 位置 81 只能attend自己
- 位置 82 只能attend自己

**稀疏度**：只有 4.70% 的连接被激活（323/6889）

#### 剩余层（第3层及之后）：Sparse Connectivity

**目的**：只允许关键信息位置之间的全局交互。

**规则**：
- 每个C+1块的**最后一个token**（即位置3, 7, 11, ..., 79）
- 加上最后3个位置（80, 81, 82）
- 这23个位置可以**互相attend**
- 其他所有位置都不能attend任何位置

**关键位置**（V=20, C=3）：
```
[3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75, 79, 80, 81, 82]
```

**稀疏度**：只有 7.68% 的连接被激活（529/6889 = 23²）

### 实现细节

```python
# 前2层的mask：block diagonal
mask_first_layers = torch.zeros(seq_len, seq_len)
for i in range(V):
    start = i * (C + 1)
    end = start + (C + 1)
    mask_first_layers[start:end, start:end] = 1
mask_first_layers[-3, -3] = 1  # 最后3个位置
mask_first_layers[-2, -2] = 1
mask_first_layers[-1, -1] = 1

# 剩余层的mask：sparse connectivity
mask_remaining_layers = torch.zeros(seq_len, seq_len)
last_token_indices = [(i + 1) * (C + 1) - 1 for i in range(V)]
last_token_indices.extend([seq_len - 3, seq_len - 2, seq_len - 1])
for i in last_token_indices:
    for j in last_token_indices:
        mask_remaining_layers[i, j] = 1
```

## 使用方法

### 1. 在Python代码中直接使用

```python
from models import LowRankTransformerModel

model = LowRankTransformerModel(
    n_dims=5,           # 输入维度
    n_positions=83,     # 序列长度（必须等于V*(C+1)+3）
    n_embd=256,         # embedding维度
    n_layer=12,         # transformer层数
    n_head=8,           # 注意力头数
    V=20,               # 块的数量
    C=3                 # 每块的大小（实际为C+1）
)

# Forward pass
xs = torch.randn(batch_size, n_points, n_dims)
ys = torch.randn(batch_size, n_points)
output = model(xs, ys)
```

### 2. 使用配置文件

创建配置文件（如 `src/conf/my_lowrank_config.yaml`）：

```yaml
inherit: 
    - base.yaml

model:
    family: lowrank_gpt2  # 使用低秩模型
    n_positions: 83       # V*(C+1)+3
    n_dims: 5
    n_embd: 256
    n_layer: 12
    n_head: 8
    V: 20                 # 必须在model配置中指定
    C: 3                  # 必须在model配置中指定

training:
    task: table_connectivity
    task_kwargs: {"V": 20, "C": 3, "rho": 0.5}
    # ... 其他训练配置
```

然后运行：
```bash
python src/train.py --config src/conf/my_lowrank_config.yaml
```

### 3. 示例配置

项目中已包含示例配置文件：
- `src/conf/table_connectivity_lowrank.yaml` - 用于table connectivity任务的低秩模型配置

## 测试和验证

### 运行单元测试

```bash
python test_lowrank_model.py
```

测试包括：
- ✓ Positional embeddings的低秩结构验证
- ✓ Attention masks的正确性验证
- ✓ Forward pass功能测试
- ✓ 参数数量对比

### 可视化attention masks

```bash
python visualize_masks.py
```

这将生成 `attention_masks_visualization.png`，展示两种attention pattern的可视化。

## 设计考虑

### 为什么使用Low-Rank Positional Embeddings？

1. **参数效率**：减少约92.7%的positional embedding参数
2. **结构化归纳偏置**：明确编码序列的重复块结构
3. **泛化能力**：共享的pattern可能帮助模型更好地识别相似的局部结构

### 为什么使用Layer-specific Attention Masks？

1. **层次化信息处理**：
   - 前2层：局部特征提取（块内信息融合）
   - 后续层：全局信息传递（关键位置间通信）

2. **计算效率**：
   - 前2层：只有4.79%的attention连接
   - 后续层：只有7.20%的attention连接
   - 大幅减少计算量

3. **可解释性**：明确的信息流动路径

## 与标准Transformer的对比

| 特性 | TransformerModel | LowRankTransformerModel |
|------|------------------|-------------------------|
| Positional Embeddings | 每个位置独立 (83 × n_embd) | 低秩结构 ((C+1+3) × n_embd) |
| Attention Pattern | 全注意力或因果掩码 | 分层自定义稀疏掩码 |
| 位置参数数量 (n_embd=256) | 21,248 | 1,792 (↓91.6%) |
| 前2层Attention稀疏度 | ~100% | 4.70% |
| 后续层Attention稀疏度 | ~100% | 7.68% |

## 限制和注意事项

1. **序列长度固定**：序列长度必须精确等于 V*(C+1)+3
2. **超参数依赖**：V和C必须在初始化时正确指定
3. **任务特定性**：这个设计是为table connectivity类型的任务优化的

## 未来改进方向

1. **动态序列长度**：支持可变的V和C
2. **可学习的掩码**：让模型学习最优的attention pattern
3. **更灵活的低秩结构**：支持更复杂的位置编码共享模式

## 参考

- 原始TransformerModel实现：`src/models.py:82-130`
- Low-Rank实现：`src/models.py:132-290`
- 测试脚本：`test_lowrank_model.py`
- 可视化脚本：`visualize_masks.py`

