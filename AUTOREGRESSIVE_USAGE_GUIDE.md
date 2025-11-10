# Autoregressive Path Search - 完整使用指南

## 📋 已实现的组件

### ✅ 核心组件

1. **`src/samplers_autoregressive.py`** - BFS路径生成sampler
   - 使用BFS找到所有valid paths（connected情况）
   - 采样exploration paths（not connected情况）
   - 固定embeddings确保一致性

2. **`src/models_autoregressive.py`** - 自回归Transformer
   - 前2层：Block diagonal attention（处理schema）
   - 剩余层：Causal attention（自回归生成）
   - 混合positional encoding

3. **`src/tasks_autoregressive.py`** - Next token prediction task
   - CrossEntropyLoss with padding ignore
   - Token-level accuracy metric

4. **`src/beam_search.py`** - Beam search推理
   - 可配置beam width和length penalty
   - 支持batch inference
   - 计算label accuracy和exact match

5. **`src/train_autoregressive.py`** - 训练脚本
   - 完整的训练循环
   - Wandb logging
   - Checkpoint saving
   - Beam search evaluation

6. **`src/conf/table_connectivity_autoregressive.yaml`** - 配置文件

7. **`test_autoregressive.py`** - 测试脚本（已验证通过✓）

### ✅ Schema更新

- 支持 `autoregressive_gpt2` model family
- 支持 `table_connectivity_autoregressive` task和data
- 添加 `vocab_size` 和 `schema_len` 参数

## 🚀 快速开始

### 1. 测试所有组件
```bash
cd /Users/yuexing/Dropbox/in-context-learning
python test_autoregressive.py
```

**预期输出**: 所有测试通过 ✓

### 2. 训练模型
```bash
cd /Users/yuexing/Dropbox/in-context-learning/src
python train_autoregressive.py --config conf/table_connectivity_autoregressive.yaml
```

### 3. 修改配置

编辑 `src/conf/table_connectivity_autoregressive.yaml`:

```yaml
model:
    V: 5              # 表的数量
    C: 3              # 每个表的列数
    n_embd: 256       # Embedding维度
    n_layer: 12       # Transformer层数
    n_head: 8         # Attention head数量

training:
    batch_size: 64
    learning_rate: 0.0001
    train_steps: 2001
    num_training_examples: 100000
```

## 📊 数据格式

### 输入 (x)
```
[table1] [col1] [col2] [col3] [table2] ... [SEP] [query_col1] [query_col2] ...
```

### 输出 (y)
```
Connected case (label=1):
  [col1] [col3] [col5] [col7] [END]  # Complete path

Not connected case (label=-1):
  [col1] [col3] [col4] [END]  # Partial exploration path
```

### 特殊Tokens
- `0`: PAD - 填充token
- `1`: START - 开始token
- `2`: SEP - 分隔符
- `3`: END - 结束token
- `4+`: Column IDs

## 🎯 模型架构详解

### Attention Pattern

#### First 2 Layers
```
Schema part:      [Block Diagonal]
Query part:       [Causal]
```

每个table的C+1个tokens形成一个block，只能互相attend。

#### Remaining Layers
```
All positions:    [Causal]
```

纯causal mask，每个位置只能看到之前的tokens。

### Positional Encoding

- **Schema part (0-20)**: 低秩positional embeddings
- **Path part (21+)**: 标准positional embeddings

## 🔬 Beam Search推理

### 基本用法

```python
from beam_search import beam_search_inference

predictions = beam_search_inference(
    model=model,
    xs_batch=test_xs,
    column_embeddings=sampler.column_embeddings,
    beam_width=5,
    max_length=15,
    device='cuda'
)

# 结果结构
for pred in predictions:
    tokens = pred['tokens']      # 预测的token序列
    score = pred['score']        # Log probability score
    all_beams = pred['all_beams']  # 所有beam candidates
```

### 评估

```python
from beam_search import evaluate_with_beam_search

accuracy, exact_match = evaluate_with_beam_search(
    model=model,
    xs_batch=xs,
    ys_batch=ys,
    labels_batch=labels,
    column_embeddings=column_embeddings,
    beam_width=5
)

print(f"Label Accuracy: {accuracy:.4f}")
print(f"Exact Match: {exact_match:.4f}")
```

## 📈 训练监控

### Wandb Metrics

训练过程中记录：
- `train/loss` - Next token prediction loss
- `train/accuracy` - Token-level accuracy
- `test/label_accuracy` - Final label正确率
- `test/exact_match` - 完整路径匹配率

### 查看训练进度
```bash
# 在wandb dashboard查看
# 或查看保存的checkpoints
ls ../models/table_connectivity_autoregressive/
```

## 🔧 高级定制

### 修改BFS策略

编辑 `src/samplers_autoregressive.py` 中的 `_bfs_find_all_paths`:

```python
def _bfs_find_all_paths(self, G, table_cols, start_col, end_col, max_length=10):
    # 修改max_length控制路径长度
    # 修改探索策略
    ...
```

### 修改Beam Search参数

```python
searcher = BeamSearcher(
    model=model,
    beam_width=10,        # 增加beam width
    max_length=20,        # 更长的生成
    length_penalty=0.6    # 调整length penalty
)
```

### 添加新的Attention Pattern

编辑 `src/models_autoregressive.py` 中的 `_register_attention_masks`:

```python
def _register_attention_masks(self):
    # 定制你自己的attention mask
    ...
```

## 📊 实验配置示例

### 小规模实验 (快速验证)
```yaml
model:
    V: 3
    C: 2
    n_embd: 128
    n_layer: 4
    n_head: 4

training:
    batch_size: 32
    train_steps: 500
```

### 中等规模实验
```yaml
model:
    V: 5
    C: 3
    n_embd: 256
    n_layer: 8
    n_head: 8

training:
    batch_size: 64
    train_steps: 2000
```

### 大规模实验
```yaml
model:
    V: 10
    C: 5
    n_embd: 512
    n_layer: 12
    n_head: 16

training:
    batch_size: 128
    train_steps: 5000
```

## 🐛 常见问题

### Q1: Out of Memory
**解决**: 减小 `batch_size` 或 `n_embd`

### Q2: 训练不收敛
**解决**: 
- 降低learning rate
- 增加train_steps
- 检查数据分布

### Q3: Beam search太慢
**解决**:
- 减小beam_width
- 减小max_length
- 使用greedy decoding (beam_width=1)

## 📝 对比：原始 vs 自回归

| 特性 | 原始Table Connectivity | 自回归Path Search |
|------|----------------------|-------------------|
| 输出 | Binary label (0/1) | Complete path |
| 模型 | Encoder-only | Autoregressive |
| Loss | Binary cross-entropy | Next token prediction |
| 推理 | Forward pass | Beam search |
| 可解释性 | 低 | 高（看到搜索路径） |
| 训练样本 | 1 per query | Multiple per query |
| 计算复杂度 | O(N) | O(N × B × L) |

## 🎓 理论背景

### BFS训练数据生成

1. **Connected cases**: 
   - 使用BFS找到所有从col1到col2的路径
   - 每条路径作为一个训练样本
   - Label = 1

2. **Not connected cases**:
   - BFS探索过程中的partial paths
   - 随机采样一部分作为负样本
   - Label = -1

### Length Normalization

使用Google NMT论文中的公式：
```
score = log_prob / ((5 + length) / 6) ^ alpha
```

避免beam search偏向短序列。

## 🚀 下一步

1. **实验不同的V和C值**
2. **尝试不同的attention patterns**
3. **对比不同的beam search策略**
4. **可视化生成的路径**
5. **分析模型学到的graph structure**

## 📚 相关文件

- 主要代码: `src/samplers_autoregressive.py`, `src/models_autoregressive.py`, `src/beam_search.py`
- 训练脚本: `src/train_autoregressive.py`
- 配置: `src/conf/table_connectivity_autoregressive.yaml`
- 测试: `test_autoregressive.py`
- 文档: `AUTOREGRESSIVE_PATH_SEARCH_README.md`, `AUTOREGRESSIVE_USAGE_GUIDE.md`

---

## ✅ 验证清单

- [x] Sampler实现并测试
- [x] Model实现并测试
- [x] Task实现并测试
- [x] Beam search实现并测试
- [x] Training script实现
- [x] Configuration文件创建
- [x] Schema更新
- [x] 集成测试通过
- [x] 文档完整

**状态**: 所有组件已实现并测试通过 ✓

开始训练吧！🚀

