# Autoregressive Path Search for Table Connectivity

## 概述

这是一个全新的任务和架构，将表连接问题转化为自回归的路径生成任务。

## 任务定义

### 输入 (x)
- 图结构编码（与原始table connectivity相同）
- 两个query columns

### 输出 (y)
- **不再是简单的0/1标签**
- 而是完整的搜索路径序列
- 使用BFS生成所有可能的路径

### 数据生成策略

#### Connected情况（label = 1）
- 使用BFS找到所有从col1到col2的valid paths
- 每条valid path作为一个独立的训练样本
- 所有样本的最终label = 1

#### Not Connected情况（label = -1）
- 生成BFS搜索过程中的部分路径（exploration paths）
- 这些路径都没有到达目标
- 从exploration中随机采样一些作为负样本
- 所有负样本的label = -1

## 模型架构

### 自回归Transformer

#### 输入格式
```
序列结构：
[schema encoding] [separator] [start_col] [step1] [step2] ... [end_col] 
```

#### Attention Mask
- **前2层**：和原来一样（block diagonal）
- **第3层开始**：Causal mask（每个token只能看到之前的tokens）

#### Positional Encoding
- **前2层**：Low-rank positional encoding
- **第3层开始**：Standard positional encoding（与所有位置相关）

### 输出层
- **Classification head**：预测下一个token
- 因为可能的columns和tables是固定的，所以输出维度 = V*C + special tokens
- Loss：Cross-entropy for next token prediction

## 训练

### Token Vocabulary
```
0: [PAD]
1: [START] - 开始token
2: [SEP] - 分隔符
3: [END] - 结束token
4-: Column IDs (0 to V*C-1)
```

### 训练过程
1. Teacher forcing：给定前缀，预测下一个column
2. Loss：对整个序列计算next token prediction loss
3. 最后一个token预测最终label（1或-1）

## 推理（Beam Search）

### Beam Search参数
- `beam_width`: 保持top-k个候选路径
- `max_length`: 最大路径长度
- `length_penalty`: 长度惩罚参数

### 算法
```python
1. 初始化：beam = [[start_col]]
2. For each step:
   a. 对beam中每个序列，预测下一个token的概率
   b. 扩展所有可能的下一步
   c. 根据累积log概率排序
   d. 保留top-k个候选
3. 返回概率最高的完整路径
```

## 文件结构

### 新文件
- `src/samplers_autoregressive.py` - 新的数据sampler
- `src/tasks_autoregressive.py` - 新的task定义
- `src/models_autoregressive.py` - 自回归transformer
- `src/beam_search.py` - Beam search实现
- `src/conf/table_connectivity_autoregressive.yaml` - 配置文件

## 使用方法

### 训练
```bash
cd src
python train.py --config conf/table_connectivity_autoregressive.yaml
```

### 测试（使用Beam Search）
```python
from beam_search import beam_search_inference

predictions = beam_search_inference(
    model=model,
    xs=test_xs,
    beam_width=5,
    max_length=10
)
```

## 关键差异

| 特性 | 原始任务 | 新任务 |
|------|---------|--------|
| 输出 | 单个label (0/1) | 完整路径序列 |
| 模型 | Encoder-only | Autoregressive (Decoder) |
| Loss | Binary classification | Next token prediction |
| 推理 | Forward pass | Beam search |
| 数据量 | 每个query 1个样本 | 每个valid path 1个样本 |

## 优势

1. **更细粒度的监督**：不仅知道是否connected，还知道如何connected
2. **可解释性**：可以看到模型的搜索过程
3. **多样性**：同一个query可能有多条valid paths
4. **更符合实际推理过程**：BFS是实际的图搜索算法

