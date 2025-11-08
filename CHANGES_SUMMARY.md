# 代码修改总结

## 完成的修改

### 1. 新增 LowRankTransformerModel 类
**文件**: `src/models.py`

创建了一个新的Transformer模型类，具有以下特性：

#### a) Low-Rank Positional Embeddings（低秩位置编码）
- 前 V*(C+1) 个位置共享一个长度为 C+1 的位置编码（重复V次）
- 最后 3 个位置使用独立的位置编码
- **参数减少**：从 83×n_embd 降低到 (C+1+3)×n_embd
  - 示例（n_embd=256）：21,248 → 1,792 参数（减少91.6%）

#### b) 自定义 Attention Masks
- **前2层**：Block diagonal pattern
  - V*(C+1) 中每 C+1 个token内部互相attend
  - 最后3个token只attend自己
  - 稀疏度：4.70%
  
- **剩余层**：Sparse connectivity
  - 每个 C+1 块的最后一个token
  - 加上最后3个token
  - 这些token之间全连接
  - 稀疏度：7.68%

### 2. 更新 build_model 函数
**文件**: `src/models.py`

添加了对 `lowrank_gpt2` 模型类型的支持，能够从配置文件中读取 V 和 C 参数。

```python
elif conf.family == "lowrank_gpt2":
    V = getattr(conf, 'V', 20)
    C = getattr(conf, 'C', 3)
    model = LowRankTransformerModel(...)
```

### 3. 更新训练脚本
**文件**: `src/train.py`

修改了模型类型检查，支持新的 `lowrank_gpt2` 模型：
```python
assert args.model.family in ["gpt2", "lstm", "lowrank_gpt2"]
```

### 4. 新增配置文件
**文件**: `src/conf/table_connectivity_lowrank.yaml`

创建了使用低秩模型的示例配置文件，包含：
- 模型配置（family: lowrank_gpt2, V: 20, C: 3）
- 训练配置（与原始table_connectivity相同）

### 5. 文档
**文件**: `LOWRANK_MODEL_README.md`

详细的使用文档，包含：
- 设计原理和实现细节
- 使用方法和示例
- 测试验证说明
- 与标准Transformer的对比

## 测试验证

所有功能已通过测试：
- ✅ Positional embeddings 的低秩结构正确
- ✅ Attention masks 的分层模式正确
- ✅ Forward pass 成功运行
- ✅ 参数数量减少符合预期

## 使用方式

### 直接使用
```python
from models import LowRankTransformerModel

model = LowRankTransformerModel(
    n_dims=5, n_positions=83, n_embd=256, 
    n_layer=12, n_head=8, V=20, C=3
)
```

### 通过配置文件
```bash
python src/train.py --config src/conf/table_connectivity_lowrank.yaml
```

## 关键设计决策

1. **Low-rank结构**：利用序列的重复块特性，大幅减少位置编码参数
2. **分层attention**：前层局部特征提取，后层全局信息传递
3. **稀疏连接**：只在关键位置间建立连接，提高计算效率

## 兼容性

- 保持了与原始 `TransformerModel` 相同的接口
- 不影响现有代码的运行
- 可以通过配置文件轻松切换模型类型

