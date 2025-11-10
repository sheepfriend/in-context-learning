# Autoregressive模型集成到实验流程 - 更新说明

## ✅ 完成的更新

### 1. **Wandb已禁用**
在 `src/train_autoregressive.py` 中添加了 `mode="disabled"` 参数，wandb不会上传数据。

```python
wandb.init(
    project=args.wandb.project,
    entity=args.wandb.entity,
    name=args.wandb.name,
    notes=args.wandb.notes,
    config=vars(args),
    mode="disabled"  # ✓ Wandb disabled
)
```

### 2. **Autoregressive添加到run_experiments.py**

#### 更新的参数
```python
MODEL_TYPES = [
    "gpt2", 
    "lowrank_gpt2", 
    "gpt2_fixed", 
    "lowrank_gpt2_fixed", 
    "autoregressive_gpt2"  # ✓ 新增
]

AUTOREGRESSIVE_CONFIG = "conf/table_connectivity_autoregressive.yaml"  # ✓ 新增
```

#### 自动配置更新
对于autoregressive模型，脚本会自动更新：
- `model.V` - 表数量
- `model.C` - 列数（固定为3）
- `model.vocab_size` - 词汇表大小 (4 + V*C)
- `model.schema_len` - Schema长度 (V*4+1)
- `training.task_kwargs` - 包含V, C, vocab_size

#### 使用不同的训练脚本
```python
if is_autoregressive:
    train_script = "train_autoregressive.py"  # ✓ Autoregressive
else:
    train_script = "train.py"  # Standard/Low-rank
```

### 3. **parse_logs.py已更新**

#### 支持新的文件名格式
```python
# 新格式支持autoregressive
pattern = r"table_connectivity_(standard|lowrank|autoregressive)_(fixed|random|auto)_V(\d+)_N(\d+)_run(\d+)_gpu(\d+)"
```

#### 支持新的测试输出格式
Autoregressive模型输出：
- `Label Accuracy` - 最终label准确率
- `Exact Match Rate` - 完整路径匹配率

Standard模型输出（保持不变）：
- `Acc` - 准确率
- `P(y=1)` - 正样本比例
- `P(hat_y=1)` - 预测正样本比例

#### 更新的对比表
包含所有5种模型：
1. Standard
2. Standard-Fixed
3. Low-Rank
4. Low-Rank-Fixed
5. **Autoregressive** ✓ 新增

## 📊 实验配置

### 当前参数（run_experiments.py）
```python
V_VALUES = [3]
NUM_EXAMPLES = [2**i for i in range(12,16)]  # [4096, 8192, 16384, 32768]
MODEL_TYPES = ["gpt2", "lowrank_gpt2", "gpt2_fixed", "lowrank_gpt2_fixed", "autoregressive_gpt2"]
NUM_RUNS = 5
NUM_GPUS = 4
```

**总实验数**: 1 (V) × 4 (num_examples) × 5 (model_types) × 5 (runs) = **100 个实验**

### Autoregressive配置（已修改）
```yaml
model:
    n_positions: 200  # 增加到200以支持更长序列
    n_embd: 64        # 从256降到64
    n_layer: 8        # 从12降到8
    
training:
    train_steps: 5001 # 从2001增加到5001
```

## 🚀 使用方法

### 运行所有实验（包括autoregressive）
```bash
cd /Users/yuexing/Dropbox/in-context-learning
python run_experiments.py
```

这将运行5种模型的100个实验：
- 20个 gpt2 实验
- 20个 lowrank_gpt2 实验
- 20个 gpt2_fixed 实验
- 20个 lowrank_gpt2_fixed 实验
- **20个 autoregressive_gpt2 实验** ✓

### 解析结果（包括autoregressive）
```bash
python parse_logs.py --logs_dir logs --output final_results
```

输出将包含autoregressive模型的结果。

## 📝 日志文件命名

### Autoregressive日志格式
```
table_connectivity_autoregressive_auto_V3_N4096_run1_gpu0.log
table_connectivity_autoregressive_auto_V3_N8192_run2_gpu1.log
...
```

### 解析的字段
- `model_type`: `autoregressive_gpt2`
- `model_name`: `Autoregressive`
- `sampler_type`: `auto`
- `test_acc`: Label accuracy
- `exact_match`: Exact match rate（仅autoregressive）

## 🔍 与其他模型的对比

| 特性 | Standard/Low-Rank | Autoregressive |
|------|------------------|----------------|
| 训练脚本 | `train.py` | `train_autoregressive.py` |
| 输出 | Binary label | Path sequence |
| 测试指标 | Acc, P(y=1), P(ŷ=1) | Label Acc, Exact Match |
| 推理 | Forward pass | Beam search |
| Sampler tag | `random`/`fixed` | `auto` |

## 📈 预期输出

### 对比表格式（示例）
```
V | N    | Standard  | Standard-Fixed | Low-Rank  | Low-Rank-Fixed | Autoregressive
--|------|-----------|----------------|-----------|----------------|---------------
3 | 4096 | 0.75±0.02 | 0.78±0.01     | 0.73±0.03 | 0.76±0.02     | 0.80±0.01
3 | 8192 | 0.82±0.01 | 0.85±0.01     | 0.80±0.02 | 0.83±0.01     | 0.87±0.01
```

## ✅ 验证清单

- [x] Wandb disabled
- [x] Autoregressive添加到MODEL_TYPES
- [x] 配置文件路径添加
- [x] run_experiment函数更新
- [x] 自动配置V, C, vocab_size, schema_len
- [x] 使用train_autoregressive.py运行
- [x] parse_logs.py支持新文件名格式
- [x] parse_logs.py支持新测试输出格式
- [x] 对比表包含Autoregressive

## 🎯 下一步

1. **运行实验**
   ```bash
   python run_experiments.py
   ```

2. **等待完成**（约20-30小时，100个实验，4个GPU并行）

3. **解析结果**
   ```bash
   python parse_logs.py --logs_dir logs --output final_results
   ```

4. **分析对比**
   - 查看 `final_results.xlsx`
   - 比较5种模型的性能
   - 特别关注Autoregressive的exact match率

---

**更新日期**: 2025-11-10
**状态**: ✅ Ready to run

