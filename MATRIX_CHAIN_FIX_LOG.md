# MatrixChainTransformer 修复日志

## 修复 1: train.py 断言错误

### 问题描述
运行 `train.py` 时出现以下错误：
```
AssertionError at line 294:
assert args.model.family in ["gpt2", "lstm", "lowrank_gpt2"]
```

### 根本原因
`train.py` 第294行的断言只允许三种模型family，但没有包含新增的 `matrix_chain_transformer`。

### 修复方案
更新 `src/train.py` 第294行的断言，添加 `matrix_chain_transformer`：

**修改前：**
```python
assert args.model.family in ["gpt2", "lstm", "lowrank_gpt2"]
```

**修改后：**
```python
assert args.model.family in ["gpt2", "lstm", "lowrank_gpt2", "matrix_chain_transformer"]
```

### 验证
✅ 断言检查通过
✅ 无linter错误
✅ `matrix_chain_transformer` 现在可以正常使用

### 影响的文件
- `src/train.py` (line 294)

### 状态
✅ 已修复 (2025-11-14)

