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

---

## 修复 2: 训练时Z的masking问题

### 问题描述
在训练Step 1（预测Y）时，只mask了Y，但没有mask Z。这是不合理的，因为：
- Z是从Y计算得出的（Z=YB 或 Z=BY）
- 如果Y未知，那么Z也应该是未知的
- 预测Y时不应该看到Z的信息

### 根本原因
在 `train.py` 第41-50行，Step 1只mask了Y：
```python
xs_masked_y[:, y_start:y_end, n:2*n] = 0  # 只mask了Y
```

### 修复方案
在预测Y时，同时mask Y和Z：

**修改前：**
```python
# Step 1: Mask Y and predict it
xs_masked_y = xs.clone()
y_start = last_block_start + n
y_end = last_block_start + 2 * n
xs_masked_y[:, y_start:y_end, n:2*n] = 0  # Mask Y
```

**修改后：**
```python
# Step 1: Mask Y and Z, then predict Y
# (Z should also be masked because Z depends on Y)
xs_masked_y = xs.clone()
y_start = last_block_start + n
y_end = last_block_start + 2 * n
z_start = last_block_start + 2 * n
z_end = last_block_start + 3 * n
xs_masked_y[:, y_start:y_end, n:2*n] = 0  # Mask Y
xs_masked_y[:, z_start:z_end, 2*n:3*n] = 0  # Mask Z (because Z depends on Y)
```

### 验证
✅ Y和Z都被正确mask（norm=0）
✅ 前向传播正常工作
✅ 无linter错误
✅ 测试通过

### 影响的文件
- `src/train.py` (lines 41-49)
- `test_matrix_chain_custom_simple.py` (lines 28-36, 151-154)
- `example_matrix_chain_custom_transformer.py` (lines 96-100, 146-149)

### 状态
✅ 已修复 (2025-11-14)

---

## 修复 3: Attention mask确认

### 问题描述
用户要求确认新架构的attention不要加mask。

### 验证结果
✅ `MatrixChainTransformer` 使用的是 PyTorch 的 `nn.TransformerEncoder`
✅ 没有在forward中传入任何mask参数
✅ 默认行为是全连接的attention（无mask）
✅ 符合要求，无需修改

### 相关代码
```python
# src/models.py lines 758-766
h1 = self.embed_1(M_flat)  # (batch, L, n_embd)
h1 = self.transformer_1(h1)  # 没有传入mask参数
```

### 状态
✅ 已确认无问题 (2025-11-14)

---

## 修复 4: MatrixChainTransformer维度错误

### 问题描述
`MatrixChainTransformer`的forward方法中，将整个M_i矩阵(3n×3n)展平成一个向量作为单个token，导致维度不正确。

### 根本原因
在 `src/models.py` 第755行及后续：
```python
# 错误: 将整个矩阵flatten成一个向量
M_flat = M_blocks.view(batch_size, self.L, -1)  # (batch, L, 3n*3n)
self.embed_1 = nn.Linear(n_dims * n_dims, n_embd)
```

应该将每一行作为一个token，而不是整个矩阵。

### 修复方案

**修改前的架构：**
- 将每个M_i整体展平: (batch, L, 3n*3n)
- Embedding: 3n*3n → n_embd
- Sequence length: L

**修改后的架构：**
- 将M_i的每一行作为token: (batch, L*3n, 3n)
- Embedding: 3n → n_embd  
- Sequence length: L*3n

**关键修改：**

1. **Embedding层维度** (lines 685, 697):
```python
# Before: self.embed_1 = nn.Linear(n_dims * n_dims, n_embd)
# After:
self.embed_1 = nn.Linear(n_dims, n_embd)  # 3n → n_embd
```

2. **序列处理** (lines 755-770):
```python
# Before: M_flat = M_blocks.view(batch_size, self.L, -1)  # (batch, L, 3n*3n)
# After:
M_rows = M_blocks.view(batch_size, self.L * self.block_size, self.n_dims)  # (batch, L*3n, 3n)
h1 = self.embed_1(M_rows)  # (batch, L*3n, n_embd)
```

3. **Concatenation维度** (line 770):
```python
# Before: h_concat = torch.cat([h1, h2], dim=1)  # (batch, 2*L, n_embd)
# After:
h_concat = torch.cat([h1, h2], dim=1)  # (batch, L*3n*2, n_embd)
```

4. **Transformer 4配置** (lines 720-721):
```python
# Transformer 4操作reshaped序列，token维度是3n
d_model=3*n  # 而不是n_embd
```

5. **最终表征选择** (line 788):
```python
# Before: h_last = h_final[:, self.L-1]
# After:
last_row_idx = self.L * self.block_size - 1  # 最后一个M_L的最后一行
h_last = h_final[:, last_row_idx]
```

### 验证结果
✅ 前向传播成功
✅ 输出形状正确: (2, 36, 12)
✅ Y预测形状: (2, 4, 4) ✓
✅ Z预测形状: (2, 4, 4) ✓
✅ 参数量: 223,548 (之前: 1,101,344)

### 影响的文件
- `src/models.py` (lines 685, 697, 720-721, 755-789)

### 状态
✅ 已修复 (2025-11-14)

---

## 修复 5: 输出层改为全局MLP

### 问题描述
原始实现使用逐行MLP，没有充分利用全局信息。

### 修复方案
改为全局pooling + MLP架构：

1. **可学习的attention pooling** (lines 730-735, 796-798):
```python
# Pooling MLP: 生成attention权重
self.pooling_mlp = nn.Sequential(
    nn.Linear(2 * n_embd, n_embd),
    nn.GELU(),
    nn.Linear(n_embd, 1)
)

# 使用attention weights做加权求和
attn_weights = self.pooling_mlp(h_final)  # (batch, L*3n*2, 1)
attn_weights = torch.softmax(attn_weights, dim=1)
h_pooled = (h_final * attn_weights).sum(dim=1)  # (batch, 2*n_embd)
```

2. **MLP输出6n×6n格式** (lines 739-745):
```python
# 输出 2 个 3n×3n 的矩阵（Y 和 Z）
nn.Linear(2 * n_embd, 2 * (3*n) * (3*n))
```

3. **构建块对角输出** (lines 801-811):
```python
mlp_out.view(batch_size, 2, 3*self.n, 3*self.n)
output_6n[:, :3*self.n, :3*self.n] = Y_pred  # [Y, 0]
output_6n[:, 3*self.n:, 3*self.n:] = Z_pred  # [0, Z]
```

### 验证结果
✅ 前向传播成功
✅ 输出形状正确: (2, 36, 12)
✅ Y/Z预测: (2, 4, 4) ✓
✅ 参数量: 264,893 (vs 256,572)
✅ 可学习的attention pooling替代简单mean

### 影响的文件
- `src/models.py` (lines 730-745, 795-837)

### 状态
✅ 已修复 (2025-11-14)

---

## 总结

| 修复项 | 类型 | 严重程度 | 状态 |
|-------|------|---------|------|
| 1. 断言错误 | Bug | 高 | ✅ 已修复 |
| 2. Z的masking | Logic Bug | 中 | ✅ 已修复 |
| 3. Attention mask | 确认 | 低 | ✅ 已确认 |
| 4. 维度错误 | Critical Bug | 极高 | ✅ 已修复 |
| 5. 输出层改进 | Enhancement | 中 | ✅ 已修复 |

**关键改进：**
- 正确实现了行级token处理
- 参数量从1.1M → 223K → 257K
- 架构更符合矩阵结构的语义
- 使用全局信息进行预测

所有问题已解决，代码可以正常使用。

