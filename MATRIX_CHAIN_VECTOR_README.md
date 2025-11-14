# MatrixChainVector Task

## 概述

`MatrixChainVector` 是一个新的in-context learning任务，与 `MatrixChain` 类似，但使用标准GPT2 Transformer，并将X矩阵reshape为列向量格式。

## 与 MatrixChain 的区别

| 特性 | MatrixChain | MatrixChainVector |
|------|-------------|-------------------|
| **模型** | 自定义MatrixChainTransformer | 标准GPT2 |
| **X格式** | 3n×3n块对角矩阵 | n²×1 列向量 |
| **数据结构** | [X,0,0; 0,Y,0; 0,0,Z] | [x,0,0; 0,Y,0; 0,0,Z] |
| **转换** | Y=AX, Z=YB | **Y=XA, Z=YB** |
| **序列长度** | L×3n | L×(n²+2n) |
| **Token维度** | 3n | n |

## 数据格式

### 输入X采样
- 每个X是 n×n 矩阵
- 每行从 N(0, I_n) 采样
- 一个prompt有L个X矩阵

### 数据组装
对于每个 M_i (i=1,...,L):

```
[x,  0,  0]    ← n²行：x是X的列向量展开 (只第一列非零)
[0,  Y,  0]    ← n行：Y矩阵
[0,  0,  Z]    ← n行：Z矩阵
```

总共: L × (n² + 2n) 行，n 列

### 转换规则（固定）
- **Y = X @ A** (X右乘A)
- **Z = Y @ B** (Y右乘B)
- A和B对同一prompt的所有X共享
- 无seed时在evaluate()中动态生成A和B

## 维度示例

对于 L=3, n=4:
- 输入X: (batch, 3, 4, 4)
- 组装后: (batch, 72, 4)
  - 每个M_i: 24行 = 16(x向量) + 4(Y) + 4(Z)
  - 总行数: 3 × 24 = 72
- n_dims: 4
- n_positions: 72

## 配置

### 文件: `src/conf/matrix_chain_vector.yaml`

```yaml
model:
    family: gpt2
    n_dims: 4          # n
    n_positions: 66    # ~L*(n²+2n)
    n_embd: 128
    n_layer: 12
    n_head: 4

training:
    task: matrix_chain_vector
    data: matrix_chain_vector
    task_kwargs: {"L": 3, "n": 4, "m": 4, "p": 4, "q": 4}
    batch_size: 64
    learning_rate: 0.0003
    train_steps: 10000
```

## 使用方法

### 1. 训练

```bash
cd src
python train.py --config conf/matrix_chain_vector.yaml
```

### 2. 测试

```bash
# 完整测试
python test_matrix_chain_vector.py

# 快速测试
python -c "
import sys
sys.path.append('src')
from samplers import MatrixChainVectorSampler
from tasks import MatrixChainVector

sampler = MatrixChainVectorSampler(n_dims=4, L=3, n=4, m=4)
xs = sampler.sample_xs(n_points=72, b_size=2)
task = MatrixChainVector(n_dims=4, batch_size=2, L=3, n=4, m=4, p=4, q=4)
xs_assembled, ys = task.evaluate(xs)
print(f'Shape: {xs_assembled.shape}')  # (2, 72, 4)
"
```

### 3. 代码示例

```python
import sys
sys.path.append('src')
import torch
from samplers import MatrixChainVectorSampler
from tasks import MatrixChainVector
from models import build_model

# 参数
L, n = 3, 4
n_dims = n
n_positions = L * (n*n + 2*n)  # 72
b_size = 64

# 创建sampler和task
sampler = MatrixChainVectorSampler(n_dims=n_dims, L=L, n=n, m=n)
task = MatrixChainVector(n_dims=n_dims, batch_size=b_size, L=L, n=n, m=n, p=n, q=n)

# 采样数据
xs = sampler.sample_xs(n_points=n_positions, b_size=b_size)
xs_assembled, ys = task.evaluate(xs)

# 使用标准GPT2模型
class Config:
    family = 'gpt2'
    n_dims = n_dims
    n_positions = n_positions
    n_embd = 128
    n_layer = 12
    n_head = 4

model = build_model(Config())

# 训练
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
output = model(xs_assembled, ys)
loss = ((output - ys) ** 2).mean()
loss.backward()
optimizer.step()
```

## 关键特性

### ✅ 无Seed时动态生成A和B
```python
# 如果没有提供seeds或pool_dict
# A和B在evaluate()中每次动态生成
if self.A_b is None:
    A_b = torch.randn(b_size, n, n, device=xs_b.device)
    B_b = torch.randn(b_size, n, n, device=xs_b.device)
```

### ✅ 固定转换顺序
```python
# 固定: Y = X @ A, Z = Y @ B
Y = X @ A_b[i]
Z = Y @ B_b[i]
```

### ✅ 列向量展开
```python
# X按列展开 (Fortran order)
x_flat = X.T.reshape(-1)  # (n*n,)
xs_assembled[i, block_start:block_start+n*n, 0] = x_flat
```

## 验证

运行测试验证：

```bash
python test_matrix_chain_vector.py
```

预期输出：
```
✓ xs_assembled shape: (2, 72, 4)
✓ x向量在第一列: 16/16非零
✓ Y = XA: ||Y - XA|| = 0.000000
✓ Z = YB: ||Z - YB|| = 0.000000
```

## 实现文件

- **Sampler**: `src/samplers.py` - `MatrixChainVectorSampler`
- **Task**: `src/tasks.py` - `MatrixChainVector`
- **Schema**: `src/schema.py` - 添加到TASK_LIST
- **Config**: `src/conf/matrix_chain_vector.yaml`
- **Test**: `test_matrix_chain_vector.py`

## 参数说明

| 参数 | 说明 | 示例 |
|------|------|------|
| `L` | 每个prompt的矩阵数量 | 3 |
| `n` | 矩阵大小 (n×n) | 4 |
| `m, p, q` | 兼容性参数（必须=n） | 4 |
| `n_dims` | Token维度 | n |
| `n_positions` | 序列长度 | L×(n²+2n) |

## 训练目标

- **Loss**: MSE on full sequence
- **Metric**: mean_squared_error
- **预测**: Next token prediction (full embedding)

## 注意事项

1. ⚠️ n=m=p=q（必须相等）
2. ⚠️ 无seed时A、B每次evaluate动态生成
3. ⚠️ 转换顺序固定: Y=XA, Z=YB
4. ✅ 使用标准GPT2，无需自定义模型
5. ✅ X以列向量形式存储（节省空间）

## 性能

标准GPT2 (n_embd=128, n_layer=12, n_head=4):
- 参数量: ~13M
- 训练速度: ~200 steps/min (GPU)
- 建议train_steps: 10000+

## 下一步

1. 调整超参数 (n_embd, n_layer, learning_rate)
2. 尝试不同的L和n值
3. 比较与MatrixChain的性能差异
4. 添加curriculum learning

---

**创建日期**: 2025-11-14
**状态**: ✅ 已完成并测试

