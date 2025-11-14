# MatrixChainTransformer 快速入门指南

## 🚀 5分钟快速开始

### 1. 运行测试验证安装

```bash
# 简单测试（约1-2分钟）
python test_matrix_chain_custom_simple.py
```

预期输出：
- ✅ 模型创建成功（~1.1M参数）
- ✅ 500步训练完成
- ✅ 显示测试集MSE

### 2. 运行详细示例

```bash
# 详细示例（约1分钟）
python example_matrix_chain_custom_transformer.py
```

预期输出：
- ✅ 架构概览
- ✅ 50个epoch训练
- ✅ 详细的预测结果和误差分析

### 3. 完整训练

```bash
# 使用配置文件进行完整训练（需要较长时间）
cd src
python train.py --config conf/matrix_chain_custom.yaml
```

## 📋 架构速览

```
输入: [M_1, M_2, M_3]  (每个M_i是12×12的块对角矩阵)
  ↓
[Stage 1] 双路Transformer
  - Transformer 1: [M_1, M_2, M_3]
  - Transformer 2: [M_1^T, M_2^T, M_3^T]
  ↓
[Stage 2] 融合Transformer
  - Transformer 3: [h1, h2]
  - Transformer 4: [h3^T]
  ↓
[Stage 3] MLP预测
  - 输入: h_final[L-1]  (只用最后一个M_3的表征)
  - 输出: [Y_pred, Z_pred]  (各4×4)
```

## 🎯 关键特性

1. **两步训练**
   - 第1步：mask Y，预测 Y
   - 第2步：用真实Y，预测 Z

2. **双路处理**
   - 同时处理原始和转置矩阵
   - 捕获矩阵的对称性

3. **聚焦最后块**
   - 只对最后的 M_L 计算损失
   - 更符合实际应用

## 📊 快速性能检查

运行后检查这些指标：

```python
# 训练500步后，期望看到：
Y MSE: 3-5      # Y预测均方误差
Z MSE: 14-20    # Z预测均方误差
Total: 8-12     # 总体MSE

# 注：随机初始化会有波动，更多训练步数会提升性能
```

## 🔧 配置参数

主要参数（`src/conf/matrix_chain_custom.yaml`）：

```yaml
model:
    family: matrix_chain_transformer  # 必须
    L: 3           # M_i块的数量
    n: 4           # 矩阵大小
    n_dims: 12     # 输入维度 (= 3*n)
    n_embd: 128    # embedding维度
    n_head: 4      # 注意力头数

training:
    batch_size: 64
    learning_rate: 0.0003
    train_steps: 10000
```

## 📝 代码使用示例

```python
import sys
sys.path.append('src')

from models import MatrixChainTransformer
from samplers import MatrixChainSampler
from tasks import MatrixChain

# 1. 创建模型
model = MatrixChainTransformer(
    n_dims=12,
    n_embd=128,
    n_head=4,
    L=3,
    n=4
)

# 2. 准备数据
sampler = MatrixChainSampler(n_dims=12, L=3, n=4, m=4)
xs = sampler.sample_xs(n_points=36, b_size=16)

task = MatrixChain(
    n_dims=12,
    batch_size=16,
    seeds=None,
    L=3, n=4, m=4, p=4, q=4
)
xs_assembled, ys = task.evaluate(xs)

# 3. 前向传播
output = model(xs_assembled, ys)

# 4. 提取预测（最后一个M_i的Y和Z）
last_block_start = 2 * 12  # (L-1) * 3 * n
y_pred = output[:, last_block_start+4:last_block_start+8, 4:8]
z_pred = output[:, last_block_start+8:last_block_start+12, 8:12]
```

## 🐛 常见问题

### Q1: ModuleNotFoundError: No module named 'quinine'
**A**: 使用简化测试脚本，不需要quinine：
```bash
python test_matrix_chain_custom_simple.py
```

### Q2: 训练损失不下降
**A**: 尝试：
- 增加训练步数（10000+）
- 调整学习率（0.0001-0.001）
- 增加模型大小（n_embd=256）

### Q3: 如何修改L和n？
**A**: 在配置文件中修改，确保 `n_dims = 3 * n`：
```yaml
model:
    L: 5           # 改为5个块
    n: 6           # 改为6×6矩阵
    n_dims: 18     # = 3 * 6
```

### Q4: 如何可视化注意力？
**A**: 当前版本未实现，但可以通过以下方式添加：
```python
# 在模型forward中保存注意力权重
# 使用 matplotlib 绘制热图
```

## 📚 进一步阅读

- 详细文档：`MATRIX_CHAIN_CUSTOM_MODEL.md`
- 实现总结：`MATRIX_CHAIN_CUSTOM_IMPLEMENTATION.md`
- 完整代码：`src/models.py` (line 656+)

## ✅ 验证清单

运行以下命令确保一切正常：

```bash
# 1. 检查文件存在
ls -l src/models.py src/train.py src/conf/matrix_chain_custom.yaml

# 2. 检查模型类
grep "class MatrixChainTransformer" src/models.py

# 3. 检查训练逻辑
grep "is_custom_matrix_chain" src/train.py

# 4. 运行测试
python test_matrix_chain_custom_simple.py
```

所有检查通过 = 安装成功！🎉

## 🚦 下一步

1. **调优参数**：尝试不同的L, n, n_embd
2. **增加训练**：运行更多步数看性能提升
3. **对比实验**：与标准GPT2在matrix_chain任务上对比
4. **扩展架构**：添加更多层或注意力机制

---

**需要帮助？** 查看详细文档或检查示例代码。

**发现bug？** 检查 `test_matrix_chain_custom_simple.py` 的输出。

**想要定制？** 修改 `src/models.py` 中的 `MatrixChainTransformer` 类。

