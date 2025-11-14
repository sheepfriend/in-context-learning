# Matrix Chain Data Type

## 概述 (Overview)

这是一个新的in-context learning任务，实现了矩阵链变换：`Y = AX` 和 `Z = YB`。

This is a new in-context learning task that implements matrix chain transformations: `Y = AX` and `Z = YB`.

## 任务描述 (Task Description)

### 1. X Sampler (数据采样)
- 每个prompt包含L个矩阵X_i
- 每个X是n×m的矩阵，每行从N(0, I_m)采样
- 为简化实现，使用方阵：n = m

Each prompt contains L matrices X_i where each X is an n×m matrix with rows sampled from N(0, I_m). For simplicity, we use square matrices: n = m.

### 2. Task Sampler (任务采样)
- 对于同一个prompt的所有X矩阵，共享一对随机矩阵(A, B)
- A和B是n×n的随机矩阵
- 计算：Y = AX，Z = YB

For all X matrices in the same prompt, we share a pair of random matrices (A, B). Both A and B are n×n random matrices. We compute: Y = AX, Z = YB.

### 3. 组装 (Assembly)
- 对于每个i=1,...,L，构建块对角矩阵M_i：
```
M_i = [X_i  0    0  ]
      [0    Y_i  0  ]
      [0    0    Z_i]
```
- M_i的大小是(3n)×(3n)
- 将所有M_i沿序列维度拼接

For each i=1,...,L, construct block diagonal matrix M_i (size 3n×3n) and concatenate all M_i along the sequence dimension.

### 4. Training Loss (训练损失)
- 对Y和Z对应位置计算MSE损失
- X位置的目标设为0（不参与损失计算）
- Y和Z的目标是各行的均值

MSE loss on Y and Z positions. X positions have target 0 (not used in loss). Y and Z targets are the mean of each row.

## 文件结构 (File Structure)

### 新增文件 (New Files)
- `src/samplers.py`: 添加了 `MatrixChainSampler` 类
- `src/tasks.py`: 添加了 `MatrixChain` 类
- `src/conf/matrix_chain.yaml`: 配置文件
- `test_matrix_chain.py`: 测试脚本

### 修改文件 (Modified Files)
- `src/schema.py`: 添加了 "matrix_chain" 到任务和数据采样器列表

## 使用方法 (Usage)

### 1. 运行测试 (Run Test)
```bash
python test_matrix_chain.py
```

### 2. 训练模型 (Train Model)
```bash
cd src
python train.py --config conf/matrix_chain.yaml
```

### 3. 自定义配置 (Custom Configuration)

创建自己的配置文件，参考 `src/conf/matrix_chain.yaml`:

```yaml
inherit: 
    - base.yaml

model:
    n_dims: 12  # 3*n (for n=4)
    n_positions: 36  # L * 3 * n (for L=3, n=4)

training:
    task: matrix_chain
    data: matrix_chain
    task_kwargs: {"L": 3, "n": 4, "m": 4, "p": 4, "q": 4}
    batch_size: 64
    learning_rate: 3e-4
    train_steps: 10000
    curriculum:
        points:
            start: 12
            end: 36
            inc: 12
            interval: 2000

out_dir: ../models/matrix_chain

wandb:
    name: "matrix_chain_standard"
```

### 参数说明 (Parameters)

- `L`: 每个prompt中X矩阵的数量 (Number of X matrices per prompt)
- `n`: 矩阵大小（n×n）(Matrix size, must be equal for n, m, p, q)
- `m`: 应该等于n (Should equal n)
- `p`: 应该等于n (Should equal n)
- `q`: 应该等于n (Should equal n)
- `n_dims`: 模型维度，应该等于3*n (Model dimension, should be 3*n)
- `n_positions`: 最大序列长度，应该等于L*3*n (Max sequence length, should be L*3*n)

## 示例输出 (Example Output)

运行测试脚本后，你会看到：

```
================================================================================
Testing MatrixChain implementation
================================================================================

Parameters:
  L (number of matrices): 3
  n (matrix size): 4 x 4
  batch_size: 2
  n_dims: 12

1. Creating MatrixChainSampler...
2. Sampling 3 matrices per batch...
   xs shape: torch.Size([2, 3, 4, 4]) (expected: (2, 3, 4, 4))

3. Creating MatrixChain task...
4. Evaluating task (computing Y=AX, Z=YB, assembling blocks)...
   xs_assembled shape: torch.Size([2, 36, 12]) (expected: (2, 36, 12))
   ys shape: torch.Size([2, 36]) (expected: (2, 36))

...

✓ All tests passed!
```

## 技术细节 (Technical Details)

### 块对角结构 (Block Diagonal Structure)

对于L=3, n=4的情况：
- 每个块M_i的大小：12×12 (3n × 3n)
- 总序列长度：36 (L * 3n = 3 * 12)
- 特征维度：12 (3n)

For L=3, n=4:
- Each block M_i size: 12×12 (3n × 3n)
- Total sequence length: 36 (L * 3n = 3 * 12)
- Feature dimension: 12 (3n)

块结构示意 (Block structure):
```
Position  0-3:  X_0 block (rows 0-3,  cols 0-3)
Position  4-7:  Y_0 block (rows 4-7,  cols 4-7)
Position  8-11: Z_0 block (rows 8-11, cols 8-11)
Position 12-15: X_1 block (rows 12-15, cols 0-3)
Position 16-19: Y_1 block (rows 16-19, cols 4-7)
Position 20-23: Z_1 block (rows 20-23, cols 8-11)
Position 24-27: X_2 block (rows 24-27, cols 0-3)
Position 28-31: Y_2 block (rows 28-31, cols 4-7)
Position 32-35: Z_2 block (rows 32-35, cols 8-11)
```

### 损失计算 (Loss Calculation)

- 训练时只在最后一个位置计算损失：`loss_func(output[:,-1], ys[:,-1])`
- 但每个位置都有对应的目标值
- Y和Z位置的目标是对应行的均值
- X位置的目标为0

During training, loss is only computed on the last position, but each position has a corresponding target. Y and Z positions have targets equal to the mean of their rows. X positions have target 0.

## 扩展 (Extensions)

如果需要使用非方阵或不同大小的矩阵，可以修改：

If you need non-square or different-sized matrices, modify:
- `MatrixChainSampler.__init__()`: 移除 `assert self.n == self.m`
- `MatrixChain.__init__()`: 移除 `assert n == m == p == q`
- `MatrixChain.evaluate()`: 调整块对角矩阵的组装逻辑

## 问题排查 (Troubleshooting)

### 常见错误 (Common Errors)

1. **维度不匹配 (Dimension mismatch)**
   - 确保 `n_dims = 3 * n`
   - 确保 `n_positions >= L * 3 * n`

2. **AssertionError: n=m=p=q**
   - 当前实现要求所有矩阵大小相同
   - 使用相同的值：`{"L": 3, "n": 4, "m": 4, "p": 4, "q": 4}`

3. **训练不收敛 (Training not converging)**
   - 尝试调整学习率
   - 增加训练步数
   - 检查curriculum设置

## 参考 (References)

参考类似任务的实现：
- `src/samplers.py`: `GaussianSampler`
- `src/tasks.py`: `LinearRegression`, `Relu2nnRegression`

