# 实验运行说明

## 修改内容

### 1. WandB 已被禁用
在 `src/train.py` 中，wandb.init() 已添加 `mode="disabled"` 参数，不会再上传数据到WandB服务器。

### 2. 实验脚本
创建了自动化实验脚本：
- `run_experiments.py` - Python脚本，核心逻辑
- `run_experiments.sh` - Bash wrapper，方便调用

## 如何运行实验

### 方式1：使用 Bash 脚本（推荐）
```bash
./run_experiments.sh
```

### 方式2：直接运行 Python 脚本
```bash
python3 run_experiments.py
```

## 实验设计

该脚本会运行以下实验组合（**总共20次实验**）：

| V  | num_training_examples | 重复次数 | 说明 |
|----|----------------------|---------|------|
| 5  | 256                  | 5次     | 小规模图，少量样本 |
| 5  | 1,000,000            | 5次     | 小规模图，大量样本 |
| 20 | 256                  | 5次     | 大规模图，少量样本 |
| 20 | 1,000,000            | 5次     | 大规模图，大量样本 |

## 实验参数说明

- **V**: 表的节点数（vertices）- 控制图的规模
- **num_training_examples**: 训练样本数量 - 控制训练集大小
- **C**: 固定为 3（连接参数）
- **rho**: 固定为 0.5（密度参数）

其他参数继承自 `src/conf/table_connectivity.yaml`。

## 技术细节

Python脚本的工作原理：
1. 读取基础配置文件 `conf/table_connectivity.yaml`
2. 为每次实验动态修改参数（V, num_training_examples, run_name）
3. 创建临时配置文件
4. 运行训练脚本
5. 清理临时配置文件
6. 记录成功/失败状态

## 输出位置

每次运行的模型和配置文件会保存在：
```
../models/table_connectivity/<run_id>/
```

每个运行会生成唯一的 run_id (UUID)，包含：
- `config.yaml` - 完整配置
- `state.pt` - 最新检查点
- `model_*.pt` - 定期保存的模型（如果配置了 keep_every_steps）

## 运行状态

脚本会显示：
- 总实验数量
- 当前实验的参数（V, num_examples）
- 每次运行的进度
- 成功/失败状态
- 最终统计信息

## 注意事项

- ⚠️ 每次运行都会生成新的 UUID，不会覆盖之前的结果
- ⚠️ 确保有足够的磁盘空间（每个模型可能较大）
- ⚠️ 如果某次运行失败，脚本会继续执行后续实验
- ✅ WandB 已禁用，不会上传数据
- ✅ 可以随时中断脚本（Ctrl+C），已完成的实验结果会保留

## 预期运行时间

取决于：
- GPU性能
- 配置的 train_steps（默认1001步）
- batch_size（默认256）

每次实验大约需要几分钟到几十分钟不等。

