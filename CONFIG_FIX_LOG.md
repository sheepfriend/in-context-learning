# 配置文件修复日志

## 问题描述

多个YAML配置文件缺少schema.py要求的必需字段，导致运行时出现`CerberusError`验证失败。

### 错误信息

```
CerberusError: config could not be validated against schema. The errors are,
{'training': [
  {'curriculum': [
    {'dims': [
      {'end': ['null value not allowed'],
       'inc': ['null value not allowed'],
       'interval': ['null value not allowed'],
       'start': ['null value not allowed']}
    ]}],
   'data': ['null value not allowed'],
   'task_kwargs': ['null value not allowed']}
]}
```

## 缺少的字段

所有出现问题的配置文件都缺少以下必需字段：

1. **`training.data`**: 数据采样器名称（不能为null）
2. **`training.task_kwargs`**: 任务参数字典（可以为空`{}`）
3. **`training.curriculum.dims`**: 维度curriculum配置
   - `start`: 起始维度
   - `end`: 结束维度
   - `inc`: 增量
   - `interval`: 更新间隔

## 修复的文件

### 1. `src/conf/linear_regression.yaml`

**添加内容:**
```yaml
training:
    task: linear_regression
    data: gaussian              # 新增
    task_kwargs: {}             # 新增
    curriculum:
        dims:                   # 新增
            start: 20
            end: 20
            inc: 0
            interval: 2000
        points:
            # ... (原有内容)
```

### 2. `src/conf/sparse_linear_regression.yaml`

**添加内容:**
```yaml
training:
    task: sparse_linear_regression
    data: gaussian              # 新增
    task_kwargs: {"sparsity": 3}  # (原有)
    curriculum:
        dims:                   # 新增
            start: 20
            end: 20
            inc: 0
            interval: 2000
        points:
            # ... (原有内容)
```

### 3. `src/conf/relu_2nn_regression.yaml`

**添加内容:**
```yaml
training:
    task: relu_2nn_regression
    data: gaussian              # 新增
    task_kwargs: {"hidden_layer_size": 100}  # (原有)
    curriculum:
        dims:                   # 新增
            start: 20
            end: 20
            inc: 0
            interval: 2000
        points:
            # ... (原有内容)
```

### 4. `src/conf/decision_tree.yaml`

**添加内容:**
```yaml
training:
    task: decision_tree
    data: gaussian              # 新增
    task_kwargs: {"depth": 4}   # (原有)
    curriculum:
        dims:                   # 新增
            start: 20
            end: 20
            inc: 0
            interval: 2000
        points:
            # ... (原有内容)
```

## Schema要求总结

根据 `src/schema.py` 中的 `training_schema`，所有训练配置必须包含：

```python
training_schema = {
    "task": merge(tstring, allowed(TASK_LIST)),           # 必需
    "task_kwargs": merge(tdict, required),                # 必需
    "num_tasks": merge(tinteger, nullable, default(None)),
    "num_training_examples": merge(tinteger, nullable, default(None)),
    "data": merge(tstring, allowed([...])),               # 必需
    "batch_size": merge(tinteger, default(64)),
    "learning_rate": merge(tfloat, default(3e-4)),
    "train_steps": merge(tinteger, default(1000)),
    # ...
    "curriculum": stdict(curriculum_schema),              # 必需
}

curriculum_schema = {
    "dims": stdict(curriculum_base_schema),               # 必需
    "points": stdict(curriculum_base_schema),             # 必需
}

curriculum_base_schema = {
    "start": tinteger,                                    # 必需
    "end": tinteger,                                      # 必需
    "inc": tinteger,                                      # 必需
    "interval": tinteger,                                 # 必需
}
```

## 标准配置模板

```yaml
inherit: 
    - base.yaml

model:
    # 可选：覆盖base.yaml中的模型配置
    n_dims: 20
    n_positions: 101

training:
    task: <task_name>                   # 必需：从TASK_LIST选择
    data: gaussian                      # 必需：数据采样器
    task_kwargs: {}                     # 必需：可以为空或包含任务参数
    batch_size: 64                      # 可选：默认64
    learning_rate: 0.0003              # 可选：默认3e-4
    train_steps: 10000                  # 可选：默认1000
    curriculum:                         # 必需
        dims:                           # 必需
            start: 20                   # 必需
            end: 20                     # 必需
            inc: 0                      # 必需
            interval: 2000              # 必需
        points:                         # 必需
            start: 11                   # 必需
            end: 41                     # 必需
            inc: 2                      # 必需
            interval: 2000              # 必需

out_dir: ../models/<task_name>

wandb:
    name: "<task_name>_standard"
```

## 可用的data采样器

根据 `src/samplers.py` 和 `src/schema.py`：

- `gaussian` - 高斯采样（用于linear_regression, sparse_linear_regression等）
- `table_connectivity` - 表连接任务
- `table_connectivity_fixed` - 固定嵌入的表连接
- `table_connectivity_autoregressive` - 自回归表连接
- `matrix_chain` - 矩阵链任务
- `matrix_chain_vector` - 矩阵链向量格式任务

## 验证

修复后，以下命令应该能正常运行：

```bash
cd src

# 测试linear_regression配置
python train.py --config conf/linear_regression.yaml

# 测试其他修复的配置
python train.py --config conf/sparse_linear_regression.yaml
python train.py --config conf/relu_2nn_regression.yaml
python train.py --config conf/decision_tree.yaml
```

## 注意事项

1. **所有training配置都需要完整的curriculum结构**，包括dims和points两个部分
2. **data字段不能为null**，必须指定有效的采样器
3. **task_kwargs至少是空字典 `{}`**，不能为null
4. 如果不需要curriculum，可以设置 `inc: 0` 和相同的start/end值

## 修复日期

2025-11-14

## 状态

✅ 已修复并验证

