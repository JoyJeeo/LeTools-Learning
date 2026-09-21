# `configs/task/kuavo.yaml` 配置说明

本文说明 `/home/ruichen/projects/kdc_new/kuavo_learning_studio/kuavo_model/external_models/GalaxeaVLA/configs/task/kuavo.yaml` 中每个字段的作用、影响和修改建议。

## 1. 配置组合关系

`kuavo.yaml` 不是完整训练配置，而是 Kuavo 任务对公共配置的覆盖。执行：

```bash
bash scripts/run/finetune.sh 1 kuavo
```

Hydra 会组合：

```text
configs/train.yaml
├── configs/model/g05.yaml
├── configs/tokenizer/actioncodec.yaml
├── configs/data/kuavo.yaml
└── configs/task/kuavo.yaml    # 最后覆盖同名字段
```

```yaml
# @package _global_
defaults:
- override /model: g05
- override /tokenizer: actioncodec
- override /data: kuavo
- _self_
```

| 配置项 | 含义 |
|---|---|
| `@package _global_` | 将本文件合并到 Hydra 根配置，而非放入 `task.kuavo` 子节点。 |
| `/model: g05` | 加载 `configs/model/g05.yaml`。 |
| `/tokenizer: actioncodec` | 加载 `configs/tokenizer/actioncodec.yaml`。 |
| `/data: kuavo` | 加载 `configs/data/kuavo.yaml`。 |
| `_self_` | 最后应用本文件，因此本文件中的同名配置优先。 |

## 2. 分布式配置

```yaml
shard_datasets_by_node: false
use_fsdp: false
```

### `shard_datasets_by_node`

设计用途是在多节点训练时按节点拆分 mixture 中的数据集：

- `false`：所有节点访问相同数据集；
- `true`：不同节点加载不同数据集分片。

单机训练保持 `false`。当前仓库的 mixture dataset 中有相关实现，但主训练入口没有发现完整接线，因此不建议依赖该选项。

### `use_fsdp`

设计用途是启用 FSDP 参数、梯度和优化器状态分片。当前 `scripts/finetune.py` 实际使用普通 DDP，未发现该字段的有效接线，所以改成 `true` 不会自动获得 FSDP。保持 `false`。

## 3. 额外 VLM 联合训练

```yaml
vlm:
  cotrain: false
  batch_size: ${model.batch_size}
  num_workers: 2
  vlm_batch_ratio: 0.2
  loss_weight: 1.0
```

| 配置项 | 作用 | 当前影响 |
|---|---|---|
| `cotrain` | 是否同时加入额外纯 VLM 数据。 | `false`，Kuavo 只训练机器人 VLA 数据。 |
| `batch_size` | 额外 VLM dataloader 的 batch size。 | 引用 `model.batch_size`，仅 cotrain 开启时有意义。 |
| `num_workers` | VLM dataloader worker 数。 | cotrain 关闭时无实际影响。 |
| `vlm_batch_ratio` | 联合训练中 VLM batch 的比例。 | cotrain 关闭时无实际影响。 |
| `loss_weight` | VLM loss 权重。 | cotrain 关闭时无实际影响。 |

当前主训练入口中没有发现这组顶层 `cfg.vlm` 参数的完整使用路径，可能是预留或遗留配置。Kuavo 保持 `cotrain: false`。

## 4. 数据归一化统计

```yaml
datastatics_path: null
```

指定已有的 `dataset_stats.json`：

```yaml
datastatics_path: /path/to/dataset_stats.json
```

当前为 `null`，再结合：

```yaml
model:
  use_pretrained_norm_stats: false
```

最终行为是：

```text
不使用 G0.5 base 的统计量
+ 没有手动指定统计文件
= 根据 Kuavo 训练数据重新计算 state/action 统计量
```

这是 Kuavo fine-tune 的推荐设置，因为 base 模型的机器人统计量不适用于 Kuavo。

## 5. 日志

```yaml
logger:
  type: wandb
  mode: offline
  workspace: null
```

| 配置项 | 作用 |
|---|---|
| `type: wandb` | 使用 W&B 记录 loss、学习率和吞吐等指标。 |
| `mode: offline` | 仅写本地日志，不在线上传。 |
| `workspace: null` | W&B entity/team；离线模式下一般不重要。 |

其余日志参数继承自 `configs/train.yaml`。

## 6. 部署 checkpoint 标志

```yaml
deploy_checkpoint:
  enabled: true
```

名称上用于控制部署 checkpoint，但当前仓库 Python 代码中没有发现该字段的消费位置，可能由外部平台读取或属于预留配置。

普通训练 checkpoint 的保存间隔由 `configs/train.yaml` 中的以下参数控制：

```yaml
checkpointing_steps: 2000
```

## 7. ActionCodec / VQ tokenizer

```yaml
tokenizer:
  vq_config:
    block_wise_autoregressive: false
    block_size: 8
    num_residuals: 2
    rule_based_key_patterns: [gripper]
    rule_based_binarize_threshold: 0.0
    use_group_markers: true
    group_order_shuffle: false
    dropout_noop_parts: true
    absent_key_fill_value: -100.0
```

G0.5 同时包含连续 Flow Matching 分支和离散 ActionCodec/AR 分支。这部分控制离散动作 token。

### `block_wise_autoregressive`

是否按动作 block 自回归生成。当前关闭，并与 `model.model_arch.ar.block_wise_autoregressive` 保持一致。

### `block_size`

block-wise AR 开启时每个动作 block 的 token 大小。当前 block-wise 关闭，因此不是主要行为。

### `num_residuals`

Residual VQ 使用两层量化：

```text
第 1 层编码动作主体
第 2 层继续编码第 1 层的量化误差
```

增加层数可以提高离散重建能力，但会增加 token 数、序列长度和计算量。它必须与 ActionCodec 权重兼容，不建议修改。

### `rule_based_key_patterns`

名称匹配 `gripper` 的 part 使用规则式编码，而不是普通 VQ。因此 `left_gripper` 和 `right_gripper` 都会命中。

### `rule_based_binarize_threshold`

规则型夹爪动作在归一化空间中的二值化阈值：

```text
value >= 0 → 一类
value < 0  → 另一类
```

它作用于归一化后的夹爪值，不是原始硬件数值。

### `use_group_markers`

在动作 token 序列中加入 part 标记，以区分：

```text
left_control
left_gripper
right_control
right_gripper
lower_body
```

### `group_order_shuffle`

是否在训练时随机打乱 action group 的 token 顺序。当前固定顺序，更便于 Kuavo 数据验证。

### `dropout_noop_parts`

省略不存在或 no-op 的 action part。Kuavo 没有 `lower_body`，左右 `control` 中也存在 padding，因此启用后可减少无效离散动作 token。

### `absent_key_fill_value`

不存在的 part 使用 `-100.0` 作为中间填充值，以明确区分：

- 真实零动作；
- 该 part 不存在。

不要改为 `0`。

## 8. Tokenizer 内部 part 布局

```yaml
parts_meta:
  left_control: 9
  left_gripper: 1
  right_control: 9
  right_gripper: 1
  lower_body: 7
```

G0.5 内部统一布局为：

```text
left_control(9)
+ left_gripper(1)
+ right_control(9)
+ right_gripper(1)
+ lower_body(7)
= 27D
```

Kuavo 原始布局仍为：

```text
left_arm(7)
+ left_gripper(1)
+ right_arm(7)
+ right_gripper(1)
= 16D
```

对应关系：

```text
left_arm(7)      → left_control(9)，末尾 padding 2D
left_gripper(1)  → left_gripper(1)
right_arm(7)     → right_control(9)，末尾 padding 2D
right_gripper(1) → right_gripper(1)
无 lower body    → lower_body padding 7D
```

```yaml
model_arch:
  codebook_size: 4096
```

表示 VQ codebook 有 4096 个离散码。必须与 `checkpoints/action_tokenizer.pt` 匹配，不建议修改。

## 9. 基础权重与统计量

```yaml
model:
  pretrained_ckpt: checkpoints/g05-base/checkpoints/model_state_dict.pt
  model_arch:
    hf_processor_path: checkpoints/qwen3_5_2b_base_processor
  use_pretrained_norm_stats: false
```

### `pretrained_ckpt`

G0.5 base 模型权重，是 fine-tune 的初始化参数。它不是完整训练状态恢复；恢复 optimizer、scheduler 等状态应使用根配置中的 `resume_ckpt`。

### `model_arch.hf_processor_path`

Qwen3.5 文本 processor 目录，必须同时包含：

```text
tokenizer.json
tokenizer_config.json
```

它与 `pretrained_ckpt` 是两套不同资产。模型权重存在并不表示 processor 会自动从权重目录中发现；路径错误时会在模型实例化阶段报 `No such file or directory`。

### `use_pretrained_norm_stats`

是否复用 base checkpoint 的 state/action 统计量。Kuavo 应保持 `false`，否则可能使用错误的机器人量纲。

## 10. BF16 和优化器显存

```yaml
enable_bf16_training: true
model_weights_to_bf16: false
use_8bit_optimizer: true
```

### `enable_bf16_training`

启用 BF16 mixed precision。大部分前向和反向计算使用 BF16，以降低激活显存并提高吞吐。

### `model_weights_to_bf16`

是否把模型参数本身转换成 BF16。当前为 `false`：

```text
模型参数主要保留 FP32
计算使用 BF16 autocast
```

这通常更稳定，但参数显存高于全 BF16 权重。

### `use_8bit_optimizer`

使用 bitsandbytes `AdamW8bit`，主要降低 optimizer state 显存。需要 bitsandbytes，且不能与 fused AdamW 同时使用。

## 11. Batch 与 dataloader

```yaml
batch_size: 1
grad_accumulation_steps: 8
num_workers: 4
prefetch_factor: 2
```

### `batch_size`

每张 GPU 每次 forward 的样本数。

### `grad_accumulation_steps`

累计 8 次 micro-batch 后执行一次 optimizer update。有效全局 batch 为：

```text
batch_size × grad_accumulation_steps × GPU 数量
```

示例：

```text
单卡：1 × 8 × 1 = 8
四卡：1 × 8 × 4 = 32
```

### `num_workers`

每个训练进程的数据加载 worker 数。多卡 DDP 下，每张卡都会创建自己的 workers。

### `prefetch_factor`

每个 worker 预取的 batch 数。当前每个训练进程大约预取：

```text
4 workers × 2 = 8 batches
```

增大可减少 GPU 等待，但会提高 CPU 内存和视频解码压力。

## 12. 训练长度

```yaml
max_epochs: 10
max_steps: null
```

当前按 epoch 训练，完整遍历数据集 10 次。

也可以改为固定 optimizer step：

```yaml
max_epochs: null
max_steps: 10000
```

不能同时设置有效的 `max_epochs` 和 `max_steps`，训练代码会直接报错。

## 13. 学习率与 scheduler

```yaml
learning_rate: 4.0e-05
warmup_ratio: null
warmup_steps: 200
lr_scheduler_type: cosine
lr_min_ratio: 0.0
constant_end_ratio: 0.08
weight_decay: 0.03
```

### `learning_rate`

主学习率为 `4e-5`。

### `warmup_ratio`

非空时按总训练步数计算 warmup：

```text
warmup_steps = total_steps × warmup_ratio
```

当前为 `null`，所以采用固定的 `warmup_steps`。

### `warmup_steps`

前 200 个 optimizer updates 进行学习率 warmup。它按参数更新次数计算，不按 micro-batch 计算。

### `lr_scheduler_type`

使用 cosine 学习率调度。

### `lr_min_ratio`

最低学习率占初始学习率的比例。当前为 0，理论最低值为 0。

### `constant_end_ratio`

scheduler 最后 8% 区间保持最终学习率，具体行为由仓库自定义 scheduler 实现决定。

### `weight_decay`

AdamW 权重衰减系数，用于正则化。过大可能限制任务适配，过小可能增加过拟合。

## 14. Processor 配置

```yaml
processor:
  vlm_input_action_norm_default_mode: null
  vlm_input_action_norm_exception_mode: null
```

不为 VLM/AR 动作输入增加第二套归一化覆盖规则，沿用 dataset processor 建立的动作表示。

### Tokenizer 只读取本地文件

```yaml
tokenizer_params:
  local_files_only: true
```

Hugging Face processor/tokenizer 只从本地加载，不联网查找。

### 相机输入尺寸

```yaml
camera_size_config:
  exterior: [256, 256]
  wrist_left: [256, 256]
  wrist_right: [256, 256]
```

头部、左腕和右腕图像进入模型前统一 resize 到 `256×256`。不会修改原始视频文件。

### Action filter

```yaml
action_filter:
  _target_: g05.data_processor.transforms.action_filter.DummyActionFilter
```

不额外屏蔽真实动作维度。padding mask 由 merger 生成，不由这里负责。

### Action/state merger

```yaml
action_state_merger:
  _target_: g05.data_processor.transforms.action_state_merger.GroupedPaddingMerger
  max_action_shape_meta: ${oc.load:configs/data/parts_meta/r1pro.yaml,parts_meta}
  max_state_shape_meta: ${oc.load:configs/data/parts_meta/r1pro.yaml,parts_meta}
  merge_spec: ${oc.load:configs/data/parts_meta/r1pro.yaml,merge_spec}
  merge: true
```

训练前：

```text
Kuavo 16D part 字典 → padding/group merge → 模型内部 27D
```

推理后：

```text
模型内部 27D → inverse merge → Kuavo 四个 part → server 拼成 16D
```

`r1pro.yaml` 的 `merge_spec` 定义：

```yaml
left_control: [left_arm, left_ee_pose]
left_gripper: [left_gripper]
right_control: [right_arm, right_ee_pose]
right_gripper: [right_gripper]
lower_body: [lower_body, torso]
```

Kuavo 提供 `left_arm/right_arm`，因此左右臂各自从 7D padding 到 9D control group。

`merge: true` 必须保持开启。

## 15. 模型内部维度

```yaml
model_arch:
  action_dim: 27
  proprio_dim: 27
```

| 参数 | 作用 |
|---|---|
| `action_dim` | Action Expert 的输入/输出维度。 |
| `proprio_dim` | 模型接收的机器人状态维度。 |

不要改成 16。16D 是 Kuavo 的外部原始协议，27D 是 G0.5 的内部 grouped layout。

完整链路：

```text
Kuavo state/action 16D
  → 拆成左右臂和夹爪
  → padding/group merge 为 27D
  → G0.5 模型输入/输出 27D
  → processor 逆变换为四个 Kuavo part
  → kuavo_server 拼成 16D action chunk
```

## 16. Activation checkpointing

```yaml
checkpoint_vision: true
checkpoint_vlm: true
checkpoint_action_expert: true
```

这些字段不是保存 checkpoint，而是 gradient/activation checkpointing：forward 不保存全部中间激活，backward 时重新计算，以计算时间换显存。

| 参数 | 作用模块 |
|---|---|
| `checkpoint_vision` | 图像编码器。 |
| `checkpoint_vlm` | Qwen VLM backbone。 |
| `checkpoint_action_expert` | 连续动作专家。 |

24GB GPU 上建议全部保持开启。

## 17. 视觉编码器学习率倍率

当前文件写法：

```yaml
model:
  model_arch:
    vision_lr_multiplier: 0.1
```

设计意图是让视觉编码器学习率为主学习率的 10%：

```text
4e-5 × 0.1 = 4e-6
```

但训练代码读取的是：

```python
cfg.model.get("vision_lr_multiplier", 1.0)
```

位置：`scripts/finetune.py:897-904`。

因此当前字段放在 `model_arch` 下不会被 optimizer 读取，实际会回退为 `1.0`。如需真正生效，应调整为：

```yaml
model:
  vision_lr_multiplier: 0.1
  model_arch:
    # 其他模型结构配置
```

## 18. Flow Matching

```yaml
fm:
  num_flow_samples: 1
```

每个机器人样本采样一组 flow time/noise。若设为 4，Action Expert 的有效 batch 约变成 `4×B`：

- 优点：降低 flow matching 梯度方差；
- 代价：增加显存、计算量和训练时间。

24GB GPU 建议保持 1。

## 19. AR 离散动作

```yaml
ar:
  block_wise_autoregressive: false
```

关闭 block-wise AR，与 tokenizer 中同名字段保持一致。

## 20. CoT 与终止标记

```yaml
predict_cot: false
input_preprocessor:
  pred_eov: false
```

### `predict_cot`

是否额外预测规划/Chain-of-Thought 文本。Kuavo 数据目前只有任务描述，没有专门的 CoT 标签，因此保持 `false`。

### `pred_eov`

是否预测 EOV 终止标记。普通固定长度 action chunk fine-tune 不需要，保持 `false`。

## 21. 修改建议

### 通常需要修改

```yaml
model:
  pretrained_ckpt: /实际/G05-base/model_state_dict.pt
  batch_size: 1
  grad_accumulation_steps: 8
  max_epochs: 10
  learning_rate: 4.0e-05
```

数据路径不在本文件中，而在：

```text
configs/data/kuavo.yaml
```

### 不建议修改

```yaml
tokenizer.vq_config.parts_meta
tokenizer.vq_config.codebook_size
model.model_arch.action_dim
model.model_arch.proprio_dim
model.processor.action_state_merger
```

这些字段共同定义 G0.5 的内部 27D 协议，必须互相一致。

### 当前需要注意的层级问题

如果希望视觉学习率倍率生效，应将：

```yaml
model.model_arch.vision_lr_multiplier
```

移动为：

```yaml
model.vision_lr_multiplier
```

## 22. 最小配置关注清单

开始训练前重点确认：

1. `pretrained_ckpt` 指向正确的 G0.5 base 权重；
2. `configs/data/kuavo.yaml` 中数据路径正确；
3. 数据真实 layout 是 `[左臂7, 左夹爪1, 右臂7, 右夹爪1]`；
4. LeRobot V3 数据包含 `meta/tasks.parquet` 和帧级 `task_index`；
5. `action_dim/proprio_dim` 保持 27；
6. `use_pretrained_norm_stats` 保持 `false`；
7. 根据显存调整 `batch_size` 和 `grad_accumulation_steps`，不要通过改变动作协议省显存。
