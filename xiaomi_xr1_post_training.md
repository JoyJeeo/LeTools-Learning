# xiaomi-robotics1使用教程
## 安装环境
```
conda create -n mibot_xr1 python=3.12 -y
conda activate mibot_xr1
cd kuavo_model/external_models/Xiaomi-Robotics-1/xr1
pip install -e .
pip install flash-attn --no-build-isolation
pip install -e /mnt/huangruichen/LeTools-Learning/third_party/lerobot #换成你自己的
pip install 'lerobot[dataset]'
pip install --no-deps 'huggingface-hub==0.36.0'
```

## 算stats
```
python tools/compute_lerobot_stats.py \
  --root /mnt/huangruichen/data/sim_task1/lerobot \
  --repo_id lerobot/kuavo_task \
  --modality_config \
    /mnt/huangruichen/LeTools-Learning/kuavo_data/modality_templates/kuavo_both_xr1.modality.json \
  --action_length 30 \
  --output outputs/norm/kuavo_xr1_stats.json
```
## 训练
改yaml也行，传args也行，下面的自己改一下
在posttrain.yaml里，这个设置为false
async_train=false
```
CUDA_VISIBLE_DEVICES=4,5,6,7 RESOURCE_GPU=4 bash scripts/train.sh \
  data=kuavo \
  model=posttrain \
  trainer.project=kuavo \
  trainer.exp_name=xr1_kuavo \
  trainer.default_root_dir=outputs/test \
  trainer.max_steps=10 \
  data.params.train_datasets.root=/mnt/huangruichen/data/sim_task1/lerobot \
  data.params.train_datasets.repo_id=lerobot/kuavo_task \
  data.params.train_datasets.modality_config_path=/mnt/huangruichen/LeTools-Learning/kuavo_data/modality_templates/kuavo_both_xr1.modality.json \
  data.params.train_datasets.stats_path=/mnt/huangruichen/LeTools-Learning/kuavo_model/external_models/Xiaomi-Robotics-1/xr1/outputs/norm/kuavo_xr1_stats.json \
  model.params.pretrained=/mnt/huangruichen/weights/XiaomiRobotics--Xiaomi-Robotics-1-5B/snapshots/master/model_states.pt \
  model.params.model.vlm_path=/mnt/huangruichen/weights/Qwen3-VL-4B-Instruct
```


> 下面是AI写的，内容全一点
# Xiaomi-Robotics-1 推理服务

## 启动

XR1 源码位于：

```text
/home/ruichen/projects/kdc_new/kuavo_learning_studio/kuavo_model/external_models/Xiaomi-Robotics-1
```

后训练 checkpoint 目录需要包含：

```text
<checkpoint>/config.py
<checkpoint>/last.ckpt/checkpoint/mp_rank_00_model_states.pt
```

启动前在 `configs/server/launcher.yaml` 的 `xiaomi1` 段填写 checkpoint 和
Qwen3-VL 路径，然后执行：

```bash
python kuavo_server/launch.py xiaomi1 --dry-run
python kuavo_server/launch.py xiaomi1
```

也可以直接覆盖配置：

```bash
python kuavo_server/launch.py xiaomi1 \
  --checkpoint /path/to/xr1/posttrain/checkpoint \
  --processor_path /path/to/Qwen3-VL-4B-Instruct
```

## 输入和输出

服务输入沿用 Kuavo 的三个 RGB 视角、任务文本和 16 维双臂状态，状态顺序为：

```text
左臂 7 关节, 左夹爪, 右臂 7 关节, 右夹爪
```

对于官方 EEF 后训练 checkpoint，适配器调用 XR1 官方 `compose_state` 将状态放入
`(1, 60)`，模型输出为 `(30, 60)`：

| 维度 | 语义 |
| --- | --- |
| `0:3` | 左臂末端位置增量 |
| `3:6` | 左臂末端轴角增量 |
| `6` | 左夹爪增量 |
| `8:11` | 右臂末端位置增量 |
| `11:14` | 右臂末端轴角增量 |
| `14` | 右夹爪增量 |
| `16` | 腰部增量 |
| `17:20` | 底盘速度 |

其余维度是 XR1 原生 padding。对于下面流程训练的 Kuavo checkpoint，`config.py`
内嵌 modality 配置，模型内部仍是 60 维，但有效维度为 `0:16`。适配器会自动按
同一配置反归一化并恢复为 16 维绝对关节/夹爪目标。

两种 checkpoint 都使用各自的 action mask、状态 `q01/q99` 和逐时间步动作
`mean/std`。`execution_horizon=10` 返回前 10 步，设为 `0` 返回完整 30 步。

官方 EEF checkpoint 的输出不是关节目标，不能直接截取前 16 维；Kuavo checkpoint
则由 adapter 输出 16 维关节目标。

## Kuavo LeRobot 后训练

使用新建的 modality 文件：

```text
kuavo_data/modality_templates/kuavo_both_xr1.modality.json
```

它定义源数据顺序为 `左臂7 + 左夹爪1 + 右臂7 + 右夹爪1`，动作表示为相对当前
状态的 delta，映射到 XR1 的 `0:16`。模型其余 44 维补零，并在统计、loss 和推理
阶段全部 mask。

先安装本仓库的 LeRobot：

```bash
conda run -n mibot_xr1 pip install -e \
  /home/ruichen/projects/kdc_new/kuavo_learning_studio/third_party/lerobot
```

计算状态和动作统计量：

```bash
cd /home/ruichen/projects/kdc_new/kuavo_learning_studio/kuavo_model/external_models/Xiaomi-Robotics-1/xr1

python tools/compute_lerobot_stats.py \
  --root /media/ruichen/5fe8ed68-6ff6-464f-af10-a89b65c040cf/sim/sim/TASK1-ToySorting/lerobot \
  --repo_id lerobot/kuavo_task \
  --modality_config \
    /home/ruichen/projects/kdc_new/kuavo_learning_studio/kuavo_data/modality_templates/kuavo_both_xr1.modality.json \
  --action_length 30 \
  --output outputs/norm/kuavo_xr1_stats.json
```

统计文件包含：

- `(30, 60)` 的 action `mean/std`；只有有效动作维度参与统计。
- `(1, 60)` 的 state `q01/q99`；只有有效状态维度参与统计。
- modality 签名，用于阻止统计文件和维度配置混用。

训练时使用 `data=kuavo`，并通过 Hydra override 指定路径，不需要修改默认 JSON：

```bash
RESOURCE_GPU=1 bash scripts/train.sh \
  data=kuavo \
  model=posttrain \
  trainer.project=kuavo \
  trainer.exp_name=xr1_kuavo \
  trainer.default_root_dir=outputs/test \
  trainer.max_steps=10000 \
  data.params.train_datasets.root=/media/ruichen/5fe8ed68-6ff6-464f-af10-a89b65c040cf/sim/sim/TASK1-ToySorting/lerobot \
  data.params.train_datasets.repo_id=lerobot/kuavo_task \
  data.params.train_datasets.modality_config_path=/home/ruichen/projects/kdc_new/kuavo_learning_studio/kuavo_data/modality_templates/kuavo_both_xr1.modality.json \
  data.params.train_datasets.stats_path=/home/ruichen/projects/kdc_new/kuavo_learning_studio/kuavo_model/external_models/Xiaomi-Robotics-1/xr1/outputs/norm/kuavo_xr1_stats.json \
  model.params.pretrained=/home/ruichen/.cache/modelscope/models/XiaomiRobotics--Xiaomi-Robotics-1-5B/snapshots/master/model_states.pt \
  model.params.model.vlm_path=/home/ruichen/Qwen3-VL-4B-Instruct
```

## 训练参数在哪里调整

默认参数分布在三个配置文件中：

| 类型 | 配置文件 |
| --- | --- |
| 数据和 DataLoader | `xr1/configs/data/kuavo.yaml` |
| 训练步数、优化器、学习率和 checkpoint | `xr1/configs/trainer/deepspeed.yaml` |
| XR1 模型训练方式 | `xr1/configs/model/posttrain.yaml` |

可以直接修改这些 YAML，也可以在训练命令末尾添加 Hydra override。推荐用 override，
这样不会修改默认配置，并且最终取值会保存到训练目录的 `config.py/config.yaml`。

### 数据和显存

| 参数 | 含义 | 命令行示例 |
| --- | --- | --- |
| `data.params.train_datasets.batch_size` | 每张 GPU 的 batch size | `data.params.train_datasets.batch_size=4` |
| `data.params.num_workers` | 每个训练进程的 DataLoader worker | `data.params.num_workers=8` |
| `data.params.prefetch_factor` | 每个 worker 预取的 batch 数 | `data.params.prefetch_factor=4` |
| `data.params.train_datasets.fps` | LeRobot 数据帧率 | `data.params.train_datasets.fps=10` |
| `RESOURCE_GPU` | 当前节点使用的 GPU 数量 | `RESOURCE_GPU=4 bash scripts/train.sh ...` |
| `trainer.accumulate_grad_batches` | 梯度累积步数 | `trainer.accumulate_grad_batches=4` |
| `trainer.precision` | 训练精度 | `trainer.precision=bf16-mixed` |

有效全局 batch size 为：

```text
batch_size × 每节点 GPU 数 × 节点数 × accumulate_grad_batches
```

`action_length` 当前应保持为 `30`，因为 XR1 的动作头和预训练权重按 30-step chunk
构建；修改它不是单纯的数据参数调整。

### 步数、保存和断点续训

| 参数 | 含义 | 命令行示例 |
| --- | --- | --- |
| `trainer.max_steps` | 优化器更新总步数 | `trainer.max_steps=20000` |
| `trainer.save_interval` | 每隔多少训练 step 保存一次 | `trainer.save_interval=2000` |
| `trainer.gradient_clip_val` | 梯度裁剪阈值 | `trainer.gradient_clip_val=1.0` |
| `trainer.seed` | 随机种子 | `trainer.seed=42` |
| `trainer.default_root_dir` | 输出根目录 | `trainer.default_root_dir=/data/xr1_outputs` |
| `trainer.project` | W&B project，同时参与输出路径命名 | `trainer.project=kuavo` |
| `trainer.exp_name` | 实验名，同时参与输出路径命名 | `trainer.exp_name=xr1_task1` |
| `trainer.ckpt_path` | 从 Lightning/DeepSpeed checkpoint 续训 | `trainer.ckpt_path=/path/to/last.ckpt` |

`model.params.pretrained` 和 `trainer.ckpt_path` 含义不同：前者用于第一次后训练时加载
XR1 发布的 `model_states.pt`；后者用于恢复已经开始的训练，包括优化器和 scheduler
状态。

如果 `max_steps` 小于 `save_interval`，周期 checkpoint 不会触发，但训练正常结束时
仍会由 `save_last=true` 写出 `last.ckpt`。

### 学习率和优化器

实际学习率主要调整 `xr1/configs/trainer/deepspeed.yaml` 中的 scheduler：

| 参数 | 含义 | 命令行示例 |
| --- | --- | --- |
| `trainer.scheduler.params.num_warmup_steps` | warmup 步数 | `trainer.scheduler.params.num_warmup_steps=500` |
| `trainer.scheduler.params.warmup_lr_start` | warmup 初始学习率 | `trainer.scheduler.params.warmup_lr_start=5e-7` |
| `trainer.scheduler.params.max_lr` | 峰值学习率 | `trainer.scheduler.params.max_lr=2e-5` |
| `trainer.scheduler.params.min_lr` | cosine 结束学习率 | `trainer.scheduler.params.min_lr=5e-6` |
| `trainer.optimizer.params.weight_decay` | weight decay | `trainer.optimizer.params.weight_decay=0.1` |
| `trainer.optimizer.params.betas` | Adam betas | `trainer.optimizer.params.betas=[0.9,0.95]` |

这里 `trainer.optimizer.params.lr=1.0` 是配合当前 `LambdaLR` 实现使用的基准倍率，
通常不要把它当作真实学习率修改；真实范围由 `warmup_lr_start/max_lr/min_lr` 控制。
`num_training_steps` 已引用 `${trainer.max_steps}`，调整 `max_steps` 后会自动同步。

### XR1 模型训练方式

| 参数 | 含义 | 命令行示例 |
| --- | --- | --- |
| `model.params.model.async_train` | 是否训练随机 action prefix | `model.params.model.async_train=false` |
| `model.params.model.freq_coefficient` | 频域 loss 权重 | `model.params.model.freq_coefficient=0.5` |
| `model.params.model.freq_excluded_dims` | 不参与频域 loss 的维度 | `model.params.model.freq_excluded_dims=[17,18,19]` |
| `model.params.model.ffn_gradient_checkpointing` | 用计算换显存 | `model.params.model.ffn_gradient_checkpointing=true` |

Kuavo modality 的有效动作只有 `0:16`，其他维度已经由 action mask 排除；默认的
`freq_excluded_dims=[17,18,19]` 因而不会排除任何 Kuavo 有效关节维度。

当前不是“冻结 VLM 后只训练动作头”：源码只冻结了 VLM input embedding，其余 VLM、
视觉模块、DiT 和 projector 都参与训练。`ffn_gradient_checkpointing` 只是降低显存，
不是冻结参数。如果需要新的冻结策略，应在
`xr1/mibot/models/VLA/XR1.py` 的 `_build_model()` 中明确设置 `requires_grad`，当前
没有对应 YAML 开关。

下面这些参数目前也固定在 `XR1.py` 中，而不是 YAML 参数：`training_repeat=4`、
`num_steps=5`、`prefix_mask_prob=0.5`、`n_choices=5`。它们影响显存、采样或模型头
语义，不建议仅为普通后训练随意修改。

### 日志

W&B 的 project 和 run name 分别由 `trainer.project`、`trainer.exp_name` 控制。
没有 W&B 写权限或暂时不上传时，可以在命令前设置：

```bash
WANDB_MODE=offline RESOURCE_GPU=1 bash scripts/train.sh ...
```

如果要让 loss 显示在终端进度条，需要修改
`xr1/mibot/models/runner/base_runner.py` 的 `training_step()`，为对应的 `self.log()`
增加 `on_step=True, on_epoch=False, prog_bar=True`。这只影响显示，不改变训练计算。

### 一个常见的低显存示例

```bash
RESOURCE_GPU=1 bash scripts/train.sh \
  data=kuavo model=posttrain \
  data.params.train_datasets.batch_size=1 \
  data.params.num_workers=4 \
  trainer.accumulate_grad_batches=8 \
  trainer.max_steps=10000 \
  trainer.save_interval=1000 \
  model.params.model.ffn_gradient_checkpointing=true \
  model.params.model.async_train=false \
  data.params.train_datasets.root=/path/to/lerobot \
  data.params.train_datasets.modality_config_path=/home/ruichen/projects/kdc_new/kuavo_learning_studio/kuavo_data/modality_templates/kuavo_both_xr1.modality.json \
  data.params.train_datasets.stats_path=/path/to/xr1_stats.json \
  model.params.pretrained=/path/to/model_states.pt \
  model.params.model.vlm_path=/path/to/Qwen3-VL-4B-Instruct
```

训练目录会生成推理所需的 `config.py` 和
`last.ckpt/checkpoint/mp_rank_00_model_states.pt`。直接把该训练目录传给
`python kuavo_server/launch.py xiaomi1 --checkpoint ...` 即可。

## 环境

XR0 和 XR1 的 Python 包都名为 `mibot`，但依赖版本并不完全相同，因此 launcher
默认使用独立的 `mibot_xr1` 环境。首次使用时执行：

```bash
conda create -n mibot_xr1 python=3.12 -y
conda run -n mibot_xr1 pip install -e \
  /home/ruichen/projects/kdc_new/kuavo_learning_studio/kuavo_model/external_models/Xiaomi-Robotics-1/xr1
conda run -n mibot_xr1 pip install flash-attn --no-build-isolation
```

XR1 源码目录也会被适配器优先加入 Python 搜索路径，避免误加载 XR0 的同名包。

XR1 源码中的后训练 loader 已支持 `vlm_path`，因此 `--processor_path` 同时用于
构建 Qwen3-VL 网络和加载 processor，可直接使用本地目录。
