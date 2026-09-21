# xiaomi xr0使用教程
做的修改有：
数据部分，dataloader兼容lerobot格式的数据，计算stats也通过lerobot数据来计算
设置了action layout来统一读取json文件中的动作空间维度，而非写死的(3+3+1+6+1)*2，默认的配置是kuavo_data/modality_templates/kuavo_both_xr0.modality.json, action_layout会读取其中的配置，算norm stats时会mask掉多余的维度
模型主体部分未修改
## 安装环境
```
conda create -n mibot python=3.12 -y
conda activate mibot
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install transformers==4.57.1
pip install zmq
pip uninstall -y ninja && pip install ninja

# 这里需要安装flash atten 2.8.3,建议直接来找我要wheel安装的比较快
pip install your_path/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl

sudo apt-get install -y libegl1 libgl1 libgles2

# 以上安装好之后
cd kuavo_model/external_models/Xiaomi-Robotics-0/xr0
pip install -e .
pip install --no-deps -e ../../../../third_party/lerobot
pip install 'lerobot[dataset]'
pip install --no-deps 'huggingface-hub==0.36.0'

```

## 训练
### 首先要转换权重
这里也建议直接来找我拿转完的权重，就不用自己下载和转换了，
当然也可以自己下载和转换，下载地址是https://huggingface.co/XiaomiRobotics/Xiaomi-Robotics-0-Pretrain
```
python tools/weight_convert.py \
  --model_path /your/pretrained/model/path \
  --output_dir pretrained_ckpt \
  --output_filename xr0_pretrained.pt
```
### 计算stats
config用`kuavo_data/modality_templates/kuavo_both_xr0.modality.json`
```
python tools/compute_lerobot_stats.py \
  --root /path/to/lerobot \
  --repo_id lerobot/kuavo_task \
  --modality_config path/to/your/kuavo_both_xr0.modality.json \  #这里自己改一下
  --action_length 30 \
  --output /home/ruichen/xr0_outputs/kuavo_task/xr0_action_stats.json
```
## 配置修改

修改这个 `configs/data/kuavo.yaml`：
`configs/trainer/deepspeed.yaml`也可以修改一些，比如save_interval
记得设置
async_train=false
训练入口会读取 modality 和 stats，并将两者完整写入最终 `config.py`。推理阶段不需要再次提供 modality JSON 或 stats 文件.

## 启动训练
按实际修改
```bash

CUDA_VISIBLE_DEVICES=0 RESOURCE_GPU=1 \
bash scripts/train.sh \
  data=kuavo \
  model=XR0 \
  trainer.project=kuavo \
  trainer.exp_name=xr0_kuavo_sync \
  trainer.default_root_dir=outputs/test \
  trainer.max_steps=60000 \
  model.params.pretrained=/pretrained_ckpt/xr0_pretrained.pt \
  model.params.model.vlm_path=/home/ruichen/Qwen3-VL-4B-Instruct
```
### 训练配置
其实这里没什么要改的，如果要改，AI整理了一下，就是下面的部分
#### 1. 训练器、学习率、保存策略

  文件：kuavo_model/external_models/Xiaomi-Robotics-0/xr0/configs/trainer/deepspeed.yaml:3

  主要参数：

  trainer:
    precision: "bf16-mixed"
    max_steps: 30000
    save_interval: 5000
    accumulate_grad_batches: 1
    gradient_clip_val: 1.0

    scheduler:
      params:
        num_warmup_steps: 2000
        warmup_lr_start: 5e-7
        max_lr: 1e-4
        min_lr: 5e-7

    project: demo
    exp_name: test
    default_root_dir: ./test/
    ckpt_path: null

  含义：

  - max_steps：optimizer 更新总步数。
  - save_interval：每多少 step 保存一次 checkpoint。
  - accumulate_grad_batches：梯度累积。
  - precision：训练精度，4090 建议保持 bf16-mixed。
  - gradient_clip_val：梯度裁剪阈值。
  - num_warmup_steps：学习率 warmup 步数。
  - max_lr/min_lr：最高、最低学习率。
  - ckpt_path：恢复完整训练状态，例如 last 或 checkpoint 路径。
  - project/exp_name/default_root_dir：W&B 和 checkpoint 输出位置。

  注意不要直接修改：

  optimizer:
    params:
      lr: 1.0

  当前自定义 scheduler 依赖基础 LR 为 1.0，实际学习率应修改 scheduler.params.max_lr/min_lr。

  val_check_interval 当前基本不起作用，因为 DataModule 的 val_dataloader() 返回空列表，没有验证集。

  ## 2. 数据参数

  文件：kuavo_model/external_models/Xiaomi-Robotics-0/xr0/configs/data/kuavo.yaml:3

  train_datasets:
    repo_id: lerobot/kuavo_task
    root: /path/to/lerobot
    fps: 10
    batch_size: 16
    action_length: 30
    modality_config_path: /path/to/kuavo_both_xr0.modality.json
    stats_path: /path/to/xr0_action_stats.json
    video_backend: pyav

  其中：

  - batch_size：每张 GPU 的 batch size。
  - fps：必须与 LeRobot 数据集一致。
  - action_length：action chunk 长度。
  - root/repo_id：LeRobot 数据集。
  - modality_config_path：动作维度和顺序。
  - stats_path：对应 modality 计算出的 mean/std。
  - video_backend：你的 AV1 数据建议用 pyav。

  以下三项必须同步：

  data.action_length
  model.action_shape[0]
  stats.action_length

  有效全局 batch size：

  batch_size × GPU 数 × 节点数 × accumulate_grad_batches

  ## 3. 模型参数

  文件：kuavo_model/external_models/Xiaomi-Robotics-0/xr0/configs/model/XR0.yaml:3

  model:
    params:
      pretrained: null
      model:
        vlm_path: Qwen/Qwen3-VL-4B-Instruct
        state_shape: [1, 32]
        action_shape: [30, 32]
        dit_num_layers: 16
        training_repeat: 4
        enable_freq: true
        prefix_mask_prob: 0.5
        async_train: false

  参数作用：

  - pretrained：XR0 预训练初始化权重。
  - vlm_path：Qwen3-VL 模型或 processor 路径。
  - training_repeat：一次 VLM 编码对应多少份 flow 训练采样；越大显存和 DiT 计算量越高。
  - enable_freq：是否使用频域 loss。
  - async_train：是否训练 action-prefix/异步执行能力。
  - prefix_mask_prob：async prefix attention 随机 mask 概率。
  - dit_num_layers：DiT 层数。

  代码还支持但 YAML 没有显式写出的参数：

  dit_hidden_size: 1024
  num_steps: 5
  flow_sampling: beta
  local_window: 4

  可以添加到 model.params.model，或者用 Hydra 的 + 参数：

  +model.params.model.num_steps=5
  +model.params.model.flow_sampling=beta
  +model.params.model.local_window=4

  num_steps主要影响推理采样速度和质量，不直接决定训练 step。

  state_shape/action_shape/dit_num_layers/dit_hidden_size涉及模型结构和预训练权重匹配，一般不要随意修改。

  ## 4. Num workers

  当前还不能在 YAML 调整，在 kuavo_model/external_models/Xiaomi-Robotics-0/xr0/mibot/data/datamodule/base_datamodule.py:33 中写死：

  num_workers=16
  prefetch_factor=8
  persistent_workers=True
  pin_memory=True

  目前若要调整，需要修改这里。

  ## 5. 冻结策略

  当前也不是配置项。实际代码在 kuavo_model/external_models/Xiaomi-Robotics-0/xr0/mibot/models/VLA/XR0.py:441：

  self.vlm.model.get_input_embeddings().requires_grad_(False)
  self.vlm.model.visual.gradient_checkpointing_enable()

  当前训练状态：

  - 输入 embedding：冻结。
  - VLM language backbone：训练。
  - VLM visual backbone：训练。
  - DiT 和 action/state projector：训练。
  - 视觉 gradient checkpointing：开启。


## 推理
推理正常用kuavo_server/launch.py xiaomi 推理就行

> 以下是AI写的，内容全一些
# Xiaomi XR0：使用配置驱动的 LeRobot 数据后训练

本文说明如何使用 LeRobot v3 数据后训练 Xiaomi-Robotics-0。机器人状态和动作的维度、顺序及其在 XR0 张量中的位置由一份 modality JSON 决定；dataset、统计量计算和推理 server 使用同一个布局实现。

## 1. 数据流

XR0 网络保持官方形状：

```yaml
state_shape: [1, 32]
action_shape: [30, 32]
```

机器人的实际维度可以小于 32。modality JSON 指定如何把原始 LeRobot state/action 放入 32 维模型张量；没有使用的维度保持 `value=0, mask=0`。

训练流程：

```text
LeRobot absolute action
  → modality pack/delta
  → 32D mean/std normalization
  → XR0 training
```

推理流程是对应的逆过程：

```text
XR0 normalized output
  → denormalization
  → modality recover
  → LeRobot 原始动作顺序
```

## 2. Kuavo modality 配置

Kuavo 使用：

```text
[left_q1..q7, left_gripper, right_q1..q7, right_gripper]
```

配置文件位于：

```text
kuavo_data/modality_templates/kuavo_both_xr0.modality.json
```

关键结构如下：

```json
{
  "model": {
    "state_dim": 32,
    "action_dim": 32
  },
  "source": {
    "state": "observation.state",
    "action": "action"
  },
  "action_representation": "delta",
  "state": {
    "left_arm": {
      "start": 0,
      "end": 7,
      "model_start": 0
    },
    "left_gripper": {
      "start": 7,
      "end": 8,
      "model_start": 7
    }
  },
  "action": {
    "left_arm": {
      "start": 0,
      "end": 7,
      "model_start": 0,
      "state_component": "left_arm"
    }
  },
  "video": {
    "head": {
      "original_key": "observation.images.head_cam_h"
    }
  }
}
```

字段含义：

| 字段 | 含义 |
|---|---|
| `start/end` | 该 component 在 LeRobot 原始向量中的切片 |
| `model_start` | 该 component 在 XR0 张量中的起始位置 |
| `state_component` | delta action 对应的当前 state component |
| `action_representation` | `delta` 或 `absolute` |
| `original_key` | LeRobot 中的相机字段名 |

维度顺序由明确的 source slice 和 `model_start` 决定，不依赖 JSON 字段书写顺序。

Kuavo 当前将 16 个有效维度连续放入模型的 `0:16`，其余 `16:32` 为 padding。若需要调整模型内位置，只修改 modality JSON 中的 `model_start`。

## 3. 为另一台机器人准备配置

复制一份 modality JSON，然后修改：

1. `source.state/action`：LeRobot feature key；
2. `state/action` component 的原始切片；
3. 每个 component 的 `model_start`；
4. delta action 对应的 `state_component`；
5. `video` 中的相机 key。

逻辑 state/action 可以采用不同顺序，但每个 delta action component 必须对应一个同长度 state component。实际模型 state/action 维度不能超过当前 XR0 的 32。

## 4. LeRobot 数据要求

Kuavo 数据应由 `kuavo_data/CvtRosbag2Lerobot.py` 生成，并包含：

```text
observation.state                     float32[16]
action                                float32[16]
observation.images.head_cam_h         image/video
observation.images.wrist_cam_l        image/video
observation.images.wrist_cam_r        image/video
task                                  text
```

转换器写入的是绝对 action。使用 `action_representation: delta` 时，通用 loader 在训练阶段计算：

```text
delta[t, k] = action[t + k] - observation.state[t]
```

不要在数据转换阶段再次计算 delta。

## 5. 环境准备

```bash
conda activate mibot

cd /home/ruichen/projects/kdc_new/kuavo_learning_studio/kuavo_model/external_models/Xiaomi-Robotics-0/xr0
pip install -e .

# XR0 固定 transformers==4.57.1，它要求 huggingface-hub<1.0。
# 本仓库 vendored LeRobot 的包元数据会请求 huggingface-hub>=1，
# 因此这里只安装本地 LeRobot 代码，不让 pip 再次解析并升级依赖。
pip install --no-deps -e /home/ruichen/projects/kdc_new/kuavo_learning_studio/third_party/lerobot
pip install --no-deps 'huggingface-hub==0.36.0'
```

确认依赖：

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
python -c "import lerobot; print(lerobot.__file__)"
python -c "import transformers, huggingface_hub; print(transformers.__version__, huggingface_hub.__version__)"
python -c "import torchcodec; print(torchcodec.__version__)"
python -c "import lightning; print(lightning.__version__)"
python -c "import flash_attn; print(flash_attn.__version__)"
```

这里期望 Transformers/hub 版本检查输出 `4.57.1 0.36.0`。不要再执行
`pip install 'lerobot[dataset]'`，否则 pip 会按 LeRobot 0.5.2 的包元数据把
`huggingface-hub` 升到 1.x，导致 XR0 使用的 Transformers 无法导入。XR0 的
PyTorch 2.8 对应 `torchcodec==0.7.0`；当前 Kuavo AV1 数据默认通过 `pyav`
解码，避免依赖不同机器上的 TorchCodec/FFmpeg 组合。

## 6. 转换 XR0 预训练权重

HF snapshot 是后训练初始化权重，不是 server checkpoint：

```bash
cd /home/ruichen/projects/kdc_new/kuavo_learning_studio/kuavo_model/external_models/Xiaomi-Robotics-0/xr0

python tools/weight_convert.py \
  --model_path /home/ruichen/.cache/huggingface/hub/models--XiaomiRobotics--Xiaomi-Robotics-0-Pretrain/snapshots/50276c277f7d5c9b1a52fcecdea79045568c6916 \
  --output_dir pretrained_ckpt \
  --output_filename xr0_pretrained.pt
```

## 7. 计算 action mean/std

统计脚本和训练 loader 使用同一个 `ActionLayout`：

```bash
cd /home/ruichen/projects/kdc_new/kuavo_learning_studio/kuavo_model/external_models/Xiaomi-Robotics-0/xr0

python tools/compute_lerobot_stats.py \
  --root /media/ruichen/5fe8ed68-6ff6-464f-af10-a89b65c040cf/sim/sim/TASK1-ToySorting/lerobot \
  --repo_id lerobot/kuavo_task \
  --modality_config /home/ruichen/projects/kdc_new/kuavo_learning_studio/kuavo_data/modality_templates/kuavo_both_xr0.modality.json \
  --action_length 30 \
  --output /home/ruichen/xr0_outputs/kuavo_task/xr0_action_stats.json
```

脚本只读取 parquet 中的 state/action，不解码视频。输出的 `mean/std` 形状为 `(30, 32)`。只有 modality 映射的有效维度参与真实数据统计；padding 维度始终是零，最终固定为中性的 `mean=0, std=1`。训练 loader 另外生成 `(30, 32)` 的 `action_mask`，有效动作维度为 1，padding 维度和 episode 尾部补帧为 0，因此这些维度不进入训练 loss。统计文件同时记录 modality signature，训练时会拒绝顺序不匹配的统计量。

更换 modality 中的动作顺序或 `model_start` 后，必须重新计算统计量。

## 8. 数据配置

编辑外部 XR0 仓库的 `configs/data/kuavo.yaml`：

```yaml
# @package _global_

data:
  type: BaseDataModule
  params:
    type: lerobot
    max_steps: ${trainer.max_steps}
    processor_path: ${model.params.model.vlm_path}

    train_datasets:
      repo_id: lerobot/kuavo_task
      root: /media/ruichen/5fe8ed68-6ff6-464f-af10-a89b65c040cf/sim/sim/TASK1-ToySorting/lerobot
      fps: 10
      batch_size: 16
      action_length: 30
      video_backend: pyav
      modality_config_path: /home/ruichen/projects/kdc_new/kuavo_learning_studio/kuavo_data/modality_templates/kuavo_both_xr0.modality.json
      stats_path: /home/ruichen/xr0_outputs/kuavo_task/xr0_action_stats.json
```

训练入口会读取 modality 和 stats，并将两者完整写入最终 `config.py`。推理阶段不需要再次提供 modality JSON 或 stats 文件。

## 9. 启动后训练

```bash
cd /home/ruichen/projects/kdc_new/kuavo_learning_studio/kuavo_model/external_models/Xiaomi-Robotics-0/xr0

CUDA_VISIBLE_DEVICES=0 RESOURCE_GPU=1 \
bash scripts/train.sh \
  data=kuavo \
  model=XR0 \
  trainer.project=kuavo \
  trainer.exp_name=xr0_kuavo_sync \
  trainer.default_root_dir=/home/ruichen/xr0_outputs \
  trainer.max_steps=1 \
  model.params.pretrained=/home/ruichen/projects/kdc_new/kuavo_learning_studio/kuavo_model/external_models/Xiaomi-Robotics-0/xr0/pretrained_ckpt/xr0_pretrained.pt \
  model.params.model.vlm_path=/home/ruichen/Qwen3-VL-4B-Instruct
```

预训练参数路径是 `model.params.pretrained`。

当前 Kuavo server 没有发送 action prefix，建议先使用同步训练：

```yaml
model.params.model.async_train: false
```

## 10. 训练产物

`config.py` 由训练入口自动生成，不需要手写：

```text
/home/ruichen/xr0_outputs/project_kuavo/xr0_kuavo_sync/
├── config.yaml
├── config.py
└── epoch=0-step=10000.ckpt/  # 也可能名为 last.ckpt
    └── checkpoint/
        └── mp_rank_00_model_states.pt
```

`checkpoint` 可以指向整个 `xr0_kuavo_sync` 实验目录，也可以直接指向其中具体的 `.ckpt` 目录。实验目录存在 `last.ckpt` 时优先加载它；否则加载 step 最大的 `.ckpt`。HF snapshot 或单独的 `.pt` 文件不能作为后训练 server checkpoint。

## 11. 启动推理 server

修改本仓库的 `configs/server/launcher.yaml`：

```yaml
xiaomi:
  adapter: xiaomi_robotics_0
  env:
    mode: conda
    conda_env: mibot
  args:
    checkpoint: /home/ruichen/xr0_outputs/project_kuavo/xr0_kuavo_sync
    processor_path: /home/ruichen/Qwen3-VL-4B-Instruct
    execution_horizon: 10
    device: cuda:0
    model_repo_root: ""  # 留空时使用 kuavo_model/external_models/Xiaomi-Robotics-0
```

启动：

```bash
cd /home/ruichen/projects/kdc_new/kuavo_learning_studio
python kuavo_server/launch.py xiaomi --dry-run
python kuavo_server/launch.py xiaomi
```

server 从 checkpoint 的 `config.py` 读取 modality，按训练时相同的布局打包 state、反归一化 action 并恢复机器人原始动作顺序。

## 12. 正确性检查

仓库测试覆盖：

- `pack_state` 的维度顺序；
- `pack_action` 和 `recover_action` round-trip；
- temporal/action padding mask；
- normalization 和 denormalization round-trip；
- server 输入 state 维度。

运行：

```bash
cd /home/ruichen/projects/kdc_new/kuavo_learning_studio
conda run -n kdc_new python -m pytest -q \
  tests/test_xiaomi_action_layout.py \
  tests/test_xiaomi_robotics_0_adapter.py
```

训练前还应从实际数据取一个样本，确认：

```text
state.shape       == [1, 32]
action.shape      == [30, 32]
action_mask.shape == [30, 32]
recover(pack(action, state), state) == original action
```

## 13. 常见错误

### 修改 modality 后继续使用旧 stats

动作位置发生变化后，旧 mean/std 的维度语义不再匹配。重新运行 `compute_lerobot_stats.py`。

### state/action shape 与 modality 不一致

检查 `source.state/action` 对应的 LeRobot feature，以及各 component 的 `start/end`。

### 缺少 `config.py`

`--checkpoint` 错误地指向了 HF 预训练 snapshot。它必须指向完整后训练实验目录。

### 加载官方 EEF checkpoint

当前配置驱动 server 需要 checkpoint 的 `config.py` 中包含训练时内嵌的 `modality`。官方 EEF/6-DoF checkpoint 应继续使用官方 client，不应按自定义 LeRobot 布局恢复动作。
