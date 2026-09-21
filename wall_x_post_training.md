# wallx使用教程
```
cd kuavo_model/external_models/wall-x
conda create -n wallx python=3.10
conda activate wallx

pip install -r requirements.txt
pip install "dmuon @ git+https://github.com/X-Square-Robot/dmuon.git"
```
# Wall-X 当前需配合 LeRobot v0.4.x；Python 3.10 环境使用 v0.4.4。
# 请把 LeRobot clone 到 kuavo_learning_studio 主仓库之外，避免嵌套 Git 仓库。
```
git clone --branch v0.4.4 --depth 1 \
  https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e .
```

# LeRobot 0.4.4 会把 huggingface-hub 降到 0.35.x，但 wall-x 的
# transformers 5.2.0 要求 huggingface-hub>=1.3,<2，安装后必须恢复。
```
python -m pip install --upgrade "huggingface-hub>=1.3,<2"

cd ..
export FLASH_ATTN_CUDA_ARCHS=$(python -c 'import torch; print(f"{torch.cuda.get_device_capability()[0]}{torch.cuda.get_device_capability()[1]}")')
MAX_JOBS=4 pip install flash-attn==2.8.3 --no-build-isolation
MAX_JOBS=8 pip install --no-build-isolation -e .
```

## 配置文件
修改`LeTools-Learning/kuavo_server/configs/wall_x/kuavo.yml`
然后算stats
```
python scripts/compute_norm_stats.py \
  --train_config /mnt/huangruichen/LeTools-Learning/kuavo_server/configs/wall_x/kuavo.yml \
  --data_root /mnt/huangruichen/data/sim_task1/lerobot \
  --output_path outputs/norm_stats.json
```

## 训练
有一点要注意，比如num_training_steps是100，save_step是100，那么就不会保存，因为train的index从0开始，实际只会训到99，所以如果想保存最后step的模型，train step要是save的倍数+1
```
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 \
  wall_x/trainer/fsdp_trainer/train_fsdp.py \
  --config /mnt/huangruichen/LeTools-Learning/kuavo_server/configs/wall_x/kuavo.yml
```


> 下面是AI写的
# Wall-X 在 Kuavo LeRobot v3 数据上的后训练与推理

## 1. 数据约定

默认模板适配 `configs/data/KuavoRosbag2Lerobot.yaml` 中：

```yaml
dataset:
  which_arm: both
  eef_type: leju_claw  # qiangnao/rq2f85 也只保留 1 个开合自由度
  dex_dof_needed: 1
```

该设置产出的 LeRobot v3 关键字段为：

| 字段 | 形状/含义 |
| --- | --- |
| `observation.state` | 16：左臂 7、左夹爪 1、右臂 7、右夹爪 1 |
| `action` | 16，顺序同 state |
| `observation.images.head_cam_h` | 头部 RGB |
| `observation.images.wrist_cam_l` | 左腕 RGB |
| `observation.images.wrist_cam_r` | 右腕 RGB |
| `task_index` | 指向 LeRobot v3 task 元数据中的自然语言任务 |

wall-x 的 prompt 在训练时来自 LeRobot task 元数据；在线推理时来自
`configs/deploy/deploy.yaml` 的 `inference.task_prompt`。两者应使用相同语言和相近措辞。

如果你的数据是单臂 8 维，复制训练模板并删除另一只手臂的 state/action/camera
条目，同时令训练与服务的 `which_arm` 都为 `left` 或 `right`。不要使用双臂模板
训练单臂数据，也不要只靠推理时切片掩盖训练布局不一致。

## 2. 安装 wall-x 环境

以下命令在 wall-x checkout 中执行：

```bash
cd /home/ruichen/projects/kdc_new/kuavo_learning_studio/kuavo_model/external_models/wall-x
conda create -n wallx python=3.10
conda activate wallx

pip install -r requirements.txt
pip install "dmuon @ git+https://github.com/X-Square-Robot/dmuon.git"

# Wall-X 当前需配合 LeRobot v0.4.x；Python 3.10 环境使用 v0.4.4。
# 请把 LeRobot clone 到 kuavo_learning_studio 主仓库之外，避免嵌套 Git 仓库。
git clone --branch v0.4.4 --depth 1 \
  https://github.com/huggingface/lerobot.git \
  /home/ruichen/projects/kdc_new/external_model/lerobot-v0.4.4
cd /home/ruichen/projects/kdc_new/external_model/lerobot-v0.4.4
pip install -e .

# LeRobot 0.4.4 会把 huggingface-hub 降到 0.35.x，但 wall-x 的
# transformers 5.2.0 要求 huggingface-hub>=1.3,<2，安装后必须恢复。
python -m pip install --upgrade "huggingface-hub>=1.3,<2"

cd /home/ruichen/projects/kdc_new/kuavo_learning_studio/kuavo_model/external_models/wall-x
export FLASH_ATTN_CUDA_ARCHS=$(python -c 'import torch; print(f"{torch.cuda.get_device_capability()[0]}{torch.cuda.get_device_capability()[1]}")')
MAX_JOBS=4 pip install flash-attn==2.8.3 --no-build-isolation
MAX_JOBS=8 pip install --no-build-isolation -e .
```

wall-x 安装会编译 CUDA 扩展。若目标机器没有可用 CUDA toolchain，安装和真实模型
推理都无法完成；adapter 的静态测试不需要加载这些扩展。

下载 Wall-OSS-0.5 基座和 Qwen processor：

```bash
huggingface-cli download X-Square-Robot/wall-oss-0.5 \
  --local-dir /path/to/wall-oss-0.5
huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct \
  --local-dir /path/to/Qwen2.5-VL-3B-Instruct
```

## 3. 检查 Kuavo LeRobot v3 数据

数据集根目录至少应包含 `meta/info.json`、`meta/tasks.parquet`、`data/` 和
`videos/`。先检查 feature 名称及维度：

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path("/path/to/kuavo_lerobot_v3")
info = json.loads((root / "meta" / "info.json").read_text())
for key in (
    "observation.state",
    "action",
    "observation.images.head_cam_h",
    "observation.images.wrist_cam_l",
    "observation.images.wrist_cam_r",
):
    print(key, info["features"][key])
PY
```

双臂模板要求 state/action 都是 16 维。还需确认数据转换配置中的
`dataset.delta_action: false`：当前服务把模型输出作为绝对关节目标直接发给 Kuavo。
若用 delta action 数据训练，必须另行增加“delta + 当前 state -> absolute action”的
adapter 后处理，不能直接使用本指南的配置。

## 4. 准备训练配置和归一化统计

复制模板到实验目录，避免后续实验相互覆盖：

```bash
cp /home/ruichen/projects/kdc_new/kuavo_learning_studio/kuavo_server/configs/wall_x/kuavo.yml \
  /path/to/experiments/kuavo_wallx.yml
```

必须替换配置中的所有 `/path/to/...`：

- `model.config_path`：`wall-oss-0.5/config.json`
- `model.processor_path` 和 `model.pretrained_path`：Qwen2.5-VL processor 目录
- `data.lerobot_config.repo_id`：本地 LeRobot v3 数据集根目录
- `data.norm_stats_path`：即将生成的 JSON 路径
- `checkpoint.save_path`：训练输出目录
- `checkpoint.resume_from`：基座的 `model.safetensors`

检查是否还有占位符：

```bash
rg -n '/path/to' /path/to/experiments/kuavo_wallx.yml
```

然后用 wall-x 原生脚本计算统计量：

```bash
cd /home/ruichen/projects/kdc_new/kuavo_learning_studio/kuavo_model/external_models/wall-x
conda activate wallx
python scripts/compute_norm_stats.py \
  --train_config /path/to/experiments/kuavo_wallx.yml \
  --data_root /path/to/kuavo_lerobot_v3 \
  --output_path /path/to/kuavo_lerobot_v3_norm_stats.json
```

不要复用其他机器人、其他手臂配置或 delta/absolute 表示不同的数据统计量。

## 5. 后训练
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 wall_x/trainer/fsdp_trainer/train_fsdp.py --config /home/ruichen/projects/kdc_new/kuavo_learning_studio/kuavo_server/configs/wall_x/kuavo.yml

多卡：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
  wall_x/trainer/fsdp_trainer/train_fsdp.py \
  --config /path/to/experiments/kuavo_wallx.yml
```

恢复训练时，把 `checkpoint.resume_from` 改为已有 checkpoint **目录**。若训练产出
FSDP shard，推理前先合并：

```bash
python scripts/merge_sharded_weights.py \
  /path/to/sharded_checkpoint \
  /path/to/wallx_kuavo_merged_checkpoint
```

推理 checkpoint 目录应包含模型权重和 normalization 数据。若合并结果没有
`normalizer_action.pth`、`normalizer_propri.pth` 或 `norm_stats.json`，复制本次数据统计：

```bash
cp /path/to/kuavo_lerobot_v3_norm_stats.json \
  /path/to/wallx_kuavo_merged_checkpoint/norm_stats.json
```

建议同时保留训练生成的 `config.yml` 和 processor 文件。也可以像下文一样通过
`--train_config_path` 显式指定原训练 YAML。`--norm_key kuavo` 是在线 inference batch
使用的 normalization key；checkpoint-local `norm_stats.json` 会以该 key 注册。若使用包含
多个 dataset key 的 normalizer `.pth`，必须传入其中真实存在的 key。

## 6. 模型侧验证与服务启动

先在 wall-x 环境做原生 smoke test：

```bash
cd /home/ruichen/projects/kdc_new/kuavo_learning_studio/kuavo_model/external_models/wall-x
conda activate wallx
python scripts/fake_inference.py \
  --checkpoint-path /path/to/wallx_kuavo_merged_checkpoint \
  --train-config-path /path/to/experiments/kuavo_wallx.yml
```

再验证 Kuavo launcher：

```bash
cd /home/ruichen/projects/kdc_new/kuavo_learning_studio
python kuavo_server/launch.py --list
python kuavo_server/launch.py wallx --dry-run \
  --checkpoint /path/to/wallx_kuavo_merged_checkpoint \
  --train_config_path /path/to/experiments/kuavo_wallx.yml
```

启动服务：

```bash
python kuavo_server/launch.py wallx \
  --checkpoint /path/to/wallx_kuavo_merged_checkpoint \
  --train_config_path /path/to/experiments/kuavo_wallx.yml \
  --action_horizon 32 \
  --execution_horizon 8 \
  --device cuda:0 \
  --norm_key kuavo
```

`action_horizon` 必须与训练配置一致。`execution_horizon` 是每次预测后实际暴露/缓存的
前 N 步，设为 `0` 表示使用完整 chunk。episode reset 会清空尚未执行的旧 chunk。

## 7. Kuavo 机器人侧推理

编辑 `configs/deploy/deploy.yaml`：

```yaml
env:
  which_arm: both
  eef_type: leju_claw

inference:
  policy_type: client
  task_prompt: "与训练数据 task 文本一致的指令"
```

在机器人/ROS 环境运行：

```bash
cd /home/ruichen/projects/kdc_new/kuavo_learning_studio
python kuavo_deploy/eval.py
```

模型环境与机器人环境可以不同，但 client 配置的 server 地址和端口必须指向 wall-x
服务。默认服务端口为 `5555`。首次真机执行前先在仿真或离线回放中检查：

1. server metadata 显示 `state_dim=16`、`action_dim=16`；
2. 返回动作形状为双臂 16 维；
3. 夹爪输出范围与数据转换时的 `[0, 1]` 定义一致；
4. prompt 与训练 task 文本一致；
5. reset 后没有上一 episode 的剩余动作。

## 8. 常见错误

- `No config.yml/config.yaml found`：传入正确的 `--train_config_path`。
- `expects 16 state dims`：训练 YAML、`which_arm` 与数据集维度不一致。
- `Cannot map Wall-X camera names`：`data.key_mappings.camera` 未使用标准 Kuavo 键。
- normalizer key/shape 错误：重新用当前数据和当前 YAML 运行 `compute_norm_stats.py`。
- 找不到 wall-x：确认仓库内 `kuavo_model/external_models/wall-x` 完整，或显式传
  `--model_repo_root` 覆盖默认路径。
- OOM：降低 `batch_size_per_gpu`、增加 `gradient_accumulation_steps`，或使用多卡 FSDP。
