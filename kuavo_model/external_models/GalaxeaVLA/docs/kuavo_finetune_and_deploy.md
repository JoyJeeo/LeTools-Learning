# Kuavo G0.5 fine-tuning and KLS deployment

For a field-by-field explanation of the Kuavo task config, see
[`kuavo_task_config_reference.md`](kuavo_task_config_reference.md).

GalaxeaVLA is vendored under
`kuavo_model/external_models/GalaxeaVLA` without nested Git metadata. The KLS
launcher and G0.5 adapter use this directory as `--model_repo_root`.

## 1. Data contract

`configs/data/kuavo.yaml` matches `kuavo_both.modality.json`:

- state/action: `[left_arm(7), left_gripper(1), right_arm(7), right_gripper(1)]`;
- images: `observation.images.head_cam_h`, `wrist_cam_l`, `wrist_cam_r`;
- language: LeRobot `task_index`, decoded through V3 `meta/tasks.parquet`;
- arm action targets are converted from absolute joint positions to deltas from
  the current state. Grippers remain absolute. Inference applies the exact inverse.

The config expects LeRobot v3.0 and a flat 16D `observation.state`/`action` column.
Camera resolution in `raw_shape` is descriptive for this loader; images are resized
by the configured transforms. Verify that all three camera keys exist and that the
stored channel order is the normal LeRobot RGB format.

## 2. Environment and checkpoints

```bash
cd /home/ruichen/projects/kdc_new/kuavo_learning_studio/kuavo_model/external_models/GalaxeaVLA
export UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
uv sync --extra dev --index-strategy unsafe-best-match
source .venv/bin/activate

export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download OpenGalaxea/G05 --repo-type model --local-dir checkpoints
```

Expected base assets are:

```text
checkpoints/action_tokenizer.pt
checkpoints/qwen3_5_2b_base_processor/
checkpoints/g05-base/checkpoints/model_state_dict.pt
```

## 3. Point the config at Kuavo data

Edit the `dataset_dirs` list at the bottom of `configs/data/kuavo.yaml`. Every
directory must be a complete LeRobot v3 dataset root. Then inspect its metadata:

```bash
DATA=/absolute/path/to/kuavo_lerobot_dataset
python - "$DATA" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
info = json.loads((p / "meta/info.json").read_text())
features = info["features"]
for key in ["observation.state", "action", "observation.images.head_cam_h",
            "observation.images.wrist_cam_l", "observation.images.wrist_cam_r"]:
    print(key, features[key])
assert features["observation.state"]["shape"] == [16]
assert features["action"]["shape"] == [16]
PY
```

Use the task table for meaningful instructions; do not train with a single empty
or placeholder task string.

## 4. Validate and fine-tune

```bash
export G05_OUTPUT_DIR=$PWD/outputs/kuavo

# Resolve configuration only.
python tools/resolve_config.py kuavo --key data.embodiment_datasets.kuavo

# One-step pipeline smoke test. This loads data and computes Kuavo statistics.
bash scripts/run/finetune.sh 1 kuavo --dry-run --max_datasets 1

# Short GPU test before a long run.
bash scripts/run/finetune.sh 1 kuavo --test --max_datasets 1 model.max_steps=10

# Full run.
EXP_NAME=kuavo_g05_base bash scripts/run/finetune.sh 1 kuavo
```

The checked-in Kuavo task uses BF16, activation checkpointing, batch size 1,
gradient accumulation 8, and an 8-bit optimizer as a practical 24GB starting
point. The upstream full-fine-tuning estimate is over 70GB at its standard
settings, so 24GB is not guaranteed. If the smoke test still OOMs, first reduce
image count/resolution or use a larger-memory GPU; do not silently change the
state/action semantics. Multi-GPU DDP improves throughput but does not shard one
model replica and therefore does not lower per-GPU memory.

Each successful run writes `.hydra/config.yaml`, `dataset_stats.json`, copied
tokenizer/processor assets, and model checkpoints under
`$G05_OUTPUT_DIR/kuavo/<EXP_NAME>/`. Keep these sidecars together for serving.

## 5. Start through kuavo_server

Edit the `g05` checkpoint in KLS `configs/server/launcher.yaml`, or override it:

```bash
cd /home/ruichen/projects/kdc_new/kuavo_learning_studio
python kuavo_server/launch.py g05 --dry-run \
  --checkpoint /absolute/path/to/model_state_dict.pt

python kuavo_server/launch.py g05 \
  --checkpoint /absolute/path/to/model_state_dict.pt
```

The adapter runs inside this repository's `uv` environment, converts KLS HWC/CHW
images to G0.5 CHW uint8, splits the 16D state, calls G0.5's native processor and
`PolicyInferencer`, then packs the denormalized absolute output back into the same
16D Kuavo order. `execution_horizon` controls how much of the 32-step model chunk
is returned, and `reset()` clears cached steps.

On the robot side, keep `policy_type: client` and use the existing KLS deployment
flow. Start with simulation or offline bag replay, check joint order/units and
gripper range, then enable real hardware with conservative limits and an emergency
stop. Code integration and a successful server boot are not a real-robot validation.