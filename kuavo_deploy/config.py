"""
Single merged configuration loader for Kuavo deploy.

Default behavior:
- load ./configs/deploy/deploy.yaml
- if a matching total config exists at ./configs/deploy/total/deploy_total.yaml,
  merge it first and let deploy.yaml override it

This mirrors the train-side `total + simple override` pattern.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple, Any, Dict
import os
from pathlib import Path
from copy import deepcopy
import yaml

DEFAULT_DEPLOY_CONFIG = "deploy.yaml"


def get_arm_joint_slice(platform_type: str) -> Tuple[int, int]:
    config_file = (
        Path(__file__).resolve().parent.parent / "configs" / "platform" / "platform_config.yaml"
    )
    with open(config_file, "r", encoding="utf-8") as f:
        platform_cfg = yaml.safe_load(f) or {}
    platforms = platform_cfg.get("platforms", {})
    platform = platforms.get(platform_type.lower())
    if not platform:
        raise ValueError(
            f"Unsupported platform type: {platform_type}. Supported: {list(platforms.keys())}"
        )
    return (platform["arm_joint_start"], platform["arm_joint_end"])


def get_lower_body_joint_slice(platform_type: str) -> Tuple[int, int]:
    config_file = (
        Path(__file__).resolve().parent.parent / "configs" / "platform" / "platform_config.yaml"
    )
    with open(config_file, "r", encoding="utf-8") as f:
        platform_cfg = yaml.safe_load(f) or {}
    platform = platform_cfg.get("platforms", {}).get(platform_type.lower())
    if not platform:
        raise ValueError(f"Unsupported platform type: {platform_type}")
    if "lower_body_joint_start" not in platform or "lower_body_joint_end" not in platform:
        raise ValueError(f"Platform {platform_type!r} does not define lower-body joints")
    return (platform["lower_body_joint_start"], platform["lower_body_joint_end"])

@dataclass
class Range:
    min: List[float]
    max: List[float]

@dataclass
class LimitsConfig:
    joint_q: Range = field(default_factory=lambda: Range([-3.14]*14, [3.14]*14))
    gripper: Range = field(default_factory=lambda: Range([0, 0], [1, 1]))
    lower_body: Range = field(default_factory=lambda: Range([-3.14]*4, [3.14]*4))

# -----------------------
# Environment Dataclass
# -----------------------
@dataclass
class ConfigEnv:
    inference_env: str = "real"  # "sim" or "real"
    env_name: str = "Kuavo-Sim"
    real: bool = False
    eef_type: str = "rq2f85"
    platform_type: str = "4pro"
    enable_5w_wholebody: bool = False
    enable_5w_base_move: bool = False
    lowerlock_5w: List[int] = field(default_factory=lambda: [0, 0, 0, 0])
    control_mode: str = "joint"
    which_arm: str = "both"
    head_init: Optional[List[float]] = field(default_factory=lambda: [0.0, 0.0])
    ros_rate: int = 10
    control_rate: int = 100  # WBC/插值后的实际控制指令频率
    enable_action_interpolation: bool = True
    direct_to_wbc: bool = False  # 是否直接将动作发送到WBC
    image_size: List[int] = field(default_factory=lambda: [640, 480])
    depth_range: List[int] = field(default_factory=lambda: [0, 1500])
    obs_key_map: Dict[str, List[Any]] = field(default_factory=dict)
    arm_state_keys: List[str]=field(default_factory=list)
    ratio: float = 0.5
    frame_alignment: bool = True
    qiangnao_dof_needed: int = 1
    sg100_dof_needed: int = 1
    sg100_dof_offset: int = 0
    sg100_open_pose: List[float] = field(default_factory=lambda: [0.0] * 22)
    sg100_close_pose: List[float] = field(default_factory=lambda: [1.57] * 22)
    sg100_init_pose: List[float] = field(default_factory=lambda: [0.0] * 22)
    sg100_force_threshold: float = 0.5
    sg100_force_kp: List[float] = field(default_factory=lambda: [0.5] * 22)
    sg100_force_kd: List[float] = field(default_factory=lambda: [0.05] * 22)
    sg100_force_torque_ff: List[float] = field(default_factory=lambda: [0.0] * 22)
    sg100_force_output_limit: List[float] = field(default_factory=lambda: [1.0] * 22)
    sg100_force_velocities: List[float] = field(default_factory=lambda: [0.0] * 22)
    is_binary_sg100_action: bool = False
    sg100_action_threshold: float = 0.5

    limits: LimitsConfig = field(default_factory=LimitsConfig)
    is_binary: bool = False
    camera_encoding: str = "jpeg"  # "jpeg" or "h265"

    # -------- Validation ----------
    def validate(self):
        if (not isinstance(self.lowerlock_5w, list)
                or len(self.lowerlock_5w) != 4
                or any(type(value) is not int or value not in (0, 1)
                       for value in self.lowerlock_5w)):
            raise ValueError("env.5w_lowerlock must be a four-element array of 0 or 1")
        if self.inference_env not in ["sim", "real"]:
            raise ValueError("env.inference_env must be 'sim' or 'real'")
        if self.eef_type not in ["rq2f85", "leju_claw", "qiangnao", "sg100"]:
            raise ValueError(f"Invalid eef_type: {self.eef_type}. Valid: rq2f85, leju_claw, qiangnao, sg100")
        if self.platform_type not in ["4pro", "5w", "5"]:
            raise ValueError(f"Invalid platform_type: {self.platform_type}. Valid: 4pro, 5w, 5")
        if self.which_arm not in ["left", "right", "both"]:
            raise ValueError(f"Invalid which_arm: {self.which_arm}. Valid: left, right, both")
        if not isinstance(self.image_size, list) or len(self.image_size) != 2:
            raise ValueError("image_size must be a list [height, width]")
        if self.camera_encoding not in ["jpeg", "h265"]:
            raise ValueError(f"Invalid camera_encoding: {self.camera_encoding}. Valid: jpeg, h265")
        # Derive dimensions from the platform/config instead of a packed 16D assumption.
        arm_start, arm_end = get_arm_joint_slice(self.platform_type)
        arm_dof = arm_end - arm_start
        if len(self.limits["joint_q"]["max"]) != len(self.limits["joint_q"]["min"]):
            raise ValueError("joint_q min/max must have equal lengths")
        if len(self.limits["joint_q"]["min"]) != arm_dof:
            raise ValueError(f"Robot joint_q limits must contain {arm_dof} arm joints")
        if len(self.limits["gripper"]["max"]) != len(self.limits["gripper"]["min"]):
            raise ValueError("gripper min/max must have equal lengths")
        if len(self.limits["gripper"]["min"]) % 2:
            raise ValueError("gripper limits must contain equal left/right dimensions")
        if self.use_5w_wholebody:
            lower_start, lower_end = get_lower_body_joint_slice(self.platform_type)
            lower_dof = lower_end - lower_start
            lower_limits = self.limits["lower_body"]
            if not (len(lower_limits["min"]) == len(lower_limits["max"]) == lower_dof):
                raise ValueError(f"lower_body min/max must each contain {lower_dof} values")
        if self.qiangnao_dof_needed != 1: # not in [1, 7]:
            raise ValueError("qiangnao_dof_needed must be 1 now!")
            # raise ValueError("qiangnao_dof_needed must be either 1 or 7")
        if self.sg100_dof_needed not in (1, 11):
            raise ValueError("sg100_dof_needed must be 1 or 11")
        if not isinstance(self.sg100_dof_offset, int) or self.sg100_dof_offset < 0:
            raise ValueError("sg100_dof_offset must be a non-negative integer")
        if self.sg100_dof_offset + self.sg100_dof_needed > 11:
            raise ValueError("sg100_dof_offset + sg100_dof_needed must be <= 11")
        for name in (
            "sg100_open_pose", "sg100_close_pose", "sg100_init_pose",
            "sg100_force_kp", "sg100_force_kd", "sg100_force_torque_ff",
            "sg100_force_output_limit", "sg100_force_velocities",
        ):
            if len(getattr(self, name)) != 22:
                raise ValueError(f"{name} must contain 22 values (11 left + 11 right)")
        if not 0.0 <= self.sg100_force_threshold <= 1.0:
            raise ValueError("sg100_force_threshold must be in [0, 1]")
        if not 0.0 <= self.sg100_action_threshold <= 1.0:
            raise ValueError("sg100_action_threshold must be in [0, 1]")

    # -------- Derived properties ----------
    @property
    def use_5w_wholebody(self) -> bool:
        return self.platform_type.lower() == "5w" and self.enable_5w_wholebody

    @property
    def use_5w_base_move(self) -> bool:
        return self.platform_type.lower() == "5w" and self.enable_5w_base_move

    @property
    def arm_dof_per_side(self) -> int:
        arm_start, arm_end = get_arm_joint_slice(self.platform_type)
        arm_dof = arm_end - arm_start
        if arm_dof % 2:
            raise ValueError("The configured arm joint range must split evenly between both arms")
        return arm_dof // 2

    @property
    def eef_dof_per_side(self) -> int:
        if self.eef_type == "sg100":
            return self.sg100_dof_needed
        return len(self.limits["gripper"]["min"]) // 2

    @property
    def joint_q_slice(self):
        arm_start, arm_end = get_arm_joint_slice(self.platform_type)
        left_end = arm_start + self.arm_dof_per_side
        right_start = left_end
        return {
            "left": [[arm_start, left_end]],
            "right": [[right_start, arm_end]],
            "both": [[arm_start, left_end], [right_start, arm_end]]
        }[self.which_arm]

    @property
    def gripper_slice(self):
        if self.eef_type == "rq2f85" or self.eef_type == "leju_claw":
            dof = self.eef_dof_per_side
            return {
                "left": [[0, dof]],
                "right": [[dof, dof * 2]],
                "both": [[0, dof], [dof, dof * 2]]
            }[self.which_arm]
        elif self.eef_type == "qiangnao" and self.qiangnao_dof_needed == 1:
            return {
                "left": [[0, 1]],
                "right": [[6, 7]],
                "both": [[0, 1], [6, 7]]
            }[self.which_arm]
        elif self.eef_type == "sg100":
            offset = self.sg100_dof_offset
            count = self.sg100_dof_needed
            return {
                "left": [[offset, offset + count]],
                "right": [[11 + offset, 11 + offset + count]],
                "both": [[offset, offset + count], [11 + offset, 11 + offset + count]],
            }[self.which_arm]
        else:
            raise ValueError("Unsupported eef_type or dof config")

    # ---------------- obs_key_map build ----------------
    _H265_TOPIC_MAP = {
        "head_cam_h": "/cam_h/color/h265_stream",
        "wrist_cam_l": "/cam_l/color/h265_stream",
        "wrist_cam_r": "/cam_r/color/h265_stream",
    }

    def build_obs_key_map(self) -> Dict[str, Any]:
        obs_map = {}
        for key, info in self.obs_key_map.items():
            base = {
                "topic": info[0],
                "msg_type": info[1],
                "frequency": info[2],
                "handle": {"params": {}}
            }
            # 统一规则化参数处理
            if len(info) == 4 and isinstance(info[3], list):
                base["handle"]["params"]["resize_wh"] = info[3]
            if len(info) == 5 and isinstance(info[3], list) and isinstance(info[4], list):
                base["handle"]["params"]["resize_wh"] = info[3]
                base["handle"]["params"]["depth_range"] = info[4]

            # 特殊键处理
            if key == "joint_q":
                base["handle"]["params"]["slice"] = self.joint_q_slice
            if key in ["rq2f85", "qiangnao", "leju_claw", "sg100"]:
                base["handle"]["params"]["slice"] = self.gripper_slice
                obs_map["gripper"] = base
                continue
            if key == "eef_pose" and len(info) >= 3 and info[0] == "computed":
                obs_map["eef_pose"] = {
                    "type": "computed",
                    "source": info[1],
                    "frequency": info[2]
                }
                continue
            obs_map[key] = base

        if self.camera_encoding == "h265":
            for key, obs_info in obs_map.items():
                if key in self._H265_TOPIC_MAP:
                    obs_info["topic"] = self._H265_TOPIC_MAP[key]
                    obs_info["h265"] = True

        if self.use_5w_wholebody:
            joint_info = obs_map.get("joint_q")
            if joint_info is None:
                raise ValueError("5w_wholebody requires joint_q in env.obs_key_map")
            lower_start, lower_end = get_lower_body_joint_slice(self.platform_type)
            lower_info = deepcopy(joint_info)
            lower_info["handle"]["params"]["slice"] = [[lower_start, lower_end]]
            obs_map["lower_body"] = lower_info
        return obs_map



# -----------------------
# Inference Dataclass
# -----------------------
@dataclass
class ConfigInference:
    go_bag_path: str = ""
    policy_type: str = "diffusion"  # 支持 diffusion, act 等
    pretrained_path: str = ""
    eval_episodes: int = 1
    seed: int = 42
    start_seed: int = 42
    device: str = "cuda"  # or "cpu"
    task: str = ""
    method: str = ""
    timestamp: str = ""
    epoch: int = 1
    max_episode_steps: int = 1000
    task_prompt: str = ""

    async_inference: bool = False
    async_control_hz: float = 0.0
    async_buffer_size: int = 32
    async_low_watermark: int = 4
    async_warmup_actions: int = 1
    async_action_timeout: float = 1.0

    # RTC-Lite (deployment-side queue merging). All fields are no-ops when
    # rtc_lite_enabled is false or async_inference is false.
    rtc_lite_enabled: bool = False
    rtc_lite_merge_mode: str = "blend_replace"
    rtc_lite_overlap_steps: int = 4
    rtc_lite_freeze_steps: int = 1
    rtc_lite_ramp: str = "cosine"
    rtc_lite_delay_mode: str = "measured"
    rtc_lite_max_delay_steps: int = 8
    rtc_lite_keep_min_actions: int = 1
    rtc_lite_log_deltas: bool = True

    # Full RTC (model-space continuity guidance). This is intentionally
    # separate from RTC-Lite, which blends already decoded robot actions.
    rtc_full_enabled: bool = False
    # Explicit RTC implementation. Choose vjp or inpainting.
    rtc_full_mode: str = "vjp"
    # Shared: RTC continuity target length. Inpainting uses it as the
    # old-action initialization length; VJP uses it as the guidance horizon.
    rtc_full_overlap_steps: int = 8
    # VJP-only controls.
    rtc_full_prefix_attention_schedule: str = "exp"
    rtc_full_max_guidance_weight: float = 5.0
    # Inpainting-only controls.
    rtc_full_frozen_steps: int = 2
    rtc_full_ramp_rate: float = 5.0
    # Async queue delay compensation and diagnostics.
    rtc_full_max_delay_steps: int = 8
    rtc_full_debug: bool = False

    def validate(self):
        supported_policy_types = [
            "",
            "client",
            "act",
            "diffusion",
            "pi0",
            "pi0_fast",
            "pi05",
            "groot",
            "smolvla",
            "xvla",
            "wall_x",
            "multi_task_dit",
        ]
        if self.policy_type not in supported_policy_types:
            raise ValueError(f"Unsupported policy_type '{self.policy_type}'")
        if self.device not in ["cuda", "cpu"]:
            raise ValueError("device must be 'cuda' or 'cpu'")

        # RTC-Lite validation. Keep the hard rules narrow; the plan flags the
        # freeze/overlap/keep_min relation as a *recommendation*, not a failure.
        supported_merge_modes = {"blend_replace"}
        if self.rtc_lite_merge_mode not in supported_merge_modes:
            raise ValueError(
                f"Unsupported rtc_lite_merge_mode '{self.rtc_lite_merge_mode}'. "
                f"Valid: {sorted(supported_merge_modes)}"
            )
        supported_ramps = {"linear", "cosine"}
        if self.rtc_lite_ramp not in supported_ramps:
            raise ValueError(
                f"Unsupported rtc_lite_ramp '{self.rtc_lite_ramp}'. Valid: {sorted(supported_ramps)}"
            )
        supported_delay_modes = {"measured"}
        if self.rtc_lite_delay_mode not in supported_delay_modes:
            raise ValueError(
                f"Unsupported rtc_lite_delay_mode '{self.rtc_lite_delay_mode}'. "
                f"Valid: {sorted(supported_delay_modes)}"
            )
        if self.rtc_lite_overlap_steps < 0:
            raise ValueError("rtc_lite_overlap_steps must be >= 0")
        if self.rtc_lite_freeze_steps < 0:
            raise ValueError("rtc_lite_freeze_steps must be >= 0")
        if self.rtc_lite_keep_min_actions < 0:
            raise ValueError("rtc_lite_keep_min_actions must be >= 0")
        if self.rtc_lite_enabled and self.rtc_full_enabled:
            raise ValueError("rtc_lite_enabled and rtc_full_enabled are mutually exclusive")
        if self.rtc_full_mode not in {"vjp", "inpainting"}:
            raise ValueError("rtc_full_mode must be vjp or inpainting")
        if self.rtc_full_overlap_steps < 0:
            raise ValueError("rtc_full_overlap_steps must be >= 0")
        if self.rtc_full_frozen_steps < 0:
            raise ValueError("rtc_full_frozen_steps must be >= 0")
        if self.rtc_full_frozen_steps > self.rtc_full_overlap_steps:
            raise ValueError("rtc_full_frozen_steps must be <= rtc_full_overlap_steps")
        if self.rtc_full_max_delay_steps < 0:
            raise ValueError("rtc_full_max_delay_steps must be >= 0")
        if self.rtc_full_ramp_rate < 0:
            raise ValueError("rtc_full_ramp_rate must be >= 0")
        if self.rtc_full_prefix_attention_schedule not in {"zeros", "ones", "linear", "exp"}:
            raise ValueError("rtc_full_prefix_attention_schedule must be zeros, ones, linear, or exp")
        if self.rtc_full_max_guidance_weight <= 0:
            raise ValueError("rtc_full_max_guidance_weight must be > 0")
        if self.rtc_lite_max_delay_steps < 0:
            raise ValueError("rtc_lite_max_delay_steps must be >= 0")


# -----------------------
# Master config
# -----------------------
@dataclass
class KuavoConfig:
    env: ConfigEnv
    inference: ConfigInference

    def validate(self):
        self.env.validate()
        self.inference.validate()


# -----------------------
# Loader
# -----------------------
def load_kuavo_config(config_path: Optional[str] = None) -> KuavoConfig:
    """
    Load config from YAML.
    Default path: ./configs/deploy/deploy.yaml

    If the selected config is a simple profile and a matching total config exists
    in a sibling `total/` directory, the total config is loaded first and then the
    simple config overrides it.
    """
    if config_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "../configs", "deploy", DEFAULT_DEPLOY_CONFIG)
    else:
        current_dir = os.path.dirname(os.path.abspath(__file__))

    config_path = os.path.abspath(config_path)

    def _load_yaml(path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(base)
        for key, value in override.items():
            if isinstance(merged.get(key), dict) and isinstance(value, dict):
                merged[key] = _deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged

    def _resolve_template(value: Any, scope: Dict[str, Any]) -> Any:
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            expr = value[2:-1]
            parts = expr.split(".")
            cur: Any = scope
            for part in parts:
                if not isinstance(cur, dict) or part not in cur:
                    raise KeyError(f"Unknown config template reference: {value}")
                cur = cur[part]
            return deepcopy(cur)
        if isinstance(value, list):
            return [_resolve_template(v, scope) for v in value]
        if isinstance(value, dict):
            return {k: _resolve_template(v, scope) for k, v in value.items()}
        return value

    def _apply_inference_env_defaults(cfg_dict: Dict[str, Any]) -> Dict[str, Any]:
        env_cfg = cfg_dict.setdefault("env", {})
        inference_env = env_cfg.get("inference_env", "real")

        if inference_env == "sim":
            env_cfg["env_name"] = "Kuavo-Sim"
            env_cfg["real"] = False

        else:
            env_cfg["env_name"] = "Kuavo-Real"
            env_cfg["real"] = True
            env_cfg.setdefault("platform_type", "4pro")
            env_cfg.setdefault("eef_type", "leju_claw")
            if env_cfg["eef_type"] not in {"leju_claw", "qiangnao", "sg100"}:
                raise ValueError("When inference_env=real, eef_type must be 'leju_claw', 'qiangnao', or 'sg100'")
            env_cfg["head_init"] = None
            env_cfg["image_size"] = [848, 480]

        return cfg_dict

    def _filter_obs_key_map_for_eef(cfg_dict: Dict[str, Any]) -> Dict[str, Any]:
        env_cfg = cfg_dict.get("env", {})
        eef_type = env_cfg.get("eef_type")
        which_arm = env_cfg.get("which_arm", "both")
        obs_key_map = env_cfg.get("obs_key_map", {})
        if not isinstance(obs_key_map, dict):
            return cfg_dict

        filtered = {}
        for key, value in obs_key_map.items():
            if key == "wrist_cam_l" and which_arm == "right":
                continue
            if key == "wrist_cam_r" and which_arm == "left":
                continue
            if key == "depth_l" and which_arm == "right":
                continue
            if key == "depth_r" and which_arm == "left":
                continue
            if key in {"rq2f85", "leju_claw", "qiangnao", "sg100"} and key != eef_type:
                continue
            filtered[key] = value
        env_cfg["obs_key_map"] = filtered
        return cfg_dict

    cfg = _load_yaml(config_path)

    config_dir = os.path.dirname(config_path)
    config_name = os.path.basename(config_path)
    total_path = os.path.join(config_dir, "total", config_name.replace(".yaml", "_total.yaml"))
    fallback_total_path = os.path.join(current_dir, "../configs", "deploy", "total", "deploy_total.yaml")
    is_total_config = (
        os.path.basename(os.path.dirname(config_path)) == "total"
        or config_name.endswith("_total.yaml")
    )
    if not is_total_config:
        base_total_path = total_path if os.path.exists(total_path) else fallback_total_path
        if os.path.exists(base_total_path):
            cfg = _deep_merge(_load_yaml(base_total_path), cfg)

    cfg = _apply_inference_env_defaults(cfg)
    cfg = _resolve_template(cfg, cfg)
    cfg = _filter_obs_key_map_for_eef(cfg)

    # YAML keeps the user-facing option name requested for 5W, while the
    # dataclass uses a valid Python identifier internally.
    for option_key, field_name in (
        ("5w_wholebody", "enable_5w_wholebody"),
        ("5w_base_move", "enable_5w_base_move"),
        ("5w_lowerlock", "lowerlock_5w"),
    ):
        if isinstance(cfg.get("env"), dict) and option_key in cfg["env"]:
            cfg["env"][field_name] = cfg["env"].pop(option_key)
        elif option_key in cfg:
            cfg[field_name] = cfg.pop(option_key)

    # The user's original YAML was mostly top-level keys (not nested under env/inference).
    # We'll support both styles:
    #  - top-level flat (as your original): keys like 'real', 'policy_type', ...
    #  - nested style: {env: {...}, inference: {...}}
    if 'env' in cfg and 'inference' in cfg:
        env_cfg: Dict[str, Any] = cfg.get('env', {})
        inf_cfg: Dict[str, Any] = cfg.get('inference', {})
    else:
        # 自动根据 dataclass 字段划分 env / inference
        env_fields = set(ConfigEnv.__dataclass_fields__.keys())
        inf_fields = set(ConfigInference.__dataclass_fields__.keys())

        env_cfg = {}
        inf_cfg = {}

        for k, v in cfg.items():
            if k in env_fields:
                env_cfg[k] = v
            elif k in inf_fields:
                inf_cfg[k] = v
            elif k == "limits" and isinstance(v, dict):
                def dict_to_range(d):
                    return Range(d.get("min", []), d.get("max", []))

                env_cfg["limits"] = LimitsConfig(
                    joint_q=dict_to_range(v.get("joint_q", {})),
                    gripper=dict_to_range(v.get("gripper", {})),
                    lower_body=dict_to_range(v.get("lower_body", {})),
                )
            else:
                env_cfg[k] = v
    # Merge defaults with provided config
    default_env = ConfigEnv()
    default_inf = ConfigInference()

    merged_env = {**asdict(default_env), **env_cfg}
    if isinstance(env_cfg.get("limits"), dict):
        merged_env["limits"] = _deep_merge(asdict(default_env.limits), env_cfg["limits"])
    if merged_env["eef_type"] == "sg100" and merged_env["sg100_dof_needed"] == 11:
        for bound in ("min", "max"):
            values = merged_env["limits"]["gripper"][bound]
            if len(values) == 2:
                merged_env["limits"]["gripper"][bound] = [values[0]] * 11 + [values[1]] * 11
    merged_inf = {**asdict(default_inf), **inf_cfg}

    env = ConfigEnv(**merged_env)
    inference = ConfigInference(**merged_inf)

    config = KuavoConfig(env=env, inference=inference)
    config.env.obs_key_map = config.env.build_obs_key_map()
    config.validate()
    return config


# -----------------------
# Quick test when run as script
# -----------------------
if __name__ == "__main__":
    cfg = load_kuavo_config()
    print(isinstance(cfg, KuavoConfig))
    print("=== Env basic ===")
    print("eef_type:", cfg.env.eef_type)
    print("eef_name:", cfg.env.env_name)
    print("which_arm:", cfg.env.which_arm)
    print("cam keys:", cfg.env.obs_key_map)
    print("slice_robot:", cfg.env.gripper_slice)
    print("qiangnao_slice:", cfg.env.joint_q_slice)
    print("=== Inference basic ===")
    print("policy_type:", cfg.inference.policy_type)
    print("device:", cfg.inference.device)
    print("arm_state_keys",cfg.env.arm_state_keys)
