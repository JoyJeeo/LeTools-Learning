#!/usr/bin/env python3
import rospy
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, JointState
from geometry_msgs.msg import Twist
import cv2
import gymnasium as gym
import time
import sys
from kuavo_humanoid_sdk import KuavoSDK, KuavoRobot, KuavoRobotState, DexterousHand
from kuavo_humanoid_sdk.msg.kuavo_msgs.msg import lejuClawCommand
from kuavo_humanoid_sdk.msg.kuavo_msgs.srv import (changeArmCtrlMode, changeArmCtrlModeRequest)
from kuavo_deploy.utils.logging_utils import setup_logger
from kuavo_deploy.config import KuavoConfig
from kuavo_deploy.utils.ros_manager import ROSManager
import traceback
import torch
from torchvision.transforms.functional import to_tensor
from kuavo_deploy.utils.obs_buffer import ObsBuffer
from kuavo_deploy.utils.signal_controller import ControlSignalManager
from kuavo_deploy.utils.lowpass_filter import LowPassFilter
from kuavo_deploy.utils.sg100_msgs import SG100HandCommand
from std_srvs.srv import SetBool

log_robot = setup_logger("robot")
BASE_VELOCITY_DOF = 3


class KuavoBaseRosEnv(gym.Env):
    """Kuavo机器人ROS环境基类"""

    def __init__(self, config: KuavoConfig):
        self._init_lower_body_lock(config)
        self._set_config(config.env)
        
        # 初始化ROS管理器
        self.ros_manager = ROSManager()
        self.control_signal_manager = ControlSignalManager()
        
        # 初始化其他组件
        self.bridge = CvBridge()
        self._set_observation_space()
        self._set_action_space()
        self._init_kuavo_sdk()
        self._set_ros_topics()
        
        # 等待ROS话题初始化
        log_robot.info(f"Inializing done!")
        print(f"Inializing done!")

    def _init_lower_body_lock(self, config: KuavoConfig):
        """Cache the configured go bag's first lower-body target in radians."""
        self.lower_body_lock_mask = np.asarray(config.env.lowerlock_5w, dtype=bool)
        self.lower_body_lock_target = None
        if not config.env.use_5w_wholebody or not self.lower_body_lock_mask.any():
            return
        bag_path = config.inference.go_bag_path
        if not bag_path:
            raise ValueError("env.5w_lowerlock requires inference.go_bag_path")

        import rosbag

        with rosbag.Bag(bag_path, "r") as bag:
            for _, msg, _ in bag.read_messages(topics=["/lb_leg_traj"]):
                positions = np.asarray(msg.position, dtype=float)
                if positions.shape != (4,) or not np.isfinite(positions).all():
                    raise ValueError(
                        "env.5w_lowerlock requires four finite positions in the "
                        "first /lb_leg_traj frame of inference.go_bag_path"
                    )
                self.lower_body_lock_target = np.deg2rad(positions)
                break
        if self.lower_body_lock_target is None:
            raise ValueError(
                "env.5w_lowerlock requires /lb_leg_traj in inference.go_bag_path"
            )

    def _set_config(self, config_kuavo_env):
        """设置配置参数"""
        self.platform_type = config_kuavo_env.platform_type
        self.use_5w_wholebody = config_kuavo_env.use_5w_wholebody
        self.use_5w_base_move = config_kuavo_env.use_5w_base_move
        self.arm_dof_per_side = config_kuavo_env.arm_dof_per_side
        self.arm_dof = self.arm_dof_per_side * 2
        self.eef_dof_per_side = config_kuavo_env.eef_dof_per_side
        self.ros_rate = config_kuavo_env.ros_rate
        self.control_mode = config_kuavo_env.control_mode
        self.obs_key_map = config_kuavo_env.obs_key_map
        self.eef_type = config_kuavo_env.eef_type
        self.which_arm = config_kuavo_env.which_arm
        self.direct_to_wbc = config_kuavo_env.direct_to_wbc
        self.qiangnao_dof_needed = config_kuavo_env.qiangnao_dof_needed
        self.sg100_dof_needed = config_kuavo_env.sg100_dof_needed
        self.sg100_dof_offset = config_kuavo_env.sg100_dof_offset
        self.sg100_open_pose = np.asarray(config_kuavo_env.sg100_open_pose, dtype=np.float64)
        self.sg100_close_pose = np.asarray(config_kuavo_env.sg100_close_pose, dtype=np.float64)
        self.sg100_init_pose = np.asarray(config_kuavo_env.sg100_init_pose, dtype=np.float64)
        self.sg100_force_threshold = config_kuavo_env.sg100_force_threshold
        self.sg100_force_kp = np.asarray(config_kuavo_env.sg100_force_kp, dtype=np.float64)
        self.sg100_force_kd = np.asarray(config_kuavo_env.sg100_force_kd, dtype=np.float64)
        self.sg100_force_torque_ff = np.asarray(config_kuavo_env.sg100_force_torque_ff, dtype=np.float64)
        self.sg100_force_output_limit = np.asarray(config_kuavo_env.sg100_force_output_limit, dtype=np.float64)
        self.sg100_force_velocities = np.asarray(config_kuavo_env.sg100_force_velocities, dtype=np.float64)
        self.is_binary_sg100_action = config_kuavo_env.is_binary_sg100_action
        self.sg100_action_threshold = config_kuavo_env.sg100_action_threshold
        self.control_rate_hz = getattr(config_kuavo_env, "control_rate", 100)
        self.enable_action_interpolation = getattr(config_kuavo_env, "enable_action_interpolation", True)
        self.interpolation_steps = (
            max(1, int(round(self.control_rate_hz / self.ros_rate)))
            if self.enable_action_interpolation
            else 1
        )

        self.is_binary = config_kuavo_env.is_binary
        self.head_init = config_kuavo_env.head_init
        self.arm_init = np.zeros(self.arm_dof, dtype=np.float64)


        # 从配置中获取limits部分
        self.limits = config_kuavo_env.limits
        self.obs_key_map = config_kuavo_env.obs_key_map
        self.obs_buffer = ObsBuffer(
            config=config_kuavo_env, 
            obs_key_map=self.obs_key_map,
        )
        self.arm_state_keys = config_kuavo_env.arm_state_keys # observation.state 的key顺序
        self.ratio = config_kuavo_env.ratio
        self.frame_alignment = config_kuavo_env.frame_alignment
        if self.direct_to_wbc:
            self.last_predicted_action = None
            self.is_first_step = True
            self.low_pass_filter = LowPassFilter(cutoff_hz=1.0, dt=1.0 / self.control_rate_hz)
            log_robot.info(
                f"Direct-to-WBC action interpolation: enabled={self.enable_action_interpolation}, "
                f"inference_rate={self.ros_rate}Hz, control_rate={self.control_rate_hz}Hz, "
                f"interpolation_steps={self.interpolation_steps}"
            )

    @property
    def _controlled_sides(self):
        return ("left", "right") if self.which_arm == "both" else (self.which_arm,)

    def _split_action(self, action):
        """Split the configured packed action without fixed 8D/16D offsets."""
        action = np.asarray(action)
        components = {}
        offset = 0
        for side in self._controlled_sides:
            joint_end = offset + self.arm_dof_per_side
            eef_end = joint_end + self.eef_dof_per_side
            components[side] = {
                "joints": action[offset:joint_end],
                "eef": action[joint_end:eef_end],
                "eef_slice": slice(joint_end, eef_end),
            }
            offset = eef_end
        if self.use_5w_wholebody:
            lower_dof = len(self.limits["lower_body"]["min"])
            lower_end = offset + lower_dof
            components["lower_body"] = action[offset:lower_end]
            offset = lower_end
        else:
            components["lower_body"] = np.array([])
        if self.use_5w_base_move:
            base_end = offset + BASE_VELOCITY_DOF
            components["base_velocity"] = action[offset:base_end]
        else:
            components["base_velocity"] = np.array([])
        return components

    def _full_arm_target(self, components):
        left = components.get("left", {}).get("joints", self.arm_init[:self.arm_dof_per_side])
        right = components.get("right", {}).get("joints", self.arm_init[self.arm_dof_per_side:])
        return np.concatenate((left, right), axis=0)

    def _set_observation_space(self):
        limits = self.limits
        obs_low, obs_high = [], []

        # -------- 构建 state 空间（joint_q + gripper） --------
        if 'joint_q' in self.obs_key_map:
            joint_min, joint_max = limits['joint_q']['min'], limits['joint_q']['max']
        else:
            joint_min, joint_max = [], []
        if 'gripper' in self.obs_key_map:
            grip_min, grip_max = limits['gripper']['min'], limits['gripper']['max']
        else:
            grip_min, grip_max = [], []
        for side_index, side in enumerate(("left", "right")):
            if side not in self._controlled_sides:
                continue
            arm_start = side_index * self.arm_dof_per_side
            arm_end = arm_start + self.arm_dof_per_side
            eef_start = side_index * self.eef_dof_per_side
            eef_end = eef_start + self.eef_dof_per_side
            obs_low.extend(joint_min[arm_start:arm_end] + grip_min[eef_start:eef_end])
            obs_high.extend(joint_max[arm_start:arm_end] + grip_max[eef_start:eef_end])
        if self.use_5w_wholebody:
            obs_low.extend(limits['lower_body']['min'])
            obs_high.extend(limits['lower_body']['max'])

        self.obs_low = np.array(obs_low)
        self.obs_high = np.array(obs_high)

        # -------- 构建图像空间 --------
        obs_spaces = {}
        for key, obs_name in self.obs_key_map.items():
            if any(tag in key for tag in ['cam', 'depth']):
                h, w = obs_name["handle"]["params"]["resize_wh"]
                if 'depth' in key:
                    low, high = obs_name['handle']['params']['depth_range']
                    obs_spaces[f"observation.{key}"] = gym.spaces.Box(
                        low=low, high=high, shape=(1, h, w), dtype=np.uint16
                    )
                else:  # cam 类键
                    obs_spaces[f"observation.images.{key}"] = gym.spaces.Box(
                        low=0, high=255, shape=(3, h, w), dtype=np.uint8
                    )

        # -------- 添加 state 空间 --------
        obs_spaces["observation.state"] = gym.spaces.Box(
            low=self.obs_low,
            high=self.obs_high,
            dtype=np.float32,
            shape=(len(self.obs_low),)
        )

        self.observation_space = gym.spaces.Dict(obs_spaces)

    def _set_action_space(self):
        limits = self.limits
        if self.control_mode != 'joint':
            raise ValueError(f"Unsupported control mode: {self.control_mode}")
        arm_low, arm_high = [], []
        for side_index, side in enumerate(("left", "right")):
            if side not in self._controlled_sides:
                continue
            arm_start = side_index * self.arm_dof_per_side
            arm_end = arm_start + self.arm_dof_per_side
            eef_start = side_index * self.eef_dof_per_side
            eef_end = eef_start + self.eef_dof_per_side
            arm_low.extend(limits['joint_q']['min'][arm_start:arm_end] + limits['gripper']['min'][eef_start:eef_end])
            arm_high.extend(limits['joint_q']['max'][arm_start:arm_end] + limits['gripper']['max'][eef_start:eef_end])
        if self.use_5w_wholebody:
            arm_low.extend(limits['lower_body']['min'])
            arm_high.extend(limits['lower_body']['max'])
        if self.use_5w_base_move:
            # No project-level velocity limits are configured. Preserve model
            # outputs here; safe limits should be enforced by the base controller.
            arm_low.extend([-np.inf] * BASE_VELOCITY_DOF)
            arm_high.extend([np.inf] * BASE_VELOCITY_DOF)

        # ===============================
        # 创建 Gym Box 空间
        # ===============================
        self.action_space = gym.spaces.Box(
            low=np.array(arm_low, dtype=np.float64),
            high=np.array(arm_high, dtype=np.float64),
            dtype=np.float64,
        )

    def _init_kuavo_sdk(self):
        """初始化Kuavo SDK"""
        if not KuavoSDK().Init():
            log_robot.error("Init KuavoSDK failed, exit!")
            sys.exit(1)
        self.robot = KuavoRobot()
        self.robot_state = KuavoRobotState()

    def _set_ros_topics(self):
        """设置ROS话题"""
        self.rate = rospy.Rate(self.ros_rate)
        self.control_rate = rospy.Rate(self.control_rate_hz)
        
        if self.eef_type == 'rq2f85':
            self.pub_eef_joint = self.ros_manager.register_publisher('/gripper/command', JointState, queue_size=10)
        elif self.eef_type == 'leju_claw':
            self.lejuclaw = LejuClaw()
        elif self.eef_type == 'qiangnao':
            self.qiangnao = DexterousHand()
        elif self.eef_type == 'sg100':
            self.sg100 = SG100Hand(ros_manager=self.ros_manager)
        if self.use_5w_base_move:
            self.cmd_vel_pub = self.ros_manager.register_publisher(
                '/cmd_vel', Twist, queue_size=10
            )
        # obs buffer 初始化            
        self.obs_buffer.wait_buffer_ready()


    def reset(self, **kwargs):
        """重置机器人状态"""
        self.stop_base()
        self._enter_external_control_mode()
        self._reset_head()
        self._reset_eef()

        # === 平均当前观测和位姿 ===
        avg_data = self._compute_average_state(average_num=10)

        # === 更新状态 ===
        self.cur_state = avg_data["state"]
        self.cur_joint_angles_action = avg_data["joint_action"]

        obs = self.get_obs()
        self.sleep_time = 0
        self.average_sleep_time = 0

        if self.direct_to_wbc:
            self.last_predicted_action = None
            self.low_pass_filter.reset()
            self.is_first_step = True
        return obs, {}

    # ==========================================================
    # 子函数 1. 外部控制模式设置
    # ==========================================================
    def _set_direct_to_wbc(self, control_mode):
        rospy.wait_for_service('/enable_wbc_arm_trajectory_control', timeout=5)
        try:
            change_mode = rospy.ServiceProxy('/enable_wbc_arm_trajectory_control', changeArmCtrlMode)
            req = changeArmCtrlModeRequest()
            req.control_mode = control_mode
            res = change_mode(req)
            if res.result:
                rospy.loginfo("wbc轨迹控制模式已更改为 %d", control_mode)
            else:
                rospy.logerr("无法将wbc轨迹控制模式更改为 %d", control_mode)
        except rospy.ServiceException as e:
            rospy.logerr("服务调用失败: %s", e)


    def _call_enable_arm_quick_mode(self, enable):
        """调用手臂快速模式切换服务"""
        try:
            # 等待服务可用
            rospy.loginfo(f"等待服务 /enable_lb_arm_quick_mode 可用...")
            rospy.wait_for_service('/enable_lb_arm_quick_mode', timeout=5.0)

            # 创建服务客户端
            service_client = rospy.ServiceProxy('/enable_lb_arm_quick_mode', SetBool)

            # 调用服务
            rospy.loginfo(f"调用服务: {'启用' if enable else '禁用'}手臂快速模式")
            response = service_client(enable)

            # 处理响应
            if response.success:
                rospy.loginfo(f"✓ 成功{'启用' if enable else '禁用'}手臂快速模式")
                if response.message:
                    rospy.loginfo(f"  消息: {response.message}")
                return True
            else:
                rospy.logwarn(f"✗ 服务调用失败")
                if response.message:
                    rospy.logwarn(f"  消息: {response.message}")
                return False

        except rospy.ROSException as e:
            rospy.logerr(f"服务不可用: {e}")
            return False
        except rospy.ServiceException as e:
            rospy.logerr(f"服务调用异常: {e}")
            return False

    def _enter_external_control_mode(self):
        self.robot.set_external_control_arm_mode()
        print("set_external_control_arm_mode", self.robot_state.arm_control_mode())
        if self.direct_to_wbc:
            if self.platform_type in ["4pro","5"]:
                self._set_direct_to_wbc(1)
            else:
                self._call_enable_arm_quick_mode(enable=True)


    # ==========================================================
    # 子函数 2. 头部复位
    # ==========================================================
    def _reset_head(self):
        if self.head_init is not None:
            self.robot.control_head(self.head_init[0], self.head_init[1])

    # ==========================================================
    # 子函数 3. 末端执行器（夹爪）复位
    # ==========================================================
    def _reset_eef(self):
        if self.which_arm == 'both':
            if self.eef_type == 'qiangnao':
                self.qiangnao.control(target_positions=[0, 100, 0, 0, 0, 0, 0, 100, 0, 0, 0, 0], target_velocities=None, target_torques=None)
            elif self.eef_type == 'leju_claw':
                self.lejuclaw.control(target_positions=[0, 0], target_velocities=None, target_torques=None)
            elif self.eef_type == 'sg100':
                self.sg100.control(self.sg100_init_pose.tolist())
        elif self.which_arm == 'left':
            if self.eef_type == 'qiangnao':
                self.qiangnao.control_left(target_positions=[0, 100, 0, 0, 0, 0], target_velocities=None, target_torques=None)
            elif self.eef_type == 'leju_claw':
                self.lejuclaw.control_left(target_positions=[0], target_velocities=None, target_torques=None)
            elif self.eef_type == 'sg100':
                self.sg100.control_left(self.sg100_init_pose[:11].tolist())
        elif self.which_arm == 'right':
            if self.eef_type == 'qiangnao':
                self.qiangnao.control_right(target_positions=[0, 100, 0, 0, 0, 0], target_velocities=None, target_torques=None)
            elif self.eef_type == 'leju_claw':
                self.lejuclaw.control_right(target_positions=[0], target_velocities=None, target_torques=None)
            elif self.eef_type == 'sg100':
                self.sg100.control_right(self.sg100_init_pose[11:].tolist())
        else:
            raise KeyError(f"Unsupported arm type: {self.which_arm}")

    # ==========================================================
    # 子函数 4. 求平均状态与末端姿态
    # ==========================================================
    def _compute_average_state(self, average_num=10):
        state_sum, joint_sum = None, None

        for i in range(average_num):
            state = self.get_obs()
            fk_joint_angles = self._get_init_joint_angles(self.arm_state["joint_q"])

            if i == 0:
                state_sum = np.array(state["observation.state"], dtype=float)
                joint_sum = np.array(fk_joint_angles, dtype=float)
            else:
                state_sum += np.array(state["observation.state"], dtype=float)
                joint_sum += fk_joint_angles
            time.sleep(0.001)

        # 平均计算
        avg_state = state_sum / average_num
        avg_joint = joint_sum / average_num

        return {
            "state": avg_state,
            "joint_action": avg_joint,
        }

    # ==========================================================
    # 子函数 5. 构造完整 FK 输入
    # ==========================================================
    def _get_init_joint_angles(self, joint_q):
        if self.which_arm == 'both':
            return np.array(joint_q)
        elif self.which_arm == 'left':
            return np.concatenate((joint_q, self.arm_init[self.arm_dof_per_side:]))
        elif self.which_arm == 'right':
            return np.concatenate((self.arm_init[:self.arm_dof_per_side], joint_q))
        else:
            raise ValueError(f"Invalid which_arm: {self.which_arm}")
    
    def check_action(self, action, mode='default'):
        if mode == 'default':  # 比较 action_space
            if len(action) != len(self.action_space.low):
                raise ValueError(f"action shape must be {len(self.action_space.low)}")
            if np.any(action < self.action_space.low) or np.any(action > self.action_space.high):
                log_robot.warning(
                    f"action out of range, action: {action}, "
                    f"low: {self.action_space.low}, high: {self.action_space.high}"
                )
                action = np.clip(action, self.action_space.low, self.action_space.high)
            return action

        raise ValueError(f"Unsupported mode: {mode}")

    def step(self, action):
        t0 = time.time()
        log_robot.info(f"action: {action}")
        # check clip action in action space
        action = self.check_action(action, mode='default')
        t1 = time.time()
        log_robot.info(f"clip action: {action}, check time: {t1 - t0:.3f}s")



        # === 4. 执行动作 ===
        t2 = time.time()
        action_components = self._split_action(action)
        self.cur_joint_angles_action = self._full_arm_target(action_components)

        if not self.direct_to_wbc:
            self.exec_action(action)
            self.rate.sleep()
        else:
            # ==== 4.1 插值下发动作（direct_to_wbc 路径）
            # 第一帧从 current state 插值过去；之后每帧在 last_predicted_action -> action 之间插值。
            # Interpolate according to the configured packed action layout.

            current_q = self._get_init_joint_angles(self.arm_state["joint_q"])
            num_inter_points = self.interpolation_steps

            if self.is_first_step:
                self.is_first_step = False

                target_joints = self._full_arm_target(action_components)
                current_lower = np.asarray(self.arm_state.get("lower_body", []), dtype=float)
                target_lower = action_components["lower_body"]

                for i in range(num_inter_points):
                    alpha = (i + 1) / num_inter_points
                    inter_joints = (1 - alpha) * current_q + alpha * target_joints
                    self.safe_control_arm(inter_joints)
                    if self.use_5w_wholebody:
                        inter_lower = (1 - alpha) * current_lower + alpha * target_lower
                        self.safe_control_lower_body(inter_lower)
                    if self.use_5w_base_move:
                        self.safe_control_base(alpha * action_components["base_velocity"])
                    self.control_rate.sleep()

                self.exec_action(action)
                self.last_predicted_action = action.copy()
            else:
                # 后续每一帧：在 last_predicted_action 与 action 之间按 action 维度插值，
                # 末端夹爪保持目标值不插（直接用本帧目标），然后调 exec_action 走对应的 left/right/both 分支
                if self.last_predicted_action is None:
                    self.last_predicted_action = action.copy()

                eef_targets = [
                    component["eef_slice"]
                    for side, component in action_components.items()
                    if side in ("left", "right")
                ]

                for i in range(num_inter_points):
                    alpha = (i + 1) / num_inter_points
                    inter_arm_action = (1 - alpha) * self.last_predicted_action + alpha * action
                    # End effectors retain the current target instead of interpolation.
                    for eef_slice in eef_targets:
                        inter_arm_action[eef_slice] = action[eef_slice]

                    self.exec_action(inter_arm_action)
                    self.control_rate.sleep()

                self.last_predicted_action = action.copy()
                # self.rate.sleep()
        # === 5. 延时与观测 ===
        
        self._record_sleep_time(t2)
        t3 = time.time()
        obs = self.get_obs()
        t4 = time.time()
        log_robot.info(f"get obs time: {t4 - t3:.3f}s")

        # === 6. 奖励与返回 ===
        reward = self.compute_reward()
        return obs, reward, False, False, {}
    
    
    def _record_sleep_time(self, t_start):
        self.sleep_time = time.time() - t_start
        self.average_sleep_time += self.sleep_time
        log_robot.info(f"rate.sleep time: {self.sleep_time:.3f}s")

    def safe_control_arm(self, target_position):
        try:
            self.robot.control_arm_joint_positions(target_position)
        except RuntimeError as e:
            # 当机器人处于 command_pose_world 状态（底盘移动）时，无法控制手臂
            if "must be in stance state" in str(e):
                log_robot.warning(f"⚠️  无法发送手臂命令：机器人当前状态不允许 (可能正在底盘移动)")
                log_robot.debug(f"   详细错误: {e}")
            else:
                raise

    def safe_control_lower_body(self, target_position):
        """Send model-space radians through the SDK's degree-based 5W API."""
        if self.lower_body_lock_target is not None:
            target_position = np.asarray(target_position, dtype=float).copy()
            target_position[self.lower_body_lock_mask] = self.lower_body_lock_target[
                self.lower_body_lock_mask
            ]
        try:
            self.robot.control_wheel_lower_joint(np.rad2deg(target_position).tolist())
        except RuntimeError as e:
            if "must be in stance state" in str(e):
                log_robot.warning("Unable to send 5W lower-body command in the current robot state")
            else:
                raise

    def safe_control_base(self, velocity):
        """Publish model-space [vx, vy, vyaw] to geometry_msgs/Twist."""
        velocity = np.asarray(velocity, dtype=float).reshape(-1)
        if velocity.size != BASE_VELOCITY_DOF or not np.isfinite(velocity).all():
            raise ValueError(
                f"Base velocity must contain {BASE_VELOCITY_DOF} finite values "
                f"[vx, vy, vyaw], got {velocity}"
            )
        msg = Twist()
        msg.linear.x = float(velocity[0])
        msg.linear.y = float(velocity[1])
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = float(velocity[2])
        self.cmd_vel_pub.publish(msg)

    def stop_base(self):
        """Publish a zero Twist when base control is active."""
        if getattr(self, 'use_5w_base_move', False) and hasattr(self, 'cmd_vel_pub'):
            self.safe_control_base(np.zeros(BASE_VELOCITY_DOF, dtype=float))

    def exec_action(self, action):
        """执行机械臂与末端执行器动作"""
        if self.direct_to_wbc:
            action = self.low_pass_filter.update(action)
        components = self._split_action(action)
        target_position = self._full_arm_target(components)
        self.safe_control_arm(target_position)
        empty_eef = np.zeros(self.eef_dof_per_side, dtype=float)
        left_eef = components.get("left", {}).get("eef", empty_eef)
        right_eef = components.get("right", {}).get("eef", empty_eef)
        self._control_eef(left_eef, right_eef)
        if self.use_5w_wholebody:
            self.safe_control_lower_body(components["lower_body"])
        if self.use_5w_base_move:
            self.safe_control_base(components["base_velocity"])



    def _control_eef(self, left_eef, right_eef):
        """根据 eef_type 控制不同的末端执行器"""
        left_eef = np.asarray(left_eef, dtype=float).reshape(-1)
        right_eef = np.asarray(right_eef, dtype=float).reshape(-1)
        if self.eef_type == 'rq2f85':
            eef_msg = JointState()
            try:
                eef_msg.name = ['left_gripper_joint', 'right_gripper_joint']
            except Exception as e:
                log_robot.info(f"_control_eef error! {e}")
            eef_msg.position = np.concatenate((left_eef, right_eef)) * 255
            self.pub_eef_joint.publish(eef_msg)

        elif self.eef_type == 'leju_claw':
            eef_msg = JointState()
            eef_msg.position = np.concatenate((left_eef, right_eef)) * 100
            self.lejuclaw.control(target_positions=eef_msg.position)

        elif self.eef_type == 'qiangnao':
            if self.qiangnao_dof_needed != 1:
                raise KeyError("qiangnao_dof_needed != 1 is not supported!")

            tem_left, tem_right = left_eef.item() * 100, right_eef.item() * 100
            target_positions = np.array([
                tem_left, 100, *([tem_left] * 4),
                tem_right, 100, *([tem_right] * 4)
            ])
            self.qiangnao.control(target_positions=target_positions)

        elif self.eef_type == 'sg100':
            left_eef = np.asarray(left_eef, dtype=np.float64).reshape(-1)
            right_eef = np.asarray(right_eef, dtype=np.float64).reshape(-1)
            if self.is_binary_sg100_action:
                left_eef = (left_eef > self.sg100_action_threshold).astype(np.float64)
                right_eef = (right_eef > self.sg100_action_threshold).astype(np.float64)

            def _expand(values: np.ndarray, hand_slice: slice):
                if values.size == 1:
                    # The single-DOF representation is a normalized open/close
                    # amount, so it must stay within [0, 1].
                    amount = float(np.clip(values[0], 0.0, 1.0))
                    positions = (
                        self.sg100_open_pose[hand_slice] * (1.0 - amount)
                        + self.sg100_close_pose[hand_slice] * amount
                    )
                    mode = (SG100HandCommand.MODE_JOINT_IMPEDANCE
                            if amount > self.sg100_force_threshold
                            else SG100HandCommand.MODE_JOINT_POSITION)
                    modes = [mode] * 11
                elif values.size == 11:
                    # Full-hand datasets store each raw joint angle divided by
                    # 1.57. Some SG100 joints legitimately use negative values
                    # (for example right thumb_j2), so do not apply the scalar
                    # open/close clipping here.
                    positions = values * 1.57
                    modes = [SG100HandCommand.MODE_JOINT_POSITION] * 11
                else:
                    raise ValueError(f"SG100 action must have 1 or 11 values per hand, got {values.size}")
                return positions.tolist(), modes

            left_positions, left_modes = _expand(left_eef, slice(0, 11))
            right_positions, right_modes = _expand(right_eef, slice(11, 22))
            self.sg100.control(
                target_positions=left_positions + right_positions,
                control_modes=left_modes + right_modes,
                kp=self.sg100_force_kp.tolist(),
                kd=self.sg100_force_kd.tolist(),
                torque_ff=self.sg100_force_torque_ff.tolist(),
                output_limit=self.sg100_force_output_limit.tolist(),
                target_velocities=self.sg100_force_velocities.tolist(),
                enable_left=self.which_arm in ('left', 'both'),
                enable_right=self.which_arm in ('right', 'both'),
            )

        else:
            raise KeyError(f"Unsupported eef_type: {self.eef_type}")

    def compute_reward(self):
        """计算奖励"""
        return 0

    def get_obs(self):
        """获取观测图像及state等"""
        obs = {}
        self.arm_state = {}

        if self.frame_alignment:
            obs_from_buffer = self.obs_buffer.get_aligned_obs(reference_keys=None, max_dt=1/self.ros_rate,ratio=self.ratio)
            if obs_from_buffer is None or not all(v is not None for v in obs_from_buffer.values()):
                obs_from_buffer = self.obs_buffer.get_aligned_obs(reference_keys=None, max_dt=float('inf'),ratio=self.ratio)
        else:
            obs_from_buffer = self.obs_buffer.get_latest_obs()
        
        for k,v in obs_from_buffer.items():
            # remap key
            if 'depth' in k:
                obs[f"observation.{k}"] = v
            elif 'cam' in k:
                obs[f"observation.images.{k}"] = v
            else:
                self.arm_state[f"{k}"] = v

        if self.is_binary:
            self.arm_state['gripper'] = np.where(self.arm_state['gripper']>0.5, 1, 0)

        assert len(self.arm_state.keys()) >= 2, f"arm_state must have exactly 2 elements, but got {len(self.arm_state.keys())}"

        state_keys = [k for k in self.arm_state_keys if k in self.arm_state and k != "lower_body"]

        arm_data = { "left": [], "right": [] }

        for key in state_keys:
            data = self.arm_state[key]
            if len(data) == 0:
                continue
            mid = len(data) // 2
            if self.which_arm == "both":
                arm_data["left"].append(data[:mid])
                arm_data["right"].append(data[mid:])
            elif self.which_arm == "left":
                arm_data["left"].append(data)
            elif self.which_arm == "right":
                arm_data["right"].append(data)
            else:
                raise KeyError(f"Unsupported which_arm: {self.which_arm}")

        # 拼接结果
        state_parts = arm_data["left"] + arm_data["right"]
        if self.use_5w_wholebody:
            state_parts.append(np.asarray(self.arm_state["lower_body"]))
        obs["observation.state"] = np.concatenate(state_parts, axis=0)
        log_robot.info(f"STATE: contained {state_keys}, concated value: {obs['observation.state']}")

        obs["observation.state"] = torch.from_numpy(obs["observation.state"]).float().unsqueeze(0)
        return obs    

    def close(self):
        """关闭环境，释放资源"""
        log_robot.info("Closing KuavoBaseRosEnv...")
        try:
            self.stop_base()
            if hasattr(self, 'obs_buffer'):
                self.obs_buffer.stop_subscribers()
                if hasattr(self.obs_buffer, 'obs_buffer_data'):
                    for k in self.obs_buffer.obs_buffer_data:
                        self.obs_buffer.obs_buffer_data[k]["data"].clear()
                        self.obs_buffer.obs_buffer_data[k]["timestamp"].clear()
                if hasattr(self.obs_buffer, 'ros_manager'):
                    self.obs_buffer.ros_manager = None
                if hasattr(self.obs_buffer, 'control_signal_manager'):
                    self.obs_buffer.control_signal_manager = None
                del self.obs_buffer
            
            if hasattr(self, 'ros_manager'):
                self.ros_manager.close()
            if hasattr(self, 'control_signal_manager'):
                self.control_signal_manager.close()
            log_robot.info("KuavoBaseRosEnv closed successfully.")
        except Exception as e:
            log_robot.error(f"Error closing KuavoBaseRosEnv: {e}")
            traceback.print_exc()

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()

class LejuClaw:
    """乐聚爪手控制器"""
    def __init__(self, ros_manager=None):
        self.ros_manager = ros_manager or ROSManager()
        self._pub_leju_claw_cmd = self.ros_manager.register_publisher('/leju_claw_command', lejuClawCommand, queue_size=10)

    def control(self, target_positions: list, target_velocities: list = None, target_torques: list = None):
        """控制双手"""
        self._validate_inputs(target_positions, target_velocities, target_torques, 2)
        
        cmd = lejuClawCommand()
        cmd.data.name = ['left_claw', 'right_claw']
        
        target_positions = [max(0.0, min(100.0, pos)) for pos in target_positions]
        target_velocities = self._get_default_velocities(target_velocities, 2)
        target_torques = self._get_default_torques(target_torques, 2)
        
        cmd.data.position = target_positions
        cmd.data.velocity = target_velocities
        cmd.data.effort = target_torques
        self._pub_leju_claw_cmd.publish(cmd)

    def control_left(self, target_positions: list, target_velocities: list = None, target_torques: list = None):
        """控制左手"""
        self._validate_inputs(target_positions, target_velocities, target_torques, 1)
        self.control(
            [target_positions[0], 0],
            [target_velocities[0] if target_velocities else 90, 0],
            [target_torques[0] if target_torques else 1.0, 0]
        )

    def control_right(self, target_positions: list, target_velocities: list = None, target_torques: list = None):
        """控制右手"""
        self._validate_inputs(target_positions, target_velocities, target_torques, 1)
        self.control(
            [0, target_positions[0]],
            [0, target_velocities[0] if target_velocities else 90],
            [0, target_torques[0] if target_torques else 1.0]
        )

    def _validate_inputs(self, positions, velocities, torques, expected_len):
        """验证输入参数"""
        assert len(positions) == expected_len, f"target_positions must be a list of length {expected_len}"
        if velocities is not None:
            assert len(velocities) == expected_len, f"target_velocities must be a list of length {expected_len}"
        if torques is not None:
            assert len(torques) == expected_len, f"target_torques must be a list of length {expected_len}"

    def _get_default_velocities(self, velocities, length):
        """获取默认速度"""
        if velocities is None:
            return [90] * length
        return [max(0.0, min(100.0, vel)) for vel in velocities]

    def _get_default_torques(self, torques, length):
        """获取默认力矩"""
        if torques is None:
            return [1.0] * length
        return [max(0.0, min(10.0, torque)) for torque in torques]

    def close(self):
        """释放资源"""
        if hasattr(self, 'ros_manager'):
            self.ros_manager.close()


class SG100Hand:
    """Publisher for the repository-local SG100 ROS command message."""

    JOINTS_PER_HAND = 11
    ALL_JOINTS_MASK = 0x07FF

    def __init__(self, ros_manager=None):
        self.ros_manager = ros_manager or ROSManager()
        self._pub = self.ros_manager.register_publisher(
            '/sg100_hand_command', SG100HandCommand, queue_size=10
        )

    @staticmethod
    def _optional_22(name, values):
        if values is None:
            return None
        if len(values) != 22:
            raise ValueError(f"{name} must contain 22 values, got {len(values)}")
        return list(values)

    def control(
        self,
        target_positions,
        control_modes=None,
        kp=None,
        kd=None,
        torque_ff=None,
        output_limit=None,
        target_velocities=None,
        *,
        enable_left=True,
        enable_right=True,
    ):
        if len(target_positions) != 22:
            raise ValueError(f"target_positions must contain 22 values, got {len(target_positions)}")

        control_modes = self._optional_22("control_modes", control_modes)
        kp = self._optional_22("kp", kp)
        kd = self._optional_22("kd", kd)
        torque_ff = self._optional_22("torque_ff", torque_ff)
        output_limit = self._optional_22("output_limit", output_limit)
        target_velocities = self._optional_22("target_velocities", target_velocities)

        cmd = SG100HandCommand()
        cmd.header.stamp = rospy.Time.now()
        positions = [float(np.clip(value, -5, 5)) for value in target_positions]
        cmd.left_hand_positions = positions[:11]
        cmd.right_hand_positions = positions[11:]
        cmd.left_enable_mask = self.ALL_JOINTS_MASK if enable_left else 0
        cmd.right_enable_mask = self.ALL_JOINTS_MASK if enable_right else 0

        if control_modes is None:
            cmd.control_mode = SG100HandCommand.MODE_JOINT_POSITION
        else:
            cmd.control_mode = int(control_modes[0])
            cmd.left_hand_control_mode = [int(value) for value in control_modes[:11]]
            cmd.right_hand_control_mode = [int(value) for value in control_modes[11:]]

        for values, left_name, right_name in (
            (target_velocities, "left_hand_velocities", "right_hand_velocities"),
            (kp, "left_hand_kp", "right_hand_kp"),
            (kd, "left_hand_kd", "right_hand_kd"),
            (torque_ff, "left_hand_torque_ff", "right_hand_torque_ff"),
            (output_limit, "left_hand_output_limit", "right_hand_output_limit"),
        ):
            if values is not None:
                setattr(cmd, left_name, [float(value) for value in values[:11]])
                setattr(cmd, right_name, [float(value) for value in values[11:]])

        self._pub.publish(cmd)

    def control_left(self, target_positions):
        if len(target_positions) != 11:
            raise ValueError("SG100 left-hand command must contain 11 values")
        self.control(list(target_positions) + [0.0] * 11, enable_left=True, enable_right=False)

    def control_right(self, target_positions):
        if len(target_positions) != 11:
            raise ValueError("SG100 right-hand command must contain 11 values")
        self.control([0.0] * 11 + list(target_positions), enable_left=False, enable_right=True)

# 使用示例
if __name__ == "__main__":
    from kuavo_deploy.config import load_kuavo_config
    
    # 使用上下文管理器确保资源正确释放
    with KuavoBaseRosEnv(load_kuavo_config()) as env:
        obs, info = env.reset()

        for _ in range(1):
            obs = env.get_obs()
            env.rate.sleep()
            print(obs.keys())
            for k, v in obs.items():
                print(k, v.shape)
                print(v.max(), v.min())
