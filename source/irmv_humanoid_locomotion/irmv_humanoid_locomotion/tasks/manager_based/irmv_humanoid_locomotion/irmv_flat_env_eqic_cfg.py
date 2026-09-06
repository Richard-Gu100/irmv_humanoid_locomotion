# irmv_humanoid_locomotion_env_eqic_cfg.py
import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity import mdp
from .irmv_humanoid_locomotion_env_arm_cfg import IrmvFlatEnvArmCfg

from .irmv_humanoid_locomotion_env_enhancedarm_cfg import (
    reward_leg_antiphase_velocity,
    reward_leg_activity,
    reward_leg_knee_coordination,
    reward_leg_phase_antiphase,
    reward_hip_swing_amplitude,
    reward_arm_velocity_antiphase,
    reward_cross_leg_arm_phase,
    reward_arm_activity,
    penalty_arm_velocity_sq,
    reward_min_height,
    penalty_hip_roll_velocity,
    penalty_hip_yaw_velocity,
)


@configclass
class IrmvFlatEnvEqicCfg(IrmvFlatEnvArmCfg):
    def __post_init__(self):
        super().__post_init__()

        self.commands.base_velocity.ranges.lin_vel_x = (0.2, 0.5)
        symmetry_rewards = [
            "arm_position_symmetry",
            "arm_offset_penalty",
            "arm_centering_ema",
            "hip_yaw_symmetry",
            "hip_roll_center",
            "knee_symmetry",
            "foot_yaw_alignment",
        ]
        for attr in symmetry_rewards:
            if hasattr(self.rewards, attr):
                delattr(self.rewards, attr)

        self.rewards.leg_antiphase_vel = mdp.RewardTermCfg(
            func=reward_leg_antiphase_velocity,
            weight=0.30,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["left_hip_pitch_joint", "right_hip_pitch_joint"])},
        )
        self.rewards.leg_activity = mdp.RewardTermCfg(
            func=reward_leg_activity,
            weight=0.10,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["left_hip_pitch_joint", "right_hip_pitch_joint"])},
        )
        self.rewards.leg_phase_antiphase = mdp.RewardTermCfg(
            func=reward_leg_phase_antiphase,
            weight=0.12,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["left_hip_pitch_joint", "right_hip_pitch_joint"])},
        )
        self.rewards.leg_knee_coordination = mdp.RewardTermCfg(
            func=reward_leg_knee_coordination,
            weight=0.08,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[
                "left_hip_pitch_joint", "right_hip_pitch_joint",
                "left_knee_joint", "right_knee_joint"
            ])},
        )

        self.rewards.hip_swing_amplitude = mdp.RewardTermCfg(
            func=reward_hip_swing_amplitude,
            weight=1.0,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["left_hip_pitch_joint", "right_hip_pitch_joint"])},
        )


        self.rewards.arm_velocity_antiphase = mdp.RewardTermCfg(
            func=reward_arm_velocity_antiphase,
            weight=0.01,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["left_shoulder_pitch_joint", "right_shoulder_pitch_joint"])},
        )
        self.rewards.cross_leg_arm_phase = mdp.RewardTermCfg(
            func=reward_cross_leg_arm_phase,
            weight=0.01,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[
                "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
                "left_hip_pitch_joint", "right_hip_pitch_joint"
            ])},
        )
        self.rewards.arm_center.weight = 5.0
        
        self.rewards.arm_activity = mdp.RewardTermCfg(
            func=reward_arm_activity,
            weight=0.01,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["left_shoulder_pitch_joint", "right_shoulder_pitch_joint"])},
        )
        self.rewards.arm_velocity_penalty = mdp.RewardTermCfg(
            func=penalty_arm_velocity_sq,
            weight=-0.001,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["left_shoulder_pitch_joint", "right_shoulder_pitch_joint"])},
        )

        self.rewards.min_height = mdp.RewardTermCfg(
            func=reward_min_height,
            weight=2.0,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )


        self.rewards.hip_roll_velocity_regularization = mdp.RewardTermCfg(
            func=penalty_hip_roll_velocity,
            weight=-0.015,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["left_hip_roll_joint", "right_hip_roll_joint"])},
        )
        self.rewards.hip_yaw_velocity_penalty = mdp.RewardTermCfg(
            func=penalty_hip_yaw_velocity,
            weight=-0.01,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["left_hip_yaw_joint", "right_hip_yaw_joint"])},
        )

        if hasattr(self.rewards, "track_lin_vel_xy_exp"):
            self.rewards.track_lin_vel_xy_exp.weight = 12.0   
        if hasattr(self.rewards, "ang_vel_xy_l2"):
            self.rewards.ang_vel_xy_l2.weight = -0.2         
        if hasattr(self.rewards, "base_height_l2"):
            self.rewards.base_height_l2.weight = -1.0        
        if hasattr(self.rewards, "flat_orientation_l2"):
            self.rewards.flat_orientation_l2.weight = -3.0   

        if hasattr(self.rewards, "feet_slide"):
            self.rewards.feet_slide.weight = -0.5           

        if hasattr(self.terminations, "root_height_below_minimum"):
            self.terminations.root_height_below_minimum.params["minimum_height"] = 0.2   

        if hasattr(self.rewards, "dof_acc_l2"):
            self.rewards.dof_acc_l2.weight = -1e-6
        if hasattr(self.rewards, "action_rate_l2"):
            self.rewards.action_rate_l2.weight = -0.02
