import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity import mdp
from .irmv_humanoid_locomotion_env_cfg import IrmvFlatEnvCfg

def reward_arm_leg_velocity_antiphase(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    joint_vel = robot.data.joint_vel
    joint_ids = asset_cfg.joint_ids
    left_shoulder, right_shoulder, left_hip, right_hip = joint_ids[:4]
    v_ls = joint_vel[:, left_shoulder]
    v_rs = joint_vel[:, right_shoulder]
    v_lh = joint_vel[:, left_hip]
    v_rh = joint_vel[:, right_hip]
    prod_l = v_ls * v_lh
    prod_r = v_rs * v_rh
    reward = 0.5 * (1 + torch.tanh(-(prod_l + prod_r)))
    return reward

def penalty_arm_velocity(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    joint_vel = robot.data.joint_vel
    joint_ids = asset_cfg.joint_ids
    shoulder_ids = joint_ids[:2]
    vel = joint_vel[:, shoulder_ids]
    return torch.sum(vel**2, dim=1)

def penalty_arm_position(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    joint_pos = robot.data.joint_pos
    joint_ids = asset_cfg.joint_ids
    left_shoulder, right_shoulder = joint_ids[:2]
    ls = joint_pos[:, left_shoulder]
    rs = joint_pos[:, right_shoulder]
    return (ls**2 + rs**2) 

def reward_arm_center(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    joint_pos = robot.data.joint_pos
    joint_ids = asset_cfg.joint_ids
    left_shoulder, right_shoulder = joint_ids[:2]
    ls = joint_pos[:, left_shoulder]
    rs = joint_pos[:, right_shoulder]
    sigma = 0.5  
    reward_l = torch.exp(-0.5 * (ls**2) / (sigma**2))
    reward_r = torch.exp(-0.5 * (rs**2) / (sigma**2))
    return (reward_l + reward_r) / 2.0



def debug_right_arm(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    joint_pos = robot.data.joint_pos
    joint_vel = robot.data.joint_vel
    joint_ids = asset_cfg.joint_ids
    left_id = joint_ids[0]
    right_id = joint_ids[1]
    if env.common_step_counter % 100 == 0:
        angl0 = joint_pos[0, left_id].item()
        ve0 = joint_vel[0, left_id].item()
        angle = joint_pos[0, right_id].item()
        vel = joint_vel[0, right_id].item()
        #print(f"[Step {env.common_step_counter}] Left shoulder: angle={angl0:.3f} rad, vel={ve0:.3f} rad/s")
        #print(f"[Step {env.common_step_counter}] Right shoulder: angle={angle:.3f} rad, vel={vel:.3f} rad/s")
    return torch.zeros(env.num_envs, device=env.device)


@configclass
class IrmvFlatEnvArmCfg(IrmvFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.commands.base_velocity.ranges.lin_vel_x = (0.3, 1.0)
        if hasattr(self.rewards, 'track_lin_vel_xy_exp'):
            self.rewards.track_lin_vel_xy_exp.weight = 1.5
            self.rewards.track_lin_vel_xy_exp.params["std"] = 1.0
        if hasattr(self.rewards, 'track_ang_vel_z_exp'):
            self.rewards.track_ang_vel_z_exp.weight = 2.0

        if hasattr(self.rewards, 'flat_orientation_l2'):
            self.rewards.flat_orientation_l2.weight = -3.0

        if hasattr(self.rewards, 'ang_vel_xy_l2'):
            self.rewards.ang_vel_xy_l2.weight = -2.0

        if hasattr(self.rewards, 'feet_air_time'):
            self.rewards.feet_air_time.weight = 0.2
            self.rewards.feet_air_time.params["threshold"] = 0.8
        if hasattr(self.rewards, 'feet_slide'):
            self.rewards.feet_slide.weight = -0.5

        if hasattr(self, 'terminations') and hasattr(self.terminations, 'base_contact'):
            self.terminations.base_contact.params["sensor_cfg"].body_names = ["torso_link"]

        if hasattr(self.rewards, 'dof_pos_limits'):
            self.rewards.dof_pos_limits.weight = -0.5

        self.rewards.dof_pos_limits_shoulder = mdp.RewardTermCfg(
            func=mdp.joint_pos_limits,
            weight=-20.0,  
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[
                "left_shoulder_pitch_joint", "right_shoulder_pitch_joint"
            ])}
        )

        if hasattr(self, 'events') and hasattr(self.events, 'reset_robot_joints'):
            if 'position_range' in self.events.reset_robot_joints.params:
                self.events.reset_robot_joints.params["position_range"] = (0.8, 1.2)
        self.rewards.arm_leg_velocity_antiphase = mdp.RewardTermCfg(
            func=reward_arm_leg_velocity_antiphase,
            weight=0.1,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "left_shoulder_pitch_joint",
                        "right_shoulder_pitch_joint",
                        "left_hip_pitch_joint",
                        "right_hip_pitch_joint"
                    ]
                )
            },
        )
        self.rewards.arm_velocity_penalty = mdp.RewardTermCfg(
            func=penalty_arm_velocity,
            weight=-0.02,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "left_shoulder_pitch_joint",
                        "right_shoulder_pitch_joint"
                    ]
                )
            },
        )
        self.rewards.arm_position_penalty = mdp.RewardTermCfg(
            func=penalty_arm_position,
            weight=-0.5,  
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "left_shoulder_pitch_joint",
                        "right_shoulder_pitch_joint"
                    ]
                )
            },
        )
        self.rewards.arm_center = mdp.RewardTermCfg(
            func=reward_arm_center,
            weight=0.5,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "left_shoulder_pitch_joint",
                        "right_shoulder_pitch_joint"
                    ]
                )
            },
        )
        self.rewards.debug_right_arm = mdp.RewardTermCfg(
            func=debug_right_arm,
            weight=1e-6,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=["left_shoulder_pitch_joint", "right_shoulder_pitch_joint"]
                )
            },
        )
