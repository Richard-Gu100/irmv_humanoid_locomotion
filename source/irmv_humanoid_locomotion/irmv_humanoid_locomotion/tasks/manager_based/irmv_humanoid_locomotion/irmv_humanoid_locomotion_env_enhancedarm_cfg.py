import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity import mdp
from .irmv_humanoid_locomotion_env_arm_cfg import IrmvFlatEnvArmCfg


def reward_leg_antiphase_velocity(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids
    vl = robot.data.joint_vel[:, ids[0]]
    vr = robot.data.joint_vel[:, ids[1]]
    mismatch = torch.tanh(vl + vr)
    return torch.exp(-(mismatch * mismatch) / 0.35)

def reward_leg_activity(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids
    vl = robot.data.joint_vel[:, ids[0]]
    vr = robot.data.joint_vel[:, ids[1]]
    speed2 = vl * vl + vr * vr
    return 1.0 - torch.exp(-0.35 * speed2)

def reward_leg_knee_coordination(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids  
    hl = robot.data.joint_vel[:, ids[0]]
    hr = robot.data.joint_vel[:, ids[1]]
    kl = robot.data.joint_vel[:, ids[2]]
    kr = robot.data.joint_vel[:, ids[3]]
    err = (kl + kr) + 0.25 * (hl + hr)
    return torch.exp(-(torch.tanh(err) ** 2) / 0.45)


def gait_phase_signal(env, asset_cfg: SceneEntityCfg):
    robot = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids
    ql = robot.data.joint_pos[:, ids[0]]
    qr = robot.data.joint_pos[:, ids[1]]
    vl = robot.data.joint_vel[:, ids[0]]
    vr = robot.data.joint_vel[:, ids[1]]
    q_scale, v_scale = 0.45, 2.0
    phase_l = torch.atan2(vl / v_scale, ql / q_scale + 1e-6)
    phase_r = torch.atan2(vr / v_scale, qr / q_scale + 1e-6)
    return phase_l, phase_r

def reward_leg_phase_antiphase(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    phase_l, phase_r = gait_phase_signal(env, asset_cfg)
    phase_diff = phase_l - phase_r
    phase_error = 1.0 + torch.cos(phase_diff)
    return torch.exp(-2.0 * phase_error)


def reward_hip_swing_amplitude(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids  # [left_hip_pitch, right_hip_pitch]
    ql = robot.data.joint_pos[:, ids[0]]
    qr = robot.data.joint_pos[:, ids[1]]
    amp = torch.abs(ql) + torch.abs(qr)  
    return torch.tanh(amp / 0.8)  
    
def reward_arm_position_symmetry(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids
    ql = robot.data.joint_pos[:, ids[0]]
    qr = robot.data.joint_pos[:, ids[1]]
    sym = torch.tanh(ql + qr)
    return torch.exp(-(sym * sym) / 0.08)

def reward_arm_velocity_antiphase(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids
    vl = robot.data.joint_vel[:, ids[0]]
    vr = robot.data.joint_vel[:, ids[1]]
    mismatch = torch.tanh((vl + vr) / 1.5)
    return torch.exp(-(mismatch * mismatch) / 0.30)
    
def reward_cross_leg_arm_phase(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids          
    ls = robot.data.joint_vel[:, joint_ids[0]]
    rs = robot.data.joint_vel[:, joint_ids[1]]
    lh = robot.data.joint_vel[:, joint_ids[2]]
    rh = robot.data.joint_vel[:, joint_ids[3]]

    ql = robot.data.joint_pos[:, joint_ids[2]]
    qr = robot.data.joint_pos[:, joint_ids[3]]

    CROSS_SIGN_LA = 1.0   
    CROSS_SIGN_RA = 1.0   
    k = 0.45


    ls_n = torch.tanh(ls / 2.0)
    rs_n = torch.tanh(rs / 2.0)
    lh_n = torch.tanh(lh / 3.0)
    rh_n = torch.tanh(rh / 3.0)

    err_left = ls_n - CROSS_SIGN_LA * k * rh_n
    err_right = rs_n - CROSS_SIGN_RA * k * lh_n
    velocity_match = torch.exp(-(err_left * err_left + err_right * err_right) / 0.28)

    q_scale = 0.45
    v_scale = 2.0
    phase_l = torch.atan2(lh / v_scale, ql / q_scale + 1e-6)
    phase_r = torch.atan2(rh / v_scale, qr / q_scale + 1e-6)

    arm_phase_l = torch.atan2(ls / 2.0, torch.ones_like(ls))   
    arm_phase_r = torch.atan2(rs / 2.0, torch.ones_like(rs))

    phase_cross = (
        0.5 * (1.0 + torch.cos(arm_phase_r - phase_l)) +
        0.5 * (1.0 + torch.cos(arm_phase_l - phase_r))
    ) * 0.5

    return velocity_match * (0.70 + 0.30 * phase_cross)
    
def penalty_arm_position_offset(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids
    ql = robot.data.joint_pos[:, ids[0]]
    qr = robot.data.joint_pos[:, ids[1]]
    return ql**2 + qr**2

def penalty_arm_velocity_sq(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids         
    vl = robot.data.joint_vel[:, joint_ids[0]]
    vr = robot.data.joint_vel[:, joint_ids[1]]
    vel = torch.tanh(torch.stack((vl, vr), dim=1) / 4.0)
    return torch.sum(vel * vel, dim=1)

def reward_arm_activity(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids          
    vl = robot.data.joint_vel[:, joint_ids[0]]
    vr = robot.data.joint_vel[:, joint_ids[1]]
    speed2 = vl * vl + vr * vr
    return 1.0 - torch.exp(-0.8 * speed2)

def reward_arm_centering_ema(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids
    ql = robot.data.joint_pos[:, ids[0]]
    qr = robot.data.joint_pos[:, ids[1]]
    current = torch.stack((ql, qr), dim=1)   # (num_envs, 2)

    if not hasattr(env, "_arm_center_ema") or env._arm_center_ema is None:
        env._arm_center_ema = current.clone()
    else:
        if env._arm_center_ema.shape != current.shape:
            env._arm_center_ema = current.clone()

    alpha = 0.04  
    env._arm_center_ema = (1.0 - alpha) * env._arm_center_ema + alpha * current

    center_error = torch.sum(env._arm_center_ema ** 2, dim=1)
    return torch.exp(-center_error / 0.06) 

def reward_hip_yaw_symmetry(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids  # [left_hip_yaw, right_hip_yaw]
    ly = robot.data.joint_pos[:, ids[0]]
    ry = robot.data.joint_pos[:, ids[1]]
    sym = ly + ry
    return torch.exp(-(sym ** 2) / 0.04)   

def reward_hip_roll_center(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids  # [left_hip_roll, right_hip_roll]
    lr = robot.data.joint_pos[:, ids[0]]
    rr = robot.data.joint_pos[:, ids[1]]
    sq_error = lr**2 + rr**2
    return torch.exp(-sq_error / 0.05)   

def reward_knee_symmetry(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids  # [left_knee, right_knee]
    kl = robot.data.joint_pos[:, ids[0]]
    kr = robot.data.joint_pos[:, ids[1]]
    diff = kl - kr
    return torch.exp(-(diff**2) / 0.06)

def reward_foot_yaw_alignment(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    try:
        left_foot_idx = robot.find_bodies("left_ankle_roll_link")[0]
        right_foot_idx = robot.find_bodies("right_ankle_roll_link")[0]
        left_rot = robot.data.body_quat_w[:, left_foot_idx, :]  # (N, 4)
        right_rot = robot.data.body_quat_w[:, right_foot_idx, :]
        def quat_to_yaw(q):
            w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
            return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        yaw_l = quat_to_yaw(left_rot)
        yaw_r = quat_to_yaw(right_rot)

        yaw_cost = yaw_l**2 + yaw_r**2
        return torch.exp(-yaw_cost / 0.05)
    except (IndexError, ValueError):
        return torch.ones(env.num_envs, device=env.device)


def reward_min_height(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    height = robot.data.root_pos_w[:, 2]
    h_min = 0.74
    delta = torch.clamp(h_min - height, min=0.0)
    return torch.exp(-5.0 * delta)

def penalty_arm_velocity_sq(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids
    vl = robot.data.joint_vel[:, ids[0]]
    vr = robot.data.joint_vel[:, ids[1]]
    vel = torch.tanh(torch.stack((vl, vr), dim=1) / 4.0)
    return torch.sum(vel * vel, dim=1)

def penalty_hip_roll_velocity(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids
    vl = robot.data.joint_vel[:, ids[0]]
    vr = robot.data.joint_vel[:, ids[1]]
    vel = torch.tanh(torch.stack((vl, vr), dim=1) / 3.0)
    return torch.sum(vel * vel, dim=1)

def penalty_hip_yaw_velocity(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids
    vl = robot.data.joint_vel[:, ids[0]]
    vr = robot.data.joint_vel[:, ids[1]]
    vel = torch.tanh(torch.stack((vl, vr), dim=1) / 2.0)
    return torch.sum(vel * vel, dim=1)


# =============================================================================
# 配置类
# =============================================================================
@configclass
class IrmvFlatEnvEnhancedarmCfg(IrmvFlatEnvArmCfg):
    def __post_init__(self):
        super().__post_init__()

        self.commands.base_velocity.ranges.lin_vel_x = (2.2, 2.8)
        old_rewards = [
            "same_side_antiphase", "same_side_inphase_penalty",
            "arm_leg_velocity_antiphase", "debug_right_arm",
            "arm_center", "arm_position_penalty",
            "antiphase_pos_vel", "cross_antiphase",
            "cross_antiphase_vel", "arm_antiphase_vel",
            "arm_activity",
        ]
        for attr in old_rewards:
            if hasattr(self.rewards, attr):
                delattr(self.rewards, attr)
        self.rewards.leg_antiphase_vel = mdp.RewardTermCfg(
            func=reward_leg_antiphase_velocity,
            weight=0.30,   
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["left_hip_pitch_joint", "right_hip_pitch_joint"])}
        )
        self.rewards.leg_activity = mdp.RewardTermCfg(
            func=reward_leg_activity,
            weight=0.10,   
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["left_hip_pitch_joint", "right_hip_pitch_joint"])}
        )
        self.rewards.leg_phase_antiphase = mdp.RewardTermCfg(
            func=reward_leg_phase_antiphase,
            weight=0.12,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["left_hip_pitch_joint", "right_hip_pitch_joint"])}
        )
        self.rewards.leg_knee_coordination = mdp.RewardTermCfg(
            func=reward_leg_knee_coordination,
            weight=0.08,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[
                "left_hip_pitch_joint", "right_hip_pitch_joint",
                "left_knee_joint", "right_knee_joint"
            ])}
        )
        self.rewards.hip_swing_amplitude = mdp.RewardTermCfg(
            func=reward_hip_swing_amplitude,
            weight=0.40,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["left_hip_pitch_joint", "right_hip_pitch_joint"])}
        )
        self.rewards.arm_position_symmetry = mdp.RewardTermCfg(
            func=reward_arm_position_symmetry,
            weight=0.40,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["left_shoulder_pitch_joint", "right_shoulder_pitch_joint"])}
        )
        self.rewards.arm_velocity_antiphase = mdp.RewardTermCfg(
            func=reward_arm_velocity_antiphase,
            weight=0.25,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["left_shoulder_pitch_joint", "right_shoulder_pitch_joint"])}
        )
        self.rewards.cross_leg_arm_phase = mdp.RewardTermCfg(
            func=reward_cross_leg_arm_phase,
            weight=0.85,
            params={"asset_cfg": SceneEntityCfg(
                "robot", joint_names=[
                    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
                    "left_hip_pitch_joint", "right_hip_pitch_joint"
                ]
            )}
        )
        self.rewards.arm_activity = mdp.RewardTermCfg(
            func=reward_arm_activity,
            weight=0.50,
            params={"asset_cfg": SceneEntityCfg(
                "robot", joint_names=["left_shoulder_pitch_joint", "right_shoulder_pitch_joint"]
            )}
        )
        if hasattr(self.rewards, "arm_velocity_penalty"):
            self.rewards.arm_velocity_penalty.weight = -0.002
        else:
            self.rewards.arm_velocity_penalty = mdp.RewardTermCfg(
                func=penalty_arm_velocity_sq,
                weight=-0.002,
                params={"asset_cfg": SceneEntityCfg(
                    "robot", joint_names=["left_shoulder_pitch_joint", "right_shoulder_pitch_joint"]
                )}
            )
        self.rewards.arm_offset_penalty = mdp.RewardTermCfg(
            func=penalty_arm_position_offset,
            weight=-1.0,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["left_shoulder_pitch_joint", "right_shoulder_pitch_joint"])}
        )
        self.rewards.arm_centering_ema = mdp.RewardTermCfg(
            func=reward_arm_centering_ema,
            weight=3.0,   
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["left_shoulder_pitch_joint", "right_shoulder_pitch_joint"])}
        )

        self.rewards.hip_yaw_symmetry = mdp.RewardTermCfg(
            func=reward_hip_yaw_symmetry,
            weight=3.0,  
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["left_hip_yaw_joint", "right_hip_yaw_joint"])}
        )

        self.rewards.hip_roll_center = mdp.RewardTermCfg(
            func=reward_hip_roll_center,
            weight=2.0,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["left_hip_roll_joint", "right_hip_roll_joint"])}
        )

        self.rewards.knee_symmetry = mdp.RewardTermCfg(
            func=reward_knee_symmetry,
            weight=0.6,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["left_knee_joint", "right_knee_joint"])}
        )

        self.rewards.foot_yaw_alignment = mdp.RewardTermCfg(
            func=reward_foot_yaw_alignment,
            weight=1.5,
            params={"asset_cfg": SceneEntityCfg("robot")}
        )

        self.rewards.hip_yaw_velocity_penalty = mdp.RewardTermCfg(
            func=penalty_hip_yaw_velocity,
            weight=-0.03,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["left_hip_yaw_joint", "right_hip_yaw_joint"])}
        )

        self.rewards.min_height = mdp.RewardTermCfg(
            func=reward_min_height,
            weight=2.0,
            params={"asset_cfg": SceneEntityCfg("robot")}
        )
        if hasattr(self.rewards, "arm_velocity_penalty"):
            self.rewards.arm_velocity_penalty.weight = -0.001
        else:
            self.rewards.arm_velocity_penalty = mdp.RewardTermCfg(
                func=penalty_arm_velocity_sq,
                weight=-0.001,
                params={"asset_cfg": SceneEntityCfg("robot", joint_names=["left_shoulder_pitch_joint", "right_shoulder_pitch_joint"])}
            )
        if hasattr(self.rewards, "dof_pos_limits_shoulder"):
            self.rewards.dof_pos_limits_shoulder.weight = -2.0
        self.rewards.hip_roll_velocity_regularization = mdp.RewardTermCfg(
            func=penalty_hip_roll_velocity,
            weight=-0.015,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["left_hip_roll_joint", "right_hip_roll_joint"])}
        )


        if hasattr(self.rewards, "track_lin_vel_xy_exp"):
            self.rewards.track_lin_vel_xy_exp.weight = 3.0
        if hasattr(self.rewards, "ang_vel_xy_l2"):
            self.rewards.ang_vel_xy_l2.weight = -2.5
        if hasattr(self.rewards, "base_height_l2"):
            self.rewards.base_height_l2.weight = -0.5
        if hasattr(self.rewards, "flat_orientation_l2"):
            self.rewards.flat_orientation_l2.weight = -1.0

