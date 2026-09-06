# irmv_flat_env_aug_cfg.py
import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity import mdp

from .irmv_humanoid_locomotion_env_cfg import IrmvFlatEnvCfg
from .irmv_flat_env_aug import IrmvFlatEnvAug


def _safe_tanh(x: torch.Tensor, scale: float) -> torch.Tensor:
    return torch.tanh(x / scale)


def _soft_limit_penalty(
    q: torch.Tensor,
    limit: float,
) -> torch.Tensor:

    excess = torch.clamp(torch.abs(q) - limit, min=0.0)
    return torch.sum(excess ** 2, dim=1)

def reward_leg_antiphase_velocity(
    env,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:

    robot = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids

    vl = robot.data.joint_vel[:, ids[0]]
    vr = robot.data.joint_vel[:, ids[1]]

    mismatch = torch.tanh((vl + vr) / 2.0)

    return torch.exp(-(mismatch ** 2) / 0.35)


def reward_leg_activity(
    env,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:

    robot = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids

    vl = robot.data.joint_vel[:, ids[0]]
    vr = robot.data.joint_vel[:, ids[1]]

    speed2 = vl ** 2 + vr ** 2

    return 1.0 - torch.exp(-0.30 * speed2)


def reward_leg_phase_antiphase(
    env,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:

    robot = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids

    ql = robot.data.joint_pos[:, ids[0]]
    qr = robot.data.joint_pos[:, ids[1]]
    vl = robot.data.joint_vel[:, ids[0]]
    vr = robot.data.joint_vel[:, ids[1]]

    q_scale = 0.45
    v_scale = 2.0

    phase_l = torch.atan2(
        vl / v_scale,
        ql / q_scale + 1e-6,
    )

    phase_r = torch.atan2(
        vr / v_scale,
        qr / q_scale + 1e-6,
    )

    phase_diff = phase_l - phase_r
    reward = 0.5 * (1.0 - torch.cos(phase_diff))

    return reward


def reward_leg_knee_coordination(
    env,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:

    robot = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids

    hl = robot.data.joint_vel[:, ids[0]]
    hr = robot.data.joint_vel[:, ids[1]]

    kl = robot.data.joint_vel[:, ids[2]]
    kr = robot.data.joint_vel[:, ids[3]]

    err = (
        (kl + kr)
        + 0.20 * (hl + hr)
    )

    err = torch.tanh(err)

    return torch.exp(-(err ** 2) / 0.45)


def reward_hip_swing_amplitude(
    env,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:

    robot = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids

    ql = robot.data.joint_pos[:, ids[0]]
    qr = robot.data.joint_pos[:, ids[1]]

    amp = torch.abs(ql) + torch.abs(qr)

    return torch.tanh(amp / 0.75)

def reward_arm_position_symmetry(
    env,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:

    robot = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids

    ql = robot.data.joint_pos[:, ids[0]]
    qr = robot.data.joint_pos[:, ids[1]]

    error = torch.tanh((ql + qr) / 0.6)

    return torch.exp(-(error ** 2) / 0.12)


def reward_arm_velocity_antiphase(
    env,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:

    robot = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids

    vl = robot.data.joint_vel[:, ids[0]]
    vr = robot.data.joint_vel[:, ids[1]]

    mismatch = torch.tanh((vl + vr) / 1.5)

    return torch.exp(-(mismatch ** 2) / 0.35)


def reward_arm_activity(
    env,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:

    robot = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids

    vl = robot.data.joint_vel[:, ids[0]]
    vr = robot.data.joint_vel[:, ids[1]]

    speed2 = vl ** 2 + vr ** 2

    return 1.0 - torch.exp(-0.55 * speed2)


def reward_arm_centering_ema(
    env,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:

    robot = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids

    ql = robot.data.joint_pos[:, ids[0]]
    qr = robot.data.joint_pos[:, ids[1]]

    current = torch.stack(
        (ql, qr),
        dim=1,
    )

    if (
        not hasattr(env, "_arm_center_ema")
        or env._arm_center_ema is None
        or env._arm_center_ema.shape != current.shape
    ):
        env._arm_center_ema = current.clone()

    alpha = 0.025

    env._arm_center_ema = (
        (1.0 - alpha) * env._arm_center_ema
        + alpha * current
    )

    center_error = torch.sum(
        env._arm_center_ema ** 2,
        dim=1,
    )

    return torch.exp(
        -center_error / 0.08
    )

def reward_arm_swing_amplitude(
    env,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:

    robot = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids

    ql = robot.data.joint_pos[:, ids[0]]
    qr = robot.data.joint_pos[:, ids[1]]

    amplitude = (
        torch.abs(ql)
        + torch.abs(qr)
    )

    return torch.tanh(
        amplitude / 0.65
    )

def reward_cross_leg_arm_phase(
    env,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:

    robot = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids

    ls = robot.data.joint_vel[:, ids[0]]
    rs = robot.data.joint_vel[:, ids[1]]

    lh = robot.data.joint_vel[:, ids[2]]
    rh = robot.data.joint_vel[:, ids[3]]

    ls_n = torch.tanh(ls / 2.0)
    rs_n = torch.tanh(rs / 2.0)

    lh_n = torch.tanh(lh / 3.0)
    rh_n = torch.tanh(rh / 3.0)


    left_error = ls_n - 0.35 * rh_n
    right_error = rs_n - 0.35 * lh_n

    error = (
        left_error ** 2
        + right_error ** 2
    )

    velocity_match = torch.exp(
        -error / 0.35
    )

    return 0.75 + 0.25 * velocity_match

def penalty_arm_velocity_sq(
    env,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:

    robot = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids

    vl = robot.data.joint_vel[:, ids[0]]
    vr = robot.data.joint_vel[:, ids[1]]

    vel = torch.tanh(
        torch.stack(
            (vl, vr),
            dim=1,
        ) / 4.0
    )

    return torch.sum(
        vel ** 2,
        dim=1,
    )


def penalty_arm_position_offset(
    env,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:

    robot = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids

    ql = robot.data.joint_pos[:, ids[0]]
    qr = robot.data.joint_pos[:, ids[1]]

    return ql ** 2 + qr ** 2

def reward_hip_roll_center(
    env,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:

    robot = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids

    q = robot.data.joint_pos[:, ids]

    error = torch.sum(
        q ** 2,
        dim=1,
    )

    return torch.exp(
        -error / 0.08
    )


def penalty_hip_roll_soft_limit(
    env,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:

    robot = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids

    q = robot.data.joint_pos[:, ids]

    limit = 0.10

    return _soft_limit_penalty(
        q,
        limit,
    )


def penalty_hip_roll_velocity(
    env,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:

    robot = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids

    v = robot.data.joint_vel[:, ids]

    v = torch.tanh(
        v / 3.0
    )

    return torch.sum(
        v ** 2,
        dim=1,
    )

def reward_hip_yaw_symmetry(
    env,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:

    robot = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids

    ly = robot.data.joint_pos[:, ids[0]]
    ry = robot.data.joint_pos[:, ids[1]]

    error = torch.tanh(
        (ly + ry) / 0.30
    )

    return torch.exp(
        -(error ** 2) / 0.10
    )


def penalty_hip_yaw_soft_limit(
    env,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:

    robot = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids

    q = robot.data.joint_pos[:, ids]

    limit = 0.10

    return _soft_limit_penalty(
        q,
        limit,
    )


def penalty_hip_yaw_velocity(
    env,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:

    robot = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids

    v = robot.data.joint_vel[:, ids]

    v = torch.tanh(
        v / 2.0
    )

    return torch.sum(
        v ** 2,
        dim=1,
    )

def reward_ankle_roll_center(
    env,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:

    robot = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids

    q = robot.data.joint_pos[:, ids]

    error = torch.sum(
        q ** 2,
        dim=1,
    )

    return torch.exp(
        -error / 0.025
    )


def penalty_ankle_roll_soft_limit(
    env,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:

    robot = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids

    q = robot.data.joint_pos[:, ids]

    limit = 0.07

    return _soft_limit_penalty(
        q,
        limit,
    )


def penalty_ankle_roll_velocity(
    env,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:

    robot = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids

    v = robot.data.joint_vel[:, ids]

    v = torch.tanh(
        v / 3.0
    )

    return torch.sum(
        v ** 2,
        dim=1,
    )

def reward_knee_symmetry(
    env,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:

    robot = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids

    kl = robot.data.joint_pos[:, ids[0]]
    kr = robot.data.joint_pos[:, ids[1]]

    error = torch.tanh(
        (kl - kr) / 0.5
    )

    return torch.exp(
        -(error ** 2) / 0.15
    )

def reward_min_height(env,asset_cfg: SceneEntityCfg,) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    height = robot.data.root_pos_w[:, 2]
    h_min = 0.74
    delta = torch.clamp(
        h_min - height,
        min=0.0,
    )
    return torch.exp(
        -5.0 * delta
    )


@configclass
class IrmvFlatEnvAugCfg(IrmvFlatEnvCfg):

    env_class = IrmvFlatEnvAug

    def __post_init__(self):

        super().__post_init__()
        self.commands.base_velocity.ranges.lin_vel_x = (2.2,2.8)
        old_rewards = [
            "same_side_antiphase",
            "same_side_inphase_penalty",
            "arm_leg_velocity_antiphase",
            "debug_right_arm",
            "arm_center",
            "arm_position_penalty",
            "antiphase_pos_vel",
            "cross_antiphase",
            "cross_antiphase_vel",
            "arm_antiphase_vel",
            "arm_activity",

        ]

        for attr in old_rewards:

            if hasattr(
                self.rewards,
                attr,
            ):
                delattr(
                    self.rewards,
                    attr,
                )

        self.rewards.leg_antiphase_vel = mdp.RewardTermCfg(

            func=reward_leg_antiphase_velocity,

            weight=0.30,

            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "left_hip_pitch_joint",
                        "right_hip_pitch_joint",
                    ],
                )
            },
        )

        self.rewards.leg_activity = mdp.RewardTermCfg(

            func=reward_leg_activity,

            weight=0.10,

            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "left_hip_pitch_joint",
                        "right_hip_pitch_joint",
                    ],
                )
            },
        )

        self.rewards.leg_phase_antiphase = mdp.RewardTermCfg(

            func=reward_leg_phase_antiphase,

            weight=0.12,

            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "left_hip_pitch_joint",
                        "right_hip_pitch_joint",
                    ],
                )
            },
        )

        self.rewards.leg_knee_coordination = mdp.RewardTermCfg(

            func=reward_leg_knee_coordination,

            weight=0.08,

            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "left_hip_pitch_joint",
                        "right_hip_pitch_joint",
                        "left_knee_joint",
                        "right_knee_joint",
                    ],
                )
            },
        )
        self.rewards.hip_swing_amplitude = mdp.RewardTermCfg(

            func=reward_hip_swing_amplitude,

            weight=0.30,

            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "left_hip_pitch_joint",
                        "right_hip_pitch_joint",
                    ],
                )
            },
        )
        self.rewards.arm_position_symmetry = mdp.RewardTermCfg(

            func=reward_arm_position_symmetry,

            weight=0.50,

            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "left_shoulder_pitch_joint",
                        "right_shoulder_pitch_joint",
                    ],
                )
            },
        )
        self.rewards.arm_velocity_antiphase = mdp.RewardTermCfg(

            func=reward_arm_velocity_antiphase,

            weight=0.30,

            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "left_shoulder_pitch_joint",
                        "right_shoulder_pitch_joint",
                    ],
                )
            },
        )
        self.rewards.arm_activity = mdp.RewardTermCfg(

            func=reward_arm_activity,

            weight=0.80,

            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "left_shoulder_pitch_joint",
                        "right_shoulder_pitch_joint",
                    ],
                )
            },
        )
        self.rewards.arm_swing_amplitude = mdp.RewardTermCfg(

            func=reward_arm_swing_amplitude,

            weight=0.30,

            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "left_shoulder_pitch_joint",
                        "right_shoulder_pitch_joint",
                    ],
                )
            },
        )
        self.rewards.arm_centering_ema = mdp.RewardTermCfg(

            func=reward_arm_centering_ema,

            weight=1.20,

            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "left_shoulder_pitch_joint",
                        "right_shoulder_pitch_joint",
                    ],
                )
            },
        )
        self.rewards.cross_leg_arm_phase = mdp.RewardTermCfg(

            func=reward_cross_leg_arm_phase,

            weight=0.45,

            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "left_shoulder_pitch_joint",
                        "right_shoulder_pitch_joint",
                        "left_hip_pitch_joint",
                        "right_hip_pitch_joint",
                    ],
                )
            },
        )
        self.rewards.arm_offset_penalty = mdp.RewardTermCfg(

            func=penalty_arm_position_offset,

            weight=-0.01,

            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "left_shoulder_pitch_joint",
                        "right_shoulder_pitch_joint",
                    ],
                )
            },
        )
        self.rewards.arm_velocity_penalty = mdp.RewardTermCfg(

            func=penalty_arm_velocity_sq,

            weight=-0.001,

            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "left_shoulder_pitch_joint",
                        "right_shoulder_pitch_joint",
                    ],
                )
            },
        )

        self.rewards.hip_roll_center = mdp.RewardTermCfg(

            func=reward_hip_roll_center,

            weight=0.80,

            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "left_hip_roll_joint",
                        "right_hip_roll_joint",
                    ],
                )
            },
        )


        self.rewards.hip_roll_soft_limit = mdp.RewardTermCfg(

            func=penalty_hip_roll_soft_limit,

            weight=-8.0,

            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "left_hip_roll_joint",
                        "right_hip_roll_joint",
                    ],
                )
            },
        )


        self.rewards.hip_roll_velocity_regularization = mdp.RewardTermCfg(

            func=penalty_hip_roll_velocity,

            weight=-0.01,

            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "left_hip_roll_joint",
                        "right_hip_roll_joint",
                    ],
                )
            },
        )
        self.rewards.hip_yaw_symmetry = mdp.RewardTermCfg(

            func=reward_hip_yaw_symmetry,

            weight=1.0,

            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "left_hip_yaw_joint",
                        "right_hip_yaw_joint",
                    ],
                )
            },
        )


        self.rewards.hip_yaw_soft_limit = mdp.RewardTermCfg(

            func=penalty_hip_yaw_soft_limit,

            weight=-8.0,

            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "left_hip_yaw_joint",
                        "right_hip_yaw_joint",
                    ],
                )
            },
        )


        self.rewards.hip_yaw_velocity_penalty = mdp.RewardTermCfg(

            func=penalty_hip_yaw_velocity,

            weight=-0.02,

            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "left_hip_yaw_joint",
                        "right_hip_yaw_joint",
                    ],
                )
            },
        )
        self.rewards.ankle_roll_center = mdp.RewardTermCfg(

            func=reward_ankle_roll_center,

            weight=0.50,

            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "left_ankle_roll_joint",
                        "right_ankle_roll_joint",
                    ],
                )
            },
        )


        self.rewards.ankle_roll_soft_limit = mdp.RewardTermCfg(

            func=penalty_ankle_roll_soft_limit,

            weight=-10.0,

            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "left_ankle_roll_joint",
                        "right_ankle_roll_joint",
                    ],
                )
            },
        )


        self.rewards.ankle_roll_velocity_regularization = mdp.RewardTermCfg(

            func=penalty_ankle_roll_velocity,

            weight=-0.01,

            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "left_ankle_roll_joint",
                        "right_ankle_roll_joint",
                    ],
                )
            },
        )

        self.rewards.knee_symmetry = mdp.RewardTermCfg(

            func=reward_knee_symmetry,

            weight=0.40,

            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "left_knee_joint",
                        "right_knee_joint",
                    ],
                )
            },
        )
        self.rewards.min_height = mdp.RewardTermCfg(

            func=reward_min_height,

            weight=1.50,

            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                )
            },
        )
        if hasattr(
            self.rewards,
            "foot_yaw_alignment",
        ):
            delattr(
                self.rewards,
                "foot_yaw_alignment",
            )
        if hasattr(
            self.rewards,
            "track_lin_vel_xy_exp",
        ):
            self.rewards.track_lin_vel_xy_exp.weight = 8.0 

        if hasattr(
            self.rewards,
            "ang_vel_xy_l2",
        ):
            self.rewards.ang_vel_xy_l2.weight = -2.0


        if hasattr(
            self.rewards,
            "base_height_l2",
        ):
            self.rewards.base_height_l2.weight = -0.5


        if hasattr(
            self.rewards,
            "flat_orientation_l2",
        ):
            self.rewards.flat_orientation_l2.weight = -1.0
        if hasattr(
            self.rewards,
            "dof_pos_limits_shoulder",
        ):
            self.rewards.dof_pos_limits_shoulder.weight = -2.0
