import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from irmv_humanoid_locomotion.assets import ASSET_DIR

ARMATURE_8520 = 0.125579552
ARMATURE_6020 = 0.0358060596

NATURAL_FREQ = 10 * 2.0 * math.pi  # 10 Hz
DAMPING_RATIO = 2.0

STIFFNESS_8520 = ARMATURE_8520 * NATURAL_FREQ**2
STIFFNESS_6020 = ARMATURE_6020 * NATURAL_FREQ**2

DAMPING_8520 = 2.0 * DAMPING_RATIO * ARMATURE_8520 * NATURAL_FREQ
DAMPING_6020 = 2.0 * DAMPING_RATIO * ARMATURE_6020 * NATURAL_FREQ

IRMV_V3_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=False,
        asset_path=f"{ASSET_DIR}/irmv_v3_description/urdf/irmv_v3.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.75),
        joint_pos={
            ".*_hip_pitch_joint": -0.3,
            ".*_knee_joint": 0.6,
            ".*_ankle_pitch_joint": -0.3,
            "left_shoulder_pitch_joint": 0.00,
            "right_shoulder_pitch_joint": 0.00,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim=50.0,
            velocity_limit_sim=30.0,
            stiffness=STIFFNESS_8520,
            damping=DAMPING_8520,
            armature=ARMATURE_8520,
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim=50.0,
            velocity_limit_sim=30.0,
            stiffness=2.0 * STIFFNESS_6020,
            damping=2.0 * DAMPING_6020,
            armature=2.0 * ARMATURE_6020,
        ),
        # 新增肩部
        "shoulders": ImplicitActuatorCfg(
            joint_names_expr=[".*_shoulder_pitch_joint"],
            effort_limit_sim=40.0,
            velocity_limit_sim=20.0,
            stiffness=3.0 *STIFFNESS_8520,
            damping=DAMPING_8520,
            armature=ARMATURE_8520,
        ),
    },
)
