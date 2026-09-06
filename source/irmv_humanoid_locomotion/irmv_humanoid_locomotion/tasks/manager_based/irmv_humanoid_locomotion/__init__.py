# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##
gym.register(
    id="Irmv-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.irmv_humanoid_locomotion_env_cfg:IrmvFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:IrmvFlatPPORunnerCfg",
    },
)

gym.register(
    id="Irmv-Flat-Arm-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.irmv_humanoid_locomotion_env_arm_cfg:IrmvFlatEnvArmCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:IrmvFlatPPORunnerCfg",
    },
)

gym.register(
    id="Irmv-Flat-Enhancedarm-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.irmv_humanoid_locomotion_env_enhancedarm_cfg:IrmvFlatEnvEnhancedarmCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:IrmvFlatPPORunnerCfg",
    },
)



gym.register(
    id="Irmv-Flat-Aug-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.irmv_flat_env_aug_cfg:IrmvFlatEnvAugCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.irmv_flat_ppo_aug_cfg:IrmvFlatPPOAUGRunnerCfg",
    },
)


gym.register(
    id="Irmv-Flat-Eqic-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.irmv_flat_env_eqic_cfg:IrmvFlatEnvEqicCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.irmv_flat_ppo_eqic_cfg:IrmvFlatPPOeqicRunnerCfg",
    },
)
