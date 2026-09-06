# irmv_flat_env_aug.py
import torch
from isaaclab.envs import ManagerBasedRLEnv
import math
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from irmv_humanoid_locomotion.robots import IRMV_V3_CFG
from irmv_humanoid_locomotion.terrains.height_field import FractalNoiseTerrainCfg
class IrmvFlatEnvAug(ManagerBasedRLEnv):  
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)

        self._joint_names = self.scene["robot"].joint_names
        self._flip_joint_names = [name for name in self._joint_names if "pitch" in name]
        self._obs_slices = None

    def _build_obs_slices(self, obs):
        obs_mgr = self.observation_manager
        group_spec = obs_mgr.group_obs_specs["policy"]
        slices = {}
        start = 0
        for term_name, term_cfg in group_spec.term_cfgs.items():
            if "base_ang_vel" in term_name:
                dim = 3
            elif "base_lin_vel" in term_name:
                dim = 3
            elif "gravity" in term_name:
                dim = 3
            elif "joint_pos" in term_name:
                dim = len(self._joint_names)
            elif "joint_vel" in term_name:
                dim = len(self._joint_names)
            elif "actions" in term_name or "last_action" in term_name:
                dim = self.action_manager.actions_dim
            else:
                continue
            slices[term_name] = slice(start, start + dim)
            start += dim
        self._obs_slices = slices

    def mirror_obs_action(self, obs, action):
        if self._obs_slices is None:
            self._build_obs_slices(obs)

        robot = self.scene["robot"]
        joint_pos = robot.data.joint_pos
        joint_vel = robot.data.joint_vel

        pos_mirror = joint_pos.clone()
        vel_mirror = joint_vel.clone()
        for name in self._flip_joint_names:
            idx = self._joint_names.index(name)
            pos_mirror[:, idx] = -pos_mirror[:, idx]
            vel_mirror[:, idx] = -vel_mirror[:, idx]

        base_lin_vel = robot.data.root_lin_vel_w
        base_ang_vel = robot.data.root_ang_vel_w
        lin_mirror = base_lin_vel.clone()
        ang_mirror = base_ang_vel.clone()
        lin_mirror[:, 1] = -lin_mirror[:, 1]
        ang_mirror[:, 1] = -ang_mirror[:, 1]

        obs_mirror = obs.clone()
        slices = self._obs_slices
        if 'base_ang_vel' in slices:
            obs_mirror[:, slices['base_ang_vel']] = ang_mirror
        if 'base_lin_vel' in slices:
            obs_mirror[:, slices['base_lin_vel']] = lin_mirror
        if 'joint_pos' in slices:
            obs_mirror[:, slices['joint_pos']] = pos_mirror
        if 'joint_vel' in slices:
            obs_mirror[:, slices['joint_vel']] = vel_mirror

        action_mirror = action.clone()
        for name in self._flip_joint_names:
            idx = self._joint_names.index(name)
            action_mirror[:, idx] = -action_mirror[:, idx]

        return obs_mirror, action_mirror
