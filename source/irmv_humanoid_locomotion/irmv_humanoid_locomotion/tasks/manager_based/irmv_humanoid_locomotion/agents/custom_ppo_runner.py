# custom_ppo_runner.py
import torch
import numpy as np
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic
from collections import deque
import copy

class PPOAugRunner(OnPolicyRunner):
    """PPO with data augmentation for symmetry (PPOaug)."""

    def __init__(self, cfg, env):
        super().__init__(env, cfg.to_dict(), log_dir=None, device=cfg.device)
        self.joint_names = env.scene["robot"].joint_names  
        self.flip_joint_indices = [
            i for i, name in enumerate(self.joint_names) if "pitch" in name
        ]

    def _get_obs_action_mirror(self, obs, action):

        if not hasattr(self, 'obs_manager'):
            self.obs_manager = self.env.observation_manager
        if hasattr(self.env, 'mirror_obs_action'):
            obs_mirror, action_mirror = self.env.mirror_obs_action(obs, action)
        else:
            for idx in self.flip_joint_indices:
                if action.shape[1] == len(self.joint_names):
                    action_mirror[:, idx] = -action_mirror[:, idx]
                else:
                    pass
            raise NotImplementedError("Please implement env.mirror_obs_action()")
        return obs_mirror, action_mirror

    def _update_network(self):
        obs = self.obs_buf
        actions = self.actions_buf
        advantages = self.advantages_buf
        returns = self.returns_buf


        obs_mirror, actions_mirror = self._get_obs_action_mirror(obs, actions)

        obs_cat = torch.cat([obs, obs_mirror], dim=0)
        actions_cat = torch.cat([actions, actions_mirror], dim=0)
        advantages_cat = torch.cat([advantages, advantages], dim=0)   
        returns_cat = torch.cat([returns, returns], dim=0)

        orig_obs = self.alg.obs_buf
        orig_actions = self.alg.actions_buf
        orig_advantages = self.alg.advantages
        orig_returns = self.alg.returns
        self.alg.obs_buf = obs_cat
        self.alg.actions_buf = actions_cat
        self.alg.advantages = advantages_cat
        self.alg.returns = returns_cat

        actor_critic = self.alg.actor_critic

        data_len = obs_cat.shape[0]
        indices = torch.randperm(data_len, device=obs.device)

        mini_batch_size = data_len // self.alg.num_mini_batches
        for epoch in range(self.alg.num_learning_epochs):
            for start in range(0, data_len, mini_batch_size):
                end = min(start + mini_batch_size, data_len)
                batch_indices = indices[start:end]

                obs_batch = obs_cat[batch_indices]
                actions_batch = actions_cat[batch_indices]
                advantages_batch = advantages_cat[batch_indices]
                returns_batch = returns_cat[batch_indices]

                with torch.no_grad():
                    old_log_prob = actor_critic.get_actions_log_prob(actions_batch)

                log_prob, entropy = actor_critic.evaluate(obs_batch, actions_batch)

                ratio = torch.exp(log_prob - old_log_prob)

                surr1 = ratio * advantages_batch
                surr2 = torch.clamp(ratio, 1.0 - self.alg.clip_param, 1.0 + self.alg.clip_param) * advantages_batch
                policy_loss = -torch.min(surr1, surr2).mean()

                value = actor_critic.evaluate_critic(obs_batch)
                value_loss = (returns_batch - value).pow(2).mean()

                entropy_loss = -entropy.mean() * self.alg.entropy_coef

                loss = policy_loss + value_loss * self.alg.value_loss_coef + entropy_loss

                self.alg.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), self.alg.max_grad_norm)
                self.alg.optimizer.step()

        self.alg.obs_buf = orig_obs
        self.alg.actions_buf = orig_actions
        self.alg.advantages = orig_advantages
        self.alg.returns = orig_returns
