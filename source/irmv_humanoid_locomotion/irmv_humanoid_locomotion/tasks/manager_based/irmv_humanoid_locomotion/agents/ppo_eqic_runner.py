# ppo_eqic_runner.py 

import torch
import torch.nn as nn
import numpy as np
import time
import os
import escnn
from escnn.group import CyclicGroup, directsum
from escnn.nn import GeometricTensor
from morpho_symm.nn.EMLP import EMLP
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.algorithms import PPO
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict
from isaaclab.utils.dict import print_dict



class PPOEqic(PPO):
    def init_storage(
        self,
        training_type,           
        num_envs,
        num_transitions_per_env,
        obs,                     
        actions_shape,
    ):
        self.storage = RolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            obs,
            actions_shape,
            self.device,
        )
        self.storage.recurrent_obs_shape = None
        self.storage.recurrent_obs = None
        print(
            f"[INFO] RolloutStorage initialized: "
            f"training_type={training_type}, "
            f"num_envs={num_envs}, "
            f"num_steps={num_transitions_per_env}, "
            f"actions_shape={actions_shape}"
        )

    def process_env_step(self, obs, rewards, dones, infos):
        # ----- 1. observation -----
        self.transition.observations = obs

        # ----- 2. rewards -----
        if not torch.is_tensor(rewards):
            rewards = torch.as_tensor(rewards, device=self.device, dtype=torch.float32)
        else:
            rewards = rewards.to(self.device)
        if rewards.ndim == 2 and rewards.shape[-1] == 1:
            rewards = rewards.squeeze(-1)
        if rewards.ndim != 1:
            raise RuntimeError(f"[PPOEqic] Invalid rewards shape: {tuple(rewards.shape)}")

        # ----- 3. dones -----
        if not torch.is_tensor(dones):
            dones = torch.as_tensor(dones, device=self.device)
        else:
            dones = dones.to(self.device)
        if dones.ndim == 2 and dones.shape[-1] == 1:
            dones = dones.squeeze(-1)
        if dones.ndim != 1:
            raise RuntimeError(f"[PPOEqic] Invalid dones shape: {tuple(dones.shape)}")

        # ----- 4. values -----
        values = self.transition.values
        if values is None:
            raise RuntimeError("[PPOEqic] transition.values is None.")
        if values.ndim == 1:
            values = values.unsqueeze(-1)
        if values.ndim != 2 or values.shape[-1] != 1:
            raise RuntimeError(f"[PPOEqic] Unexpected critic value shape: {tuple(values.shape)}")
        self.transition.values = values

        # ----- 5. timeout bootstrap -----
        if "time_outs" in infos:
            time_outs = infos["time_outs"]
            if not torch.is_tensor(time_outs):
                time_outs = torch.as_tensor(time_outs, device=values.device)
            else:
                time_outs = time_outs.to(values.device)
            if time_outs.ndim == 1:
                time_outs = time_outs.unsqueeze(-1)
            elif time_outs.ndim == 2 and time_outs.shape[-1] == 1:
                pass
            else:
                raise RuntimeError(f"[PPOEqic] Invalid time_outs shape: {tuple(time_outs.shape)}")
            timeout_bootstrap = (self.gamma * values * time_outs.to(dtype=values.dtype)).squeeze(-1)
            if timeout_bootstrap.shape != rewards.shape:
                raise RuntimeError(
                    f"[PPOEqic] Timeout bootstrap shape mismatch: "
                    f"rewards {tuple(rewards.shape)} vs result {tuple(timeout_bootstrap.shape)}"
                )
            rewards = rewards + timeout_bootstrap

        # ----- 6. transition -----
        self.transition.rewards = rewards
        self.transition.dones = dones

        # ----- 7. storage -----
        self.storage.add_transitions(self.transition)

        # ----- 8. clear -----
        self.transition.clear()


class EquivariantActorCritic(nn.Module):
    def __init__(self, actor, critic, in_type, act_dim, device='cuda'):
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.in_type = in_type
        self.act_dim = act_dim
        self.obs_dim = in_type.size
        self.device = device
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.8, device=device))
        self.is_recurrent = False
        self.recurrent_obs_groups = None
        self.actor_obs_groups = None
        self.critic_obs_groups = None
        self._action_mean = None
        self._action_std = None
        self.entropy = None

    @property
    def action_mean(self):
        return self._action_mean

    @property
    def action_std(self):
        return torch.exp(self.log_std)

    def _get_obs_tensor(self, obs):
        if isinstance(obs, TensorDict):
            for key in ['policy', 'critic', 'obs', 'actor_obs']:
                if key in obs.keys():
                    return obs[key].to(self.device)
            raise KeyError(f"No suitable observation key found in {obs.keys()}")
        return obs

    def _reshape_mean(self, mean, batch_size):
        if mean.dim() == 1:
            actual_act_dim = mean.size(0) // batch_size
            if actual_act_dim != self.act_dim:
                self.act_dim = actual_act_dim
            return mean.view(batch_size, self.act_dim)
        else:
            if mean.size(1) != self.act_dim:
                self.act_dim = mean.size(1)
            return mean

    def act(self, obs, **kwargs):
        obs_tensor = self._get_obs_tensor(obs)
        geo_obs = GeometricTensor(obs_tensor, self.in_type)
        action_geo = self.actor(geo_obs)
        mean = self._reshape_mean(action_geo.tensor, obs_tensor.shape[0])
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        
        self._action_mean = mean.detach()
        self._action_std = std.detach()
        self.entropy = dist.entropy().sum(dim=-1).detach()
        return action

    def evaluate(self, obs, actions=None, masks=None, hidden_state=None):
        obs_tensor = self._get_obs_tensor(obs)
        geo_obs = GeometricTensor(obs_tensor, self.in_type)
        value = self.critic(geo_obs).tensor
        if value.ndim == 1:
            value = value.unsqueeze(-1)
        if value.ndim != 2 or value.shape[-1] != 1:
            raise RuntimeError(f"[EQIC] Critic output must be [N,1], got {tuple(value.shape)}")
        if actions is None:
            return value
        action_geo = self.actor(geo_obs)
        mean = self._reshape_mean(action_geo.tensor, obs_tensor.shape[0])
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return value, log_prob, entropy

    def get_actions_log_prob(self, *args):
        if len(args) == 1:
            actions = args[0]
            if self._action_mean is None:
                raise RuntimeError("No stored action mean. Call act() first.")
            mean = self._action_mean
            std = self._action_std
        elif len(args) == 2:
            obs, actions = args
            obs_tensor = self._get_obs_tensor(obs)
            geo_obs = GeometricTensor(obs_tensor, self.in_type)
            action_geo = self.actor(geo_obs)
            mean = self._reshape_mean(action_geo.tensor, obs_tensor.shape[0])
            std = torch.exp(self.log_std)
        else:
            raise TypeError(f"get_actions_log_prob() takes 1 or 2 arguments, got {len(args)}")
        dist = torch.distributions.Normal(mean, std)
        return dist.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs):
        pass

    def reset(self, dones=None):
        pass

    def train(self, mode=True):
        super().train(mode)
        self.actor.train(mode)
        self.critic.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def to(self, device):
        super().to(device)
        self.actor.to(device)
        self.critic.to(device)
        self.log_std.data = self.log_std.data.to(device)
        return self




class PPOeqicRunner(OnPolicyRunner):
    def __init__(self, env, train_cfg, log_dir=None, device="cuda"):
        self.env = env
        self.train_cfg = train_cfg
        self.log_dir = log_dir
        self.device = device
        self.num_envs = env.num_envs
        self.num_steps_per_env = train_cfg["num_steps_per_env"]
        self.git_status_repos = []
        self.writer = None
        self.disable_logs = False

        self.gpu_world_size = 1
        self.gpu_local_rank = 0
        self.gpu_global_rank = 0
        self.multi_gpu_cfg = None
        self.is_distributed = False
        self.current_learning_iteration = 0
        self.tot_timesteps = 0
        self.tot_time = 0
        self.logger_type = "tensorboard"


        G = CyclicGroup(2)
        rep_trivial = G.irrep(0)
        rep_sign = G.irrep(1)
        R3_rep = directsum([rep_trivial, rep_sign, rep_trivial], name='R3')
        R3_pseudo_rep = directsum([rep_sign, rep_trivial, rep_sign], name='R3_pseudo')
        Q_rep = directsum([rep_trivial] * 7 + [rep_sign] * 7, name='Q_js')
        G.representations['R3'] = R3_rep
        G.representations['R3_pseudo'] = R3_pseudo_rep
        G.representations['Q_js'] = Q_rep
        G.representations['TqQ_js'] = Q_rep
        self.G = G
        #print("手动加载对称系统成功！")



        self.gspace = escnn.gspaces.no_base_space(self.G)
        rep_Q = self.G.representations["Q_js"]
        rep_V = self.G.representations["TqQ_js"]
        rep_R3 = self.G.representations["R3"]
        rep_R3_pseudo = self.G.representations["R3_pseudo"]

        self.in_type = escnn.nn.FieldType(self.gspace, [
            rep_R3_pseudo, rep_R3, rep_Q, rep_V, rep_Q, rep_R3
        ])
        self.out_type_actor = escnn.nn.FieldType(self.gspace, [rep_Q])
        self.out_type_critic = escnn.nn.FieldType(self.gspace, [self.G.irrep(0)])




        actor_hidden_dims = train_cfg["policy"]["actor_hidden_dims"]
        activation = train_cfg["policy"]["activation"]
        hidden_size = actor_hidden_dims[0] if actor_hidden_dims else 256
        num_layers = len(actor_hidden_dims) + 1 if actor_hidden_dims else 3

        self.actor_network = EMLP(
            in_type=self.in_type,
            out_type=self.out_type_actor,
            num_hidden_units=hidden_size,
            num_layers=num_layers,
            activation=activation,
            bias=True,
            head_with_activation=False,
        )
        self.critic_network = EMLP(
            in_type=self.in_type,
            out_type=self.out_type_critic,
            num_hidden_units=hidden_size,
            num_layers=num_layers,
            activation=activation,
            bias=True,
            head_with_activation=False,
        )

        act_dim = self.out_type_actor.size
        obs_dim = self.in_type.size

        self.actor_critic = EquivariantActorCritic(
            self.actor_network, self.critic_network,
            self.in_type, act_dim, device=self.device
        )

        # ----- PPOEqic -----
        alg_cfg = train_cfg["algorithm"]
        self.alg = PPOEqic(
            policy=self.actor_critic,
            device=self.device,
            num_learning_epochs=alg_cfg["num_learning_epochs"],
            num_mini_batches=alg_cfg["num_mini_batches"],
            clip_param=alg_cfg["clip_param"],
            gamma=alg_cfg["gamma"],
            lam=alg_cfg["lam"],
            value_loss_coef=alg_cfg["value_loss_coef"],
            entropy_coef=alg_cfg["entropy_coef"],
            learning_rate=alg_cfg["learning_rate"],
            schedule=alg_cfg["schedule"],
            desired_kl=alg_cfg["desired_kl"],
            max_grad_norm=alg_cfg["max_grad_norm"],
            use_clipped_value_loss=alg_cfg["use_clipped_value_loss"],
        )


        if hasattr(env, "get_observations"):
            obs_sample = env.get_observations()
        else:
            obs_sample, _ = env.reset()

        if not isinstance(obs_sample, TensorDict):
            obs_sample = TensorDict({"obs": obs_sample}, batch_size=[self.num_envs])


        self.alg.init_storage(
            training_type="rl",
            num_envs=self.num_envs,
            num_transitions_per_env=self.num_steps_per_env,
            obs=obs_sample,
            actions_shape=(act_dim,),
        )


        self.obs_buf = torch.zeros((self.num_steps_per_env, self.num_envs, obs_dim), device=self.device)
        self.actions_buf = torch.zeros((self.num_steps_per_env, self.num_envs, act_dim), device=self.device)
        self.advantages_buf = torch.zeros((self.num_steps_per_env, self.num_envs), device=self.device)
        self.returns_buf = torch.zeros((self.num_steps_per_env, self.num_envs), device=self.device)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        print("[INFO] PPOeqicRunner fully initialized with PPOEqic (new API).")
        print(f"[INFO] Obs dim: {obs_dim}, Act dim: {act_dim}")
        print("Timeout bootstrap now safe (broadcast fixed).")


        self.print_env_config()
        self.print_reward_config()


    def print_env_config(self):
        pass

    def print_reward_config(self):
        print("\n" + "=" * 80)
        print("Eqic Reward Config")
        print("=" * 80)
        try:
            cfg = None
            if hasattr(self.env, "unwrapped") and hasattr(self.env.unwrapped, "cfg"):
                cfg = self.env.unwrapped.cfg
            elif hasattr(self.env, "cfg"):
                cfg = self.env.cfg
            if cfg is None:
                print("[WARN] Cannot find environment config.")
                return
            if not hasattr(cfg, "rewards"):
                print("[WARN] cfg.rewards not found.")
                return
            rewards = cfg.rewards
            for name in dir(rewards):
                if name.startswith("_"):
                    continue
                try:
                    term = getattr(rewards, name)
                except Exception:
                    continue
                if term is None:
                    continue
                if hasattr(term, "weight"):
                    weight = term.weight
                    func = getattr(term, "func", None)
                    params = getattr(term, "params", None)
                    func_name = getattr(func, "__name__", str(func)) if func is not None else "None"
                    print(f"\n{name}")
                    print(f"    weight : {weight}")
                    print(f"    func   : {func_name}")
                    if params:
                        print(f"    params : {params}")
        except Exception as e:
            print(f"[WARN] Failed to print reward config: {e}")
        print("=" * 80 + "\n")

    def print_iteration_info(self, it, loss_dict, start_time, learn_time, mean_episode_reward, mean_episode_length):
        print("\n" + "-" * 80)
        if loss_dict is not None:
            for key, value in loss_dict.items():
                if torch.is_tensor(value):
                    if value.numel() == 1:
                        value = value.item()
                    else:
                        value = value.mean().item()
                try:
                    print(f"{key:<35}: {float(value):.6f}")
                except Exception:
                    print(f"{key:<35}: {value}")

        print(f"{'Learning rate':<35}: {self.alg.learning_rate:.8f}")
        if hasattr(self.actor_critic, "action_std"):
            try:
                std = self.actor_critic.action_std
                print(f"{'Action std mean':<35}: {std.mean().item():.6f}")
                print(f"{'Action std min':<35}: {std.min().item():.6f}")
                print(f"{'Action std max':<35}: {std.max().item():.6f}")
            except Exception:
                pass
        if mean_episode_reward is not None:
            print(f"{'Mean episode reward':<35}: {mean_episode_reward:.6f}")
        if mean_episode_length is not None:
            print(f"{'Mean episode length':<35}: {mean_episode_length:.3f}")
        print(f"{'Collection time':<35}: {start_time:.3f} s")
        print(f"{'Learning time':<35}: {learn_time:.3f} s")
        print("-" * 80 + "\n")

    def _get_reward_manager(self):
        try:
            if hasattr(self.env, "unwrapped"):
                env = self.env.unwrapped
            else:
                env = self.env

            if hasattr(env, "reward_manager"):
                return env.reward_manager

            if hasattr(env, "reward_mgr"):
                return env.reward_mgr

        except Exception as e:
            print(f"[WARN] Cannot get reward manager: {e}")

        return None

    def _get_reward_term_values(self):
        reward_manager = self._get_reward_manager()

        if reward_manager is None:
            return {}

        result = {}

        try:
            term_names = reward_manager.active_terms
            step_reward = reward_manager._step_reward  # [num_envs, num_terms]

            if step_reward is None:
                return {}

            for term_idx, name in enumerate(term_names):
                value = step_reward[:, term_idx]   # [num_envs]
                if value.ndim > 1:
                    value = value.reshape(self.num_envs, -1).mean(dim=-1)
                result[name] = value.detach()

        except Exception as e:
            print(f"[WARN] Failed to read reward terms from _step_reward: {e}")

        return result

    def _print_reward_statistics(self, reward_sum, reward_count):
        print("\n" + "=" * 80)
        print("                    EQIC REWARD STATISTICS")
        print("=" * 80)

        if not reward_sum:
            print("[WARN] No reward term data collected.")
            print("=" * 80 + "\n")
            return

        total_reward = 0.0

        for name in sorted(reward_sum.keys()):
            count = reward_count.get(name, 0)
            if count <= 0:
                continue
            avg_reward = reward_sum[name] / count   
            total_reward += avg_reward
            print(f"{name:<40}: {avg_reward:+.6f}")

        print("-" * 80)
        print(f"{'Total reward (sum of terms)':<40}: {total_reward:+.6f}")
        print("=" * 80 + "\n")

    def learn(self, num_learning_iterations, init_at_random_ep_len=True):
        if init_at_random_ep_len:
            self.env.reset()
            if hasattr(self.env, "reset_to_random_episode_length"):
                self.env.reset_to_random_episode_length()

        obs = self.env.get_observations() if hasattr(self.env, "get_observations") else self.env.reset()[0]
        if not isinstance(obs, TensorDict):
            obs = TensorDict({"obs": obs}, batch_size=[self.num_envs])


        self.alg.storage.clear()


        episode_reward_buf = torch.zeros(self.num_envs, device=self.device)
        episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        episode_rewards = []
        episode_lengths = []

        for it in range(num_learning_iterations):
            start_time = time.time()

            reward_sum = {}
            reward_count = {}

            for _ in range(self.num_steps_per_env):
                with torch.no_grad():
                    action = self.alg.act(obs)
                    next_obs, rewards, dones, infos = self.env.step(action)
                    if not isinstance(next_obs, TensorDict):
                        next_obs = TensorDict({"obs": next_obs}, batch_size=[self.num_envs])
                    current_rewards = self._get_reward_term_values()
                    for name, value in current_rewards.items():
                        if value.ndim == 0:
                            value = value.repeat(self.num_envs)
                        elif value.ndim > 1:
                            value = value.reshape(self.num_envs, -1).mean(dim=-1)
                        if name not in reward_sum:
                            reward_sum[name] = 0.0
                            reward_count[name] = 0
                        reward_sum[name] += value.mean().item()
                        reward_count[name] += 1

                    if "time_outs" not in infos:
                        infos["time_outs"] = dones.clone()

                    self.alg.process_env_step(obs, rewards, dones, infos)

                    episode_reward_buf += rewards
                    episode_length_buf += 1
                    done_indices = torch.nonzero(dones, as_tuple=True)[0]
                    for idx in done_indices:
                        episode_rewards.append(episode_reward_buf[idx].item())
                        episode_lengths.append(episode_length_buf[idx].item())
                        episode_reward_buf[idx] = 0.0
                        episode_length_buf[idx] = 0

                    obs = next_obs

            collect_time = time.time() - start_time

            print("\n" + "=" * 40)
            print(f"Epoch {it + 1}")
            print("=" * 40)
            self._print_reward_statistics(reward_sum, reward_count)

            with torch.no_grad():
                pass


            self.alg.compute_returns(obs)
            learn_start = time.time()
            loss_dict = self.alg.update()
            learn_time = time.time() - learn_start
            mean_ep_reward = np.mean(episode_rewards[-self.num_envs*2:]) if episode_rewards else None
            mean_ep_length = np.mean(episode_lengths[-self.num_envs*2:]) if episode_lengths else None
            self.print_iteration_info(it, loss_dict, collect_time, learn_time,
                                      mean_ep_reward, mean_ep_length)

            if (it + 1) % 50 == 0 and self.log_dir is not None:
                save_path = os.path.join(self.log_dir, f"model_{it+1}.pt")
                torch.save({
                    'model_state_dict': self.actor_critic.state_dict(),
                    'alg_state_dict': self.alg.state_dict() if hasattr(self.alg, 'state_dict') else None,
                }, save_path)
                print(f"[INFO] Model saved to {save_path}")

            self.current_learning_iteration = it
            self.tot_timesteps += self.num_envs * self.num_steps_per_env

        if self.log_dir is not None:
            final_path = os.path.join(self.log_dir, "model_final.pt")
            torch.save({
                'model_state_dict': self.actor_critic.state_dict(),
                'alg_state_dict': self.alg.state_dict() if hasattr(self.alg, 'state_dict') else None,
            }, final_path)
            print(f"[INFO] Final model saved to {final_path}")

        print("[INFO] Training completed.")
