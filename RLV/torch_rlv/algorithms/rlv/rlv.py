import torch as T
import gym
import os
import pickle
from RLV.torch_rlv.models.inverse_model_network import InverseModelNetwork
from RLV.torch_rlv.algorithms.rlv.inversemodel import InverseModel
from RLV.torch_rlv.buffer.buffers import DictReplayBuffer, ReplayBuffer
from RLV.torch_rlv.algorithms.sac.sac import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.noise import NormalActionNoise
from RLV.torch_rlv.buffer.type_aliases import ReplayBufferSamples
from RLV.torch_rlv.human_data.adapter import Adapter
from RLV.torch_rlv.algorithms.sac.softactorcritic import SoftActorCritic, SaveOnBestTrainingRewardCallback
from RLV.torch_rlv.visualizer.plot import plot_learning_curve, plot_env_step, animate_env_obs
import numpy as np
from datetime import datetime


class RLV(SAC):
    def __init__(self, warmup_steps=1500, beta_inverse_model=0.0003, env_name='acrobot_continuous', policy='MlpPolicy',
                 env=None, learning_rate=0.0003, buffer_size=1000000, learning_starts=1000, batch_size=256, tau=0.005,
                 gamma=0.99, train_freq=1, gradient_steps=1, optimize_memory_usage=False, ent_coef='auto',
                 target_update_interval=1, target_entropy='auto', time_steps=0, initial_exploration_steps=1000,
                 log_dir="/tmp/gym/", domain_shift=False, domain_shift_generator_weight=0.01,
                 domain_shift_discriminator_weight=0.01, paired_loss_scale=1.0):

        super().__init__(policy, env, learning_rate, buffer_size, learning_starts, batch_size, tau, gamma, train_freq,
                         gradient_steps, optimize_memory_usage, ent_coef, target_update_interval, target_entropy)
        self.policy = policy
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.learnings_starts = learning_starts
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        self.env_name = env_name
        self.warmup_steps = warmup_steps
        self.beta_inverse_model = beta_inverse_model
        self.ent_coef = ent_coef
        self.target_update_interval = target_update_interval
        self.target_entropy = target_entropy

        self.replay_buffer_class = replay_buffer_class

        self.domain_shift = domain_shift
        self.inverse_model_lr = 3e-4
        self.domain_shift_discrim_lr = 3e-4
        self.paired_loss_lr = 3e-4
        self.paired_loss_scale = paired_loss_scale

        self.domain_shift = domain_shift
        self.domain_shift_generator_weight = domain_shift_generator_weight
        self.domain_shift_discriminator_weight = domain_shift_discriminator_weight

        self.log_dir = log_dir

        self.time_steps = time_steps
        self.initial_exploration_steps = initial_exploration_steps

        self.env = env
        self.env_name = env_name
        self.env = Monitor(env, self.log_dir)

        if 'multi_world' in self.env_name:
            self.n_actions = env.action_space.shape[0]
        else:
            self.n_actions = env.action_space.shape[-1]

        self.inverse_model = InverseModel(observation_space_dims=env.observation_space.shape[-1],
                                          action_space_dims=self.n_actions,
                                          env=self.env, warmup_steps=self.warmup_steps)
        self.log_dir = "/tmp/gym/"
        os.makedirs(self.log_dir, exist_ok=True)

        self.n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(self.n_actions), sigma=0.1 * np.ones(self.n_actions))
        self.callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=self.log_dir)

        self.action_free_replay_buffer = ReplayBuffer(
            buffer_size=buffer_size, observation_space=env.observation_space,
            action_space=env.action_space, device='cpu', n_envs=1,
            optimize_memory_usage=optimize_memory_usage, handle_timeout_termination=False)

    def fill_action_free_buffer(self):
        data = Adapter(data_type='unpaired', env_name='Acrobot')

        observations = data.observations
        next_observations = data.next_observations
        actions = data.actions
        rewards = data.rewards
        terminals = data.terminals

        for i in range(0, observations.shape[1]):
            self.action_free_replay_buffer.add(
                obs=observations[i],
                next_obs=next_observations[i],
                action=actions[i],
                reward=rewards[i],
                done=terminals[i],
                infos={'': ''}
            )

    def set_reward(self, done=False):
        if self.env_name == 'acrobot_continuous':
            if done:
                return 10
            else:
                return -1
        else:
            return -1

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        for gradient_step in range(gradient_steps):
            state_obs, target_action, next_state_obs, _, done_obs \
                = self.action_free_replay_buffer.sample(batch_size=self.batch_size)

            # get predicted action from inverse model
            input_inverse_model = T.cat((state_obs, next_state_obs), dim=1)
            action_obs = self.inverse_model.network(input_inverse_model)

            # set rewards for observational data
            reward_obs = T.zeros(self.batch_size, 1)
            for i in range(0, self.batch_size):
                reward_obs[i] = self.set_reward(done=done_obs[i])

            # get robot data - sample from replay pool from the SAC model
            data_int = self.model.model.replay_buffer.sample(self.batch_size, env=self._vec_normalize_env)

            # replace the data used in SAC for each gradient steps by observational plus robot data
            replay_data = ReplayBufferSamples(
                observations=T.cat((data_int.observations, state_obs), dim=0),
                actions=T.cat((data_int.actions, action_obs), dim=0),
                next_observations=T.cat((data_int.next_observations, next_state_obs), dim=0),
                dones=T.cat((data_int.dones, done_obs), dim=0),
                rewards=T.cat((data_int.rewards, reward_obs), dim=0)
            )

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward(retain_graph=True)
                self.ent_coef_optimizer.step()

            with th.no_grad():
                # Select action according to policy
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                # Compute the next Q values: min over all critics targets
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                # add entropy term
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                # td error + entropy term
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = 0.5 * sum([F.mse_loss(current_q, target_q_values) for current_q in current_q_values])
            critic_losses.append(critic_loss.item())

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            self.critic.optimizer.step()

            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Mean over all critic networks
            q_values_pi = th.cat(self.critic.forward(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.actor.optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

        self.inverse_model.calculate_loss(action_obs, target_action)
        self.inverse_model.update()

