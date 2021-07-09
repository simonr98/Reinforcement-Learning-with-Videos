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


class RLV():
    def __init__(self, warmup_steps=1500, beta_inverse_model=0.0003, env_name='acrobot_continuous',
                 policy='MlpPolicy', env=None, learning_rate=0.0003, buffer_size=1000000, learning_starts=1000,
                 batch_size=256, tau=0.005, gamma=0.99, train_freq=1, gradient_steps=1, optimize_memory_usage=False,
                 ent_coef='auto', target_update_interval=1, target_entropy='auto', time_steps=0,
                 initial_exploration_steps=1000, log_dir="/tmp/gym/", domain_shift=False,
                 domain_shift_generator_weight=0.01, domain_shift_discriminator_weight=0.01, paired_loss_scale=1.0):

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

        self.model = SoftActorCritic(policy='MlpPolicy', env_name=self.env_name, env=env, verbose=1,
                                     learning_starts=1000, wandb_log=False, project_name='rlv_exper',
                                     gradient_steps=1)

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

    def run(self):
        # warmup inverse model
        self.inverse_model.warmup()

        # fill action free buffer
        self.fill_action_free_buffer()
        self.model.run(total_timesteps=1)

        for s in range(1, 2500):
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
            data_int = self.model.model.replay_buffer.sample(self.batch_size, env=self.model.model._vec_normalize_env)

            # replace the data used in SAC for each gradient steps by observational plus robot data
            combined_data = ReplayBufferSamples(
                observations=T.cat((data_int.observations, state_obs), dim=0),
                actions=T.cat((data_int.actions, action_obs), dim=0),
                next_observations=T.cat((data_int.next_observations, next_state_obs), dim=0),
                dones=T.cat((data_int.dones, done_obs), dim=0),
                rewards=T.cat((data_int.rewards, reward_obs), dim=0)
            )

            self.model.model.rlv_data = combined_data

            # 1000 exploration steps - one gradient step
            self.time_steps += 1000
            self.model.model.gradient_steps = 1

            callback = SaveOnBestTrainingRewardCallback(check_freq=500, log_dir=self.log_dir,
                                                        total_timesteps=self.model.total_timesteps)
            self.callback = callback
            self.model.model.learn(total_timesteps=self.time_steps, callback=self.callback)

            self.inverse_model.calculate_loss(action_obs, target_action)
            self.inverse_model.update()

        print('Run successfull')
