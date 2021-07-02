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
from RLV.torch_rlv.human_data.adapter import Adapter
from RLV.torch_rlv.algorithms.sac.softactorcritic import SoftActorCritic, SaveOnBestTrainingRewardCallback
from RLV.torch_rlv.visualizer.plot import plot_learning_curve, plot_env_step, animate_env_obs
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
import numpy as np
from datetime import datetime


class RLV(SAC):
    def __init__(self, warmup_steps=1500, beta_inverse_model=0.0003, env_name='acrobot_continuous',
                 policy='MlpPolicy', env=None, learning_rate=0.0003, buffer_size=1000000, learning_starts=1000,
                 batch_size=256, tau=0.005, gamma=0.99, train_freq=1, gradient_steps=1, optimize_memory_usage=False,
                 ent_coef='auto', target_update_interval=1, target_entropy='auto'):
        super().__init__(policy, env, learning_rate, buffer_size, learning_starts,
                         batch_size, tau, gamma, train_freq, gradient_steps, optimize_memory_usage,
                         ent_coef, target_update_interval, target_entropy)
        self.env_name = env_name
        self.warmup_steps = warmup_steps
        self.beta_inverse_model = beta_inverse_model

        if 'multi_world' in self.env_name:
            self.n_actions = env.action_space.shape[0]
        else:
            self.n_actions = env.action_space.shape[-1]

        self.inverse_model = InverseModel(observation_space_dims=env.observation_space.shape[-1],
                                          action_space_dims=self.n_actions,
                                          env=self.env, warmup_steps=self.warmup_steps)
        self.log_dir = "/tmp/gym/"
        os.makedirs(self.log_dir, exist_ok=True)

        self.env = env
        self.env_name = env_name
        self.env = Monitor(env, self.log_dir)

        self.batch_size = batch_size

        self.n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(self.n_actions), sigma=0.1 * np.ones(self.n_actions))
        self.callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=self.log_dir)

        self.action_free_replay_buffer = ReplayBuffer(
            buffer_size=buffer_size, observation_space=env.observation_space,
            action_space=env.action_space, device='cpu', n_envs=1,
            optimize_memory_usage=optimize_memory_usage,
            handle_timeout_termination=False)

        self.model = SoftActorCritic(policy='MlpPolicy', env_name=self.env_name, env=env, verbose=1, learning_starts=1000)

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


    def run(self):
        #self.inverse_model.warmup()
        self.fill_action_free_buffer()
        obs, target_action, next_obs, reward, done = self.action_free_replay_buffer.sample(batch_size=self.batch_size)

        input_inverse_model = T.cat((obs, next_obs), dim=1)

        predicted_action = self.inverse_model.network(input_inverse_model)

        self.inverse_model.calculate_loss(predicted_action, target_action)
        self.inverse_model.update()

        pass

