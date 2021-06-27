import torch as T
import os
from RLV.torch_rlv.models.inverse_model_network import InverseModelNetwork
from RLV.torch_rlv.algorithms.rlv.inversemodel import InverseModel
from RLV.torch_rlv.algorithms.sac.sac import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.noise import NormalActionNoise
from RLV.torch_rlv.algorithms.sac.softactorcritic import SoftActorCritic, SaveOnBestTrainingRewardCallback
from RLV.torch_rlv.visualizer.plot import plot_learning_curve, plot_env_step, animate_env_obs
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

        self.n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(self.n_actions), sigma=0.1 * np.ones(self.n_actions))
        self.callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=self.log_dir)

        self.model = SoftActorCritic(policy='MlpPolicy', env_name=self.env_name, env=env, verbose=1, learning_starts=1000)

    def run(self):
        self.inverse_model.warmup()


