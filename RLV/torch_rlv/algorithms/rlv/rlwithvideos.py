from RLV.torch_rlv.algorithms.rlv.rlv import RLV

import os

import numpy as np
import matplotlib.pyplot as plt
import wandb
from RLV.torch_rlv.algorithms.sac.sac import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from RLV.torch_rlv.buffer.type_aliases import TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from RLV.torch_rlv.algorithms.sac.softactorcritic import SaveOnBestTrainingRewardCallback



class RlWithVideos:
    def __init__(self, policy='MlpPolicy', env_name=None, config=None, wandb_log=False,
                 env=None, learning_rate=0.0003, buffer_size=1000000, learning_starts=1000,
                 batch_size=256, tau=0.005, gamma=0.99, train_freq=1, gradient_steps=1, _init_setup_model=True,
                 project_name='rlv_experiment', run_name='test_rlv'):
        self.log_dir = "/tmp/gym/"
        os.makedirs(self.log_dir, exist_ok=True)

        self.config = config

        self.env = env
        #self.env = Monitor(env, self.log_dir)

        self.env_name = env_name

        self.project_name = project_name
        self.run_name = run_name

        self.learning_rate = learning_rate
        self.policy = policy
        self.buffer_size = buffer_size
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps

        if 'multi_world' in self.env_name:
            self.n_actions = env.action_space.shape[0]
        else:
            self.n_actions = env.action_space.shape[-1]

        action_noise = NormalActionNoise(mean=np.zeros(self.n_actions), sigma=0.1 * np.ones(self.n_actions))
        self.callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=self.log_dir)

        self.model = RLV(warmup_steps=1, beta_inverse_model=0.0003, env_name='acrobot_continuous', policy='MlpPolicy',
                         env=self.env, learning_rate=0.0003, buffer_size=1000000, learning_starts=1000, batch_size=256, tau=0.005,
                         gamma=0.99, train_freq=self.train_freq, gradient_steps=1, optimize_memory_usage=False, ent_coef='auto',
                         target_update_interval=1, target_entropy='auto', time_steps=0, initial_exploration_steps=1000,
                         log_dir="/tmp/gym/", domain_shift=False, domain_shift_generator_weight=0.01,
                         domain_shift_discriminator_weight=0.01, paired_loss_scale=1.0)
        if wandb_log:
            self.logger = wandb.init(project=project_name,
                                     config=self.config,
                                     name=run_name,
                                     reinit=True,  # allow things to be run multiple times
                                     settings=wandb.Settings(start_method="thread"))

    def run(self, total_timesteps=int(250000), plot=False):
        #self.model.fill_action_free_buffer()
        #self.model.inverse_model.warmup()

        callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=self.log_dir)
        self.callback = callback
        self.model.learn(total_timesteps=total_timesteps, callback=self.callback)

        if plot:
            plot_results(self.log_dir)

    def get_env(self):
        return self.model.get_env()

    def get_params(self):
        return self.model.get_parameters()

    def load(self, path, env):
        self.model.save(path, env)

    def save(self, path):
        self.model.save(path)

    def save_replay_buffer(self, path):
        self.model.save_replay_buffer(path)

    def set_env(self, env):
        self.model.set_env(env)

    def set_params(self, load_path_or_dict):
        self.model.set_parameters(load_path_or_dict)

    def set_random_seed(self, seed):
        self.model.set_random_seed(seed=seed)

    def train(self, gradient_step, batch_size):
        self.model.train(gradient_step, batch_size=batch_size)

    def load_replay_buffer(self, path):
        self.model.load_replay_buffer(path)

    def predict(self, observation, state=None, mask=None, deterministic=False):
        return self.model.predict(observation, state=state, mask=mask, deterministic=deterministic)

