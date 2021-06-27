import os
import gym
import numpy as np
import multiworld
from RLV.torch_rlv.algorithms.sac.softactorcritic import plot_results
from ..environments.utils import get_environment
from RLV.torch_rlv.algorithms.utils import init_algorithm
from RLV.torch_rlv.algorithms.sac.sac import SAC
from RLV.torch_rlv.algorithms.sac.softactorcritic import SaveOnBestTrainingRewardCallback, SoftActorCritic
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from multiworld.core.flat_goal_env import FlatGoalEnv


class Experiment:
    def __init__(self, config):
        self.config = config
        self.env_name = config['env_name']
        self.env = None
        self.algo_name = config['algo_name']
        self.policy = config['policy']
        self.lr_inverse_model = config['lr_inverse_model']

    def run_experiment(self):
        log_dir = "/tmp/gym/"
        os.makedirs(log_dir, exist_ok=True)

        multiworld.register_all_envs()

        self.env = get_environment(self.env_name)
        #env = Monitor(env, log_dir)

        # TODO: Hyperparameters: check_freq, log_dir, layer size etc
        algorithm = init_algorithm(self.algo_name, self)
        algorithm.run()

        self.env.close()

        if self.algo_name == 'sac':
            plot_results(log_dir)