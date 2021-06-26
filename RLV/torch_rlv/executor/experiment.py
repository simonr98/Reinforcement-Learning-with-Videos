import os
import gym
import numpy as np
import multiworld
from ..environments.utils import get_environment
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
        self.algo_name = config['algo_name']

    def run_experiment(self):
        log_dir = "/tmp/gym/"
        os.makedirs(log_dir, exist_ok=True)

        multiworld.register_all_envs()

        env = get_environment(self.env_name)
        print(env)
        #env = Monitor(env, log_dir)

        # TODO: Hyperparameters: check_freq, log_dir, layer size etc
        model = SoftActorCritic(policy='MlpPolicy', env_name=self.env_name, env=env, verbose=1, learning_starts=1000)
        model.run()

        env.close()

        plot_results(log_dir)