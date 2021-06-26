import os
import gym
import numpy as np
from ..environments.utils import get_environment
from RLV.torch_rlv.algorithms.sac.sac import SAC
from RLV.torch_rlv.algorithms.sac.softactorcritic import SaveOnBestTrainingRewardCallback, SoftActorCritic
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from multiworld.core.flat_goal_env import FlatGoalEnv
import RLV.torch_rlv.rlv_base.custom_envs


class Experiment:
    def __init__(self, config):
        self.config = config
        self.env_name = config['env_name']
        self.algo_name = config['algo_name']

    def run_experiment(self):
        log_dir = "/tmp/gym/"
        os.makedirs(log_dir, exist_ok=True)

        env = get_environment('acrobot_continuous')
        env = Monitor(env, log_dir)

        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
        model = SoftActorCritic('MlpPolicy', env, verbose=1, learning_starts=1000)
        model.run()

        env.close()

        plot_results(log_dir)