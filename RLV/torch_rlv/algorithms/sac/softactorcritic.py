import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
from RLV.torch_rlv.algorithms.sac.sac import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from datetime import datetime


class SoftActorCritic:
    def __init__(self, policy='MlpPolicy', env_name=None, env=None, learning_rate=0.0003, buffer_size=1000000, learning_starts=1000,
                 batch_size=256, tau=0.005, gamma=0.99, train_freq=1, gradient_steps=1,  optimize_memory_usage=False,
                 ent_coef='auto', target_update_interval=1, target_entropy='auto', use_sde=False, sde_sample_freq=- 1,
                 use_sde_at_warmup=False, tensorboard_log=None, create_eval_env=False, policy_kwargs=None, verbose=0,
                 seed=None, device='auto', _init_setup_model=True, project_name='sac_experiment', run_name='test_sac',
                 log_dir='../output/tmp/gym/', total_steps=250000, wandb_log=False, algo_name='sac'):

        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        self.env = env
        self.env_name = env_name
        self.env = Monitor(env, self.log_dir)
        self.algo_name = algo_name

        self.total_steps = total_steps
        self.n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(self.n_actions), sigma=0.1 * np.ones(self.n_actions))

        self.model = SAC(policy, env, env_name, total_steps=total_steps, learning_rate=learning_rate, buffer_size=buffer_size,
                         learning_starts=learning_starts, batch_size=batch_size, tau=tau, gamma=gamma,
                         train_freq=train_freq, gradient_steps=gradient_steps, action_noise=action_noise,
                         optimize_memory_usage=optimize_memory_usage, ent_coef=ent_coef,
                         target_update_interval=target_update_interval, target_entropy=target_entropy, use_sde=use_sde,
                         sde_sample_freq=sde_sample_freq, use_sde_at_warmup=use_sde_at_warmup,
                         tensorboard_log=tensorboard_log, create_eval_env=create_eval_env,
                         verbose=verbose, seed=seed, device=device, _init_setup_model=_init_setup_model,
                         wandb_log=wandb_log, wandb_config = {'project_name': project_name, 'run_name': run_name,
                                                              'algo_name': self.algo_name})

    def run(self, plot=False, make_dataset=False):
        callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=self.log_dir)
        self.model.learn(total_timesteps=self.total_steps, callback=callback)
        self.model.save(f'../output/{self.algo_name}/{self.env_name}/{datetime.now()}_{self.algo_name}_trained_for_{self.total_steps}_steps')

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


class SaveOnBestTrainingRewardCallback(BaseCallback):
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(f"Best mean reward: {self.best_mean_reward:.2f} - "
                          f"Last mean reward per episode: {mean_reward:.2f}")

                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path}.zip")
                    self.model.save(self.save_path)
        return True

def moving_average(values, window):
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def plot_results(log_folder, title='Learning Curve'):
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    y = moving_average(y, window=50)
    x = x[len(x) - len(y):]
    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")
    plt.show()
