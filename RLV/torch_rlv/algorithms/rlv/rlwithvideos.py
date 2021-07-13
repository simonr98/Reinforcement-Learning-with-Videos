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
from RLV.torch_rlv.algorithms.sac.softactorcritic import SoftActorCritic


class RlWithVideos(SoftActorCritic):
    def __init__(self, policy='MlpPolicy', env_name=None, config=None, wandb_log=False, env=None, learning_rate=0.0003,
                 buffer_size=1000000, learning_starts=1000, batch_size=256, tau=0.005, gamma=0.99, train_freq=1,
                 gradient_steps=1, optimize_memory_usage=False, ent_coef='auto', target_update_interval=1,
                 target_entropy='auto', use_sde=False, sde_sample_freq=- 1, use_sde_at_warmup=False,
                 tensorboard_log=None, create_eval_env=False, policy_kwargs=None, verbose=0, seed=None, device='auto',
                 _init_setup_model=True, project_name='sac_experiment', run_name='test_sac'):

        super().__init__(policy, env_name, config, wandb_log, env, learning_rate, buffer_size, learning_starts,
                         batch_size, tau, gamma, train_freq, gradient_steps, optimize_memory_usage, ent_coef,
                         target_update_interval, target_entropy, use_sde, sde_sample_freq, use_sde_at_warmup,
                         tensorboard_log, create_eval_env, policy_kwargs, verbose, seed, device, _init_setup_model,
                         project_name, run_name)

        action_noise = NormalActionNoise(mean=np.zeros(self.n_actions), sigma=0.1 * np.ones(self.n_actions))

        self.model = RLV(warmup_steps=500, beta_inverse_model=0.0003, env_name=env_name, policy=policy,
                         env=self.env, learning_rate=learning_rate, buffer_size=buffer_size,
                         learning_starts=learning_starts, batch_size=batch_size, tau=tau, gamma=gamma,
                         train_freq=train_freq, gradient_steps=gradient_steps, action_noise=action_noise,
                         optimize_memory_usage=optimize_memory_usage, ent_coef=ent_coef,
                         target_update_interval=target_update_interval, target_entropy=target_entropy, use_sde=use_sde,
                         sde_sample_freq=sde_sample_freq, use_sde_at_warmup=use_sde_at_warmup,
                         initial_exploration_steps=1000, domain_shift=False,
                         domain_shift_generator_weight=0.01, domain_shift_discriminator_weight=0.01,
                         create_eval_env=create_eval_env, tensorboard_log=tensorboard_log,
                         paired_loss_scale=1.0, verbose=verbose, seed=seed, device=device,
                         _init_setup_model=_init_setup_model)

        if wandb_log:
            self.logger = wandb.init(project=project_name,
                                     config=self.config,
                                     name=run_name,
                                     reinit=True,  # allow things to be run multiple times
                                     settings=wandb.Settings(start_method="thread"))

    def run(self, total_timesteps=int(250000), plot=False):
        self.model.fill_action_free_buffer()
        self.model.inverse_model.warmup()

        callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=self.log_dir)
        self.model.learn(total_timesteps=total_timesteps, callback=callback)
        self.total_timesteps =+ total_timesteps

        if plot:
            plot_results(self.log_dir)
