from RLV.torch_rlv.algorithms.rlv.rlv import RLV

import os

import numpy as np
import matplotlib.pyplot as plt
from RLV.torch_rlv.algorithms.sac.sac import SAC
from RLV.torch_rlv.utils.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from RLV.torch_rlv.utils.type_aliases import TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from RLV.torch_rlv.data.acrobot_continuous_data.adapter_acrobot import AcrobotAdapter
from RLV.torch_rlv.algorithms.sac.softactorcritic import SaveOnBestTrainingRewardCallback
from RLV.torch_rlv.algorithms.sac.softactorcritic import SoftActorCritic


class RlWithVideos(SoftActorCritic):
    def __init__(self, policy='MlpPolicy', env_name=None, config=None, wandb_log=False, env=None, learning_rate=0.0003,
                 learning_rate_inverse_model=0.0003, buffer_size=1000000, learning_starts=1000, batch_size=256,
                 tau=0.005, gamma=0.99, train_freq=1, gradient_steps=1, optimize_memory_usage=False, ent_coef='auto',
                 target_update_interval=1, target_entropy='auto', use_sde=False, sde_sample_freq=- 1,
                 use_sde_at_warmup=False, tensorboard_log=None, create_eval_env=False, policy_kwargs=None, verbose=1,
                 seed=None, device='auto', _init_setup_model=True, project_name='sac_experiment', run_name='test_sac',
                 acrobot_paper_data=False, log_dir='../output/tmp/gym/', total_steps=1000, algo_name='rlv'):

        super().__init__(policy=policy, env_name=env_name, env=env, learning_rate=learning_rate, buffer_size=buffer_size,
                         learning_starts=learning_starts, batch_size=batch_size, tau=tau, gamma=gamma,
                         train_freq=train_freq, gradient_steps=gradient_steps, optimize_memory_usage=optimize_memory_usage,
                         ent_coef=ent_coef, target_update_interval=target_update_interval, target_entropy=target_entropy,
                         use_sde=use_sde, sde_sample_freq=sde_sample_freq, use_sde_at_warmup=use_sde_at_warmup,
                         tensorboard_log=tensorboard_log, create_eval_env=create_eval_env, policy_kwargs=policy_kwargs,
                         verbose=verbose, seed=seed, device=device, _init_setup_model=_init_setup_model,
                         project_name=project_name, run_name=run_name, wandb_log=wandb_log, algo_name=algo_name)

        #action_noise = NormalActionNoise(mean=np.zeros(self.n_actions), sigma=0.1 * np.ones(self.n_actions))

        self.acrobot_paper_data = acrobot_paper_data
        self.log_dir = log_dir
        self.total_steps = total_steps
        self.env_name=env_name

        domain_shift = False if self.env_name == 'acrobot_continuous' else True

        self.model = RLV(warmup_steps=1500, total_steps=total_steps, beta_inverse_model=learning_rate_inverse_model, env_name=env_name,
                         policy=policy, env=self.env, learning_rate=learning_rate, buffer_size=buffer_size,
                         learning_starts=learning_starts, batch_size=batch_size, tau=tau, gamma=gamma,
                         train_freq=train_freq, gradient_steps=gradient_steps, optimize_memory_usage=optimize_memory_usage,
                         ent_coef=ent_coef, target_update_interval=target_update_interval, target_entropy=target_entropy,
                         domain_shift=domain_shift, device=device, _init_setup_model=_init_setup_model, wandb_log=wandb_log,
                         wandb_config={'project_name': project_name, 'run_name': run_name, 'algo_name': self.algo_name})

    def run(self, total_timesteps=int(1000000), plot=False):
        if self.env_name == "acrobot_continuous":
            self.model.fill_action_free_buffer_acrobot(paper_data=self.acrobot_paper_data)
            self.model.warmup_inverse_model()
        else:
            self.model.warmup_encoder()

        callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=self.log_dir)
        self.model.learn(total_timesteps=total_timesteps, callback=callback, log_interval=4)
        self.model.save(f'/sac_models/trained_for_{self.total_steps}')

        if plot:
            plot_results(self.log_dir)
