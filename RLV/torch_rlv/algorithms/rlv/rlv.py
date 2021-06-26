import torch as T
from RLV.torch_rlv.models.inverse_model import InverseModelNetwork
from RLV.torch_rlv.algorithms.sac.sac_old.sac import SAC
from RLV.torch_rlv.visualizer.plot import plot_learning_curve, plot_env_step, animate_env_obs
import numpy as np
from datetime import datetime


class RLV(SAC):
    def __init__(self, warmup_steps=500, beta_inverse_model=0.0003,
                 policy='MlpPolicy', env=None, learning_rate=0.0003, buffer_size=1000000, learning_starts=1000,
                 batch_size=256, tau=0.005, gamma=0.99, train_freq=1, gradient_steps=1, optimize_memory_usage=False,
                 ent_coef='auto', target_update_interval=1, target_entropy='auto'):
        super().__init__(policy, env, learning_rate, buffer_size, learning_starts,
                         batch_size, tau, gamma, train_freq, gradient_steps, optimize_memory_usage,
                         ent_coef, target_update_interval, target_entropy)
        self.warmup_steps = warmup_steps
        self.beta_inverse_model = beta_inverse_model
        self.inverse_model = InverseModel(observation_space=env.observation_space.shape[-1],
                                          action_space=env.action_space.shape[-1],
                                          env=self.env)
        policy.device = device
        self.log_dir = "/tmp/gym/"
        os.makedirs(self.log_dir, exist_ok=True)

        self.env = env
        self.env = Monitor(env, self.log_dir)

        self.n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(self.n_actions), sigma=0.1 * np.ones(self.n_actions))
        self.callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=self.log_dir)

        self.model = SAC(policy, self.env, learning_rate=learning_rate, buffer_size=buffer_size,
                         learning_starts=learning_starts, batch_size=batch_size, tau=tau, gamma=gamma,
                         train_freq=train_freq, gradient_steps=gradient_steps, action_noise=action_noise,
                         optimize_memory_usage=optimize_memory_usage, ent_coef=ent_coef,
                         target_update_interval=target_update_interval, target_entropy=target_entropy, use_sde=use_sde,
                         sde_sample_freq=sde_sample_freq, use_sde_at_warmup=use_sde_at_warmup,
                         tensorboard_log=tensorboard_log, create_eval_env=create_eval_env, policy_kwargs=policy_kwargs,
                         verbose=verbose, seed=seed, device=device, _init_setup_model=_init_setup_model)

        def run(self):
            self.inverse_model.warmup(warmup_steps=self.warmup_steps)

            # add encoder
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# def set_reward(reward):
#     if reward == -1:
#         return -1
#     else:
#         return 10
#
#
#
#
#
#
# class RLV_:
#     def __init__(self, env_name, env, agent, iterations=500, warmup_steps=500, base_algorithm=None, lr=0.003,
#                  experiment_name='RLV', pre_steps=1000, pre_training_steps=25000):
#         super(RLV, self).__init__()
#         self.experiment_name = experiment_name
#         self.env_name = env_name
#         self.env = env
#         self.steps_count = 0
#         self.lr = lr
#         self.pre_steps = pre_steps
#         self.warmup_steps = warmup_steps
#         self.pre_training_steps = pre_training_steps
#         self.score_history = []
#         self.agent = agent
#         self.inverse_model = InverseModelNetwork(beta=0.0003, input_dims=13)
#         self.iterations = iterations  # TODO
#         self.filename = env_name + '.png'
#         self.figure_file = 'output/plots/RLV_' + self.filename
#         self.date_time = datetime.now().strftime("%m_%d_%Y_%H:%M")
#         self.algorithm = base_algorithm
#
#     def get_action_free_buffer(self):
#         steps = 0
#         pre_train_agent = self.agent
#         pre_training = SAC(env_name=self.env_name, env=self.env, agent=pre_train_agent,
#                            n_games=20000, pre_steps=100, lr=self.lr, experiment_name=self.experiment_name)
#         while steps < self.pre_training_steps:
#             pre_training.run()
#             steps = pre_training.steps_count
#         return pre_training.agent.memory
#
#     def warmup_inverse_model(self, warmup_steps):
#         for itrn in range(0, warmup_steps):
#             state_obs, target, reward, next_state_obs, done_obs \
#                 = self.agent.memory_action_free.sample_buffer(self.agent.batch_size)
#             done_obs = np.reshape(done_obs, (256, 1))
#
#             # get actions and rewards for observational data
#             input_inverse_model = T.cat((T.from_numpy(state_obs), T.from_numpy(next_state_obs),
#                                          T.from_numpy(done_obs)), dim=1).float()
#
#             action_obs_t = self.inverse_model(input_inverse_model)
#             reward_obs = np.zeros((self.agent.batch_size, 1))
#             for __ in range(0, self.agent.batch_size):
#                 reward_obs[__] = set_reward(reward[__])
#
#             target_t = T.from_numpy(target).float()
#             self.inverse_model.optimizer.zero_grad()
#             loss = self.inverse_model.criterion(action_obs_t, target_t)
#
#             if itrn % 50 == 0:
#                 print(f"Warmup Step: {itrn} - Loss Inverse Model: {loss}")
#
#             # Update Inverse Model
#             loss.backward()
#             self.inverse_model.optimizer.step()
#
#     def run(self, plot=False):
#         p_steps = self.pre_steps
#         self.agent.memory_action_free = self.get_action_free_buffer()
#
#         self.warmup_inverse_model(warmup_steps=self.warmup_steps)
#
#         for itrn in range(0, self.iterations):
#             state_obs, target, reward, next_state_obs, done_obs \
#                 = self.agent.memory_action_free.sample_buffer(self.agent.batch_size)
#             done_obs = np.reshape(done_obs, (256, 1))
#
#             # get actions and rewards for observational data
#             input_inverse_model = T.cat((T.from_numpy(state_obs), T.from_numpy(next_state_obs),
#                                          T.from_numpy(done_obs)), dim=1).float()
#
#             action_obs_t = self.inverse_model(input_inverse_model)
#
#             reward_obs = np.zeros((self.agent.batch_size, 1))
#             for __ in range(0, self.agent.batch_size):
#                 reward_obs[__] = set_reward(reward[__])
#
#             # define observational data
#             action_obs = action_obs_t.detach().numpy()
#
#             observational_batch = {
#                 'state': state_obs,
#                 'action': action_obs,
#                 'reward': reward_obs,
#                 'next_state': next_state_obs,
#                 'done_obs': done_obs
#             }
#
#             # Inverse Model
#             target_t = T.from_numpy(target).float()
#             self.inverse_model.optimizer.zero_grad()
#             loss = self.inverse_model.criterion(action_obs_t, target_t)
#             print(f"Iteration: {itrn} - Loss Inverse Model: {loss}")
#
#             rlv_args = {
#                 'experiment_name': 'rlv_exp_' + str(self.lr),
#                 'loss_inverse_model': loss,
#                 'warmup_steps': self.warmup_steps
#             }
#
#             # perform sac based on initial data obtained by environment step plus additional
#             # observational data
#             if self.algorithm is None:
#                 self.algorithm = SAC(env_name=self.env_name, env=self.env, agent=self.agent,
#                                      n_games=1, pre_steps=p_steps, score_history=self.score_history,
#                                      additional_data=observational_batch, steps_count=self.steps_count,
#                                      lr=self.lr, rlv_config=rlv_args, experiment_name=self.experiment_name)
#             # execute pre steps only in first iteration
#             if itrn > 0:
#                 self.algorithm.run(cnt=itrn, execute_pre_steps=False)
#             else:
#                 self.algorithm.run(cnt=itrn)
#
#             # update steps count of RLV based on steps executed in SAC
#             self.steps_count = self.algorithm.get_step_count()
#             self.score_history = self.algorithm.get_score_history()
#
#             p_steps = 0
#
#             # Update Inverse Model
#             loss.backward()
#             self.inverse_model.optimizer.step()
#
#         # Plot in pdf file with visualizer
#         if plot:
#             env_states = self.algorithm.get_env_states()
#             plot_env_step(env_states, 'output/plots/SAC_' + self.env_name
#                           + '_' + self.date_time)
#             observations = self.algorithm.get_env_obs()
#             animate_env_obs(observations, 'output/videos/RLV_' + self.env_name + '_' + self.date_time)
#             x = [i + 1 for i in range(len(self.score_history))]
#             plot_learning_curve(x, self.score_history, self.figure_file)
