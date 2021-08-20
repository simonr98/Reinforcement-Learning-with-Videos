import numpy as np
import torch as th
import torch.nn as nn
import wandb
import pickle

from typing import Any, Dict, Optional, Union
from torch.nn import functional as F
from RLV.torch_rlv.models.inverse_model_network import InverseModelNetwork
from RLV.torch_rlv.buffer.buffers import ReplayBuffer
from RLV.torch_rlv.algorithms.sac.sac import SAC
from stable_baselines3.common.noise import ActionNoise
from RLV.torch_rlv.buffer.type_aliases import ReplayBufferSamples
from RLV.torch_rlv.data.acrobot_human_data.adapter_acrobot import AcrobotAdapter
from RLV.torch_rlv.data.acrobot_sac_data.sac_adapter_acrobot import AdapterSAC
from stable_baselines3.common.utils import polyak_update
from RLV.torch_rlv.models.convnet import ConvNet
from RLV.torch_rlv.models.discriminator import DiscriminatorNetwork
from RLV.torch_rlv.data.pusher_simulated_data.adapter_visual_img_data import AdapterVisualImgData
from RLV.torch_rlv.buffer.action_free_buffer import ActionFreeReplayBuffer


class RLV(SAC):
    def __init__(self, env_name, total_steps, warmup_steps=1500, beta_inverse_model=0.0003, policy='MlpPolicy',
                 env=None, learning_rate=0.0003, buffer_size=1000000, learning_starts=1000, batch_size=256,
                 tau=0.005, gamma=0.99, train_freq=1, gradient_steps=1, optimize_memory_usage=False, ent_coef='auto',
                 target_update_interval=10, target_entropy='auto', initial_exploration_steps=1000, wandb_log=False,
                 domain_shift=True, action_noise: Optional[ActionNoise] = None,
                 replay_buffer_class: Optional[ReplayBuffer] = None,
                 replay_buffer_kwargs: Optional[Dict[str, Any]] = None, use_sde: bool = False, sde_sample_freq: int = -1,
                 use_sde_at_warmup: bool = False, tensorboard_log: Optional[str] = None, create_eval_env: bool = False,
                 policy_kwargs: Dict[str, Any] = None, verbose: int = 0,
                 seed: Optional[int] = None,
                 device: Union[th.device, str] = "auto",
                 _init_setup_model: bool = True,
                 ):
        super(RLV, self).__init__(
            policy=policy,
            env=env,
            env_name=env_name,
            total_steps = total_steps,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=(batch_size * 2),
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            target_entropy=target_entropy,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            optimize_memory_usage=optimize_memory_usage,
            ent_coef=ent_coef
        )

        self.half_batch_size = batch_size

        self.target_update_interval = target_update_interval
        self.wandb_log = wandb_log
        self.inverse_model_loss = 0
        self.warmup_steps = warmup_steps
        self.beta_inverse_model = beta_inverse_model
        self.domain_shift = domain_shift

        self.initial_exploration_steps = initial_exploration_steps
        self.n_actions = env.action_space.shape[-1]

        self.inverse_model = InverseModelNetwork(
            beta=beta_inverse_model,
            input_dims=env.observation_space.shape[-1] * 2,
            output_dims=env.action_space.shape[-1],
            fc1_dims=64, fc2_dims=64, fc3_dims=64)

        if self.domain_shift:
            self.encoder = ConvNet(output_dims=self.env.observation_space.shape[-1])
            self.discriminator = DiscriminatorNetwork(input_dims=self.env.observation_space.shape[-1],
                                                      beta=3e-8, fc1_dims=64, fc2_dims=64, fc3_dims=64)
            # Optimizers
            self.encoder_optimizer = th.optim.Adam(self.encoder.parameters(), lr=3e-8)
            self.discriminator_optimizer = th.optim.Adam(self.discriminator.parameters(), lr=3e-8)

            # loss
            self.domain_shift_loss = nn.BCELoss()

            # data
            self.simulation_data = AdapterVisualImgData()

        if self.env_name == 'acrobot_continuous':
            self.action_free_replay_buffer = ReplayBuffer(
                buffer_size=buffer_size, observation_space=env.observation_space,
                action_space=env.action_space, device='cpu', n_envs=1,
                optimize_memory_usage=optimize_memory_usage, handle_timeout_termination=False)
        else:
            self.action_free_replay_buffer = \
                ActionFreeReplayBuffer(observation=self.simulation_data.observation,
                                       observation_img=self.simulation_data.observation_img,
                                       observation_img_raw=self.simulation_data.observation_img_raw,
                                       done=self.simulation_data.done)


    def fill_action_free_buffer(self, human_data=False, num_steps=200000, replay_buffer=None):
        if human_data:
            data = AcrobotAdapter(data_type='unpaired', env_name='Acrobot')
            observations = data.observations
            next_observations = data.next_observations
            actions = data.actions
            rewards = data.rewards
            terminals = data.terminals

            for i in range(0, observations.shape[0]):
                self.action_free_replay_buffer.add(
                    obs=observations[i],
                    next_obs=next_observations[i],
                    action=actions[i],
                    reward=rewards[i],
                    done=terminals[i],
                    infos={'': ''}
                )
        else:
            if replay_buffer is not None:
                print('Training done')

                data = {'observations': replay_buffer.observations, 'actions': replay_buffer.actions,
                        'next_observations': replay_buffer.next_observations, 'rewards': replay_buffer.rewards,
                        'terminals': replay_buffer.dones}

                with open(f"../data/sac_data/data_from_sac_trained_for_{num_steps}_steps.pickle", 'wb') \
                        as handle:
                    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

                self.action_free_replay_buffer = replay_buffer
            else:
                print('Use Data generated by training SAC for action free buffer')
                data = AdapterSAC()
                observations = data.observations
                next_observations = data.next_observations
                actions = data.actions
                rewards = data.rewards
                terminals = data.terminals

                ind = [0]
                for i in range(0, observations.shape[0]):
                    if int(terminals[i][0]) == 1:
                        ind.append(i)
                    if int(terminals[i][0]) == 1 and rewards[i] > -1.0:
                        start = ind[-2] + 1

                        for k in range(0, i - start + 1):
                            self.action_free_replay_buffer.add(
                                obs=observations[start+k],
                                next_obs=next_observations[start+k],
                                action=actions[start+k],
                                reward=rewards[start+k],
                                done=terminals[start+k],
                                infos={'': ''}
                            )

    def set_reward_acrobot(self, reward_obs):
        if reward_obs > -1:
            return 10
        else:
            return -1


    def warmup_inverse_model(self):
        "Loss inverse model:"
        for x in range(0, self.warmup_steps):
            obs_data = self.action_free_replay_buffer.sample(batch_size=self.half_batch_size)
            state_obs = obs_data.observations
            target_action = obs_data.actions
            next_state_obs = obs_data.next_observations

            input_inverse_model = th.cat((state_obs, next_state_obs), dim=1)
            action_obs = self.inverse_model.forward(input_inverse_model)

            self.inverse_model_loss = self.inverse_model.criterion(action_obs, target_action)

            self.inverse_model.optimizer.zero_grad()
            self.inverse_model_loss.backward()
            self.inverse_model.optimizer.step()

            if x % 100 == 0:
                print(f"Steps {x}, Loss: {self.inverse_model_loss.item()}")


    def warmup_encoder(self):
        for x in range(0, self.warmup_steps):
            lower_bound = np.random.randint(0, high=self.simulation_data.n - self.half_batch_size, size=None, dtype=int)
            upper_bound = lower_bound + self.half_batch_size

            observation = self.simulation_data.observation[lower_bound:upper_bound]
            observation_img = self.simulation_data.observation_img[lower_bound:upper_bound]

            output_encoder = self.encoder.forward(observation_img.float())

            loss = self.encoder.criterion(output_encoder, observation)

            self.encoder_optimizer.zero_grad()
            loss.backward()
            self.encoder_optimizer.step()

            if x % 100 == 0:
                print(f"Steps {x}, Encoder Loss: {loss.item()}")


    def train_encoder(self, observation, observation_img, observation_img_raw):
        self.encoder_optimizer.zero_grad()

        encoder_input, encoder_target, true_labels = observation_img, observation, th.ones(self.half_batch_size)
        encoder_output = self.encoder.forward(encoder_input)

        paired_loss = th.cdist(self.encoder.forward(observation_img_raw), self.encoder.forward(observation_img), p=2.0) ** 2

        # Train the generator
        generator_discriminator_out = self.discriminator.forward(encoder_output)
        # TODO: check if paired loss is added here or below
        generator_loss = self.domain_shift_loss(generator_discriminator_out, true_labels)
        generator_loss.backward()
        self.encoder_optimizer.step()

        # Train the discriminator on the true/generated data
        self.discriminator_optimizer.zero_grad()
        true_discriminator_out = self.discriminator.forward(encoder_target)
        true_discriminator_loss = self.domain_shift_loss(true_discriminator_out, true_labels)

        # add .detach() here think about this
        generator_discriminator_out = self.discriminator.forward(encoder_output.detach())
        generator_discriminator_loss = self.domain_shift_loss(generator_discriminator_out, th.zeros(self.half_batch_size))
        #TODO: check if paired loss is added here or above
        discriminator_loss = ((true_discriminator_loss + generator_discriminator_loss) / 2 + paired_loss) * 0.001
        discriminator_loss.backward()
        self.discriminator_optimizer.step()


    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Update optimizers learning rate
        global logging_parameters
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        for gradient_step in range(gradient_steps):

            # get robot data - sample from replay pool from the SAC model
            data_int = self.replay_buffer.sample(self.half_batch_size, env=self._vec_normalize_env)

            if self.env_name == 'acrobot_continuous':
                obs_data = self.action_free_replay_buffer.sample(batch_size=self.half_batch_size)
                state_obs = obs_data.observations
                target_action = obs_data.actions
                next_state_obs = obs_data.next_observations
                done_obs = obs_data.dones

                # get predicted action from inverse model
                input_inverse_model = th.cat((state_obs.detach(), next_state_obs.detach()), dim=1)
                action_obs = self.inverse_model.forward(input_inverse_model)

                # Compute inverse model loss
                self.inverse_model_loss = self.inverse_model.criterion(action_obs, target_action)

                # set rewards for observational data
                reward_obs = th.zeros(self.half_batch_size, 1)
                for i in range(0, self.half_batch_size):
                    reward_obs[i] = self.set_reward_acrobot(reward_obs=reward_obs[i])

                # replace the data used in SAC for each gradient steps by observational plus robot data
                replay_data = ReplayBufferSamples(
                    observations=th.cat((data_int.observations, state_obs.detach()), dim=0),
                    actions=th.cat((data_int.actions, action_obs.detach()), dim=0),
                    next_observations=th.cat((data_int.next_observations, next_state_obs.detach()), dim=0),
                    dones=th.cat((data_int.dones, done_obs.detach()), dim=0),
                    rewards=th.cat((data_int.rewards, reward_obs.detach()), dim=0)
                )

            else:
                state_obs, state_obs_img, state_obs_img_raw, next_state_obs, next_state_obs_img, \
                next_state_obs_img_raw, done_obs = self.action_free_replay_buffer.sample(batch_size=self.half_batch_size)
                # get
                obs_int, action_int, next_obs_int, reward_int, done_int = data_int.observations, data_int.actions, \
                                                                          data_int.next_observations, data_int.rewards, \
                                                                          data_int.dones


                # Get domain invariant encodings
                h_int, h_int_next = self.encoder.forward(obs_int), self.encoder.forward(next_obs_int)

                h_obs, h_obs_next = self.encoder.forward(state_obs_img), self.encoder.forward(next_state_obs_img)


                #Inverse Model

                # inputs
                int_input_inverse_model = th.cat((h_int, h_int_next), dim=1)
                obs_input_inverse_model = th.cat((h_obs, h_obs_next), dim=1)

                #outputs
                predicted_int_action = self.inverse_model.forward(int_input_inverse_model)
                predicted_obs_action = self.inverse_model.forward(obs_input_inverse_model)

                self.inverse_model_loss = self.inverse_model.criterion(predicted_int_action, action_int)


                # set rewards for observational data
                reward_obs = th.zeros(self.half_batch_size, 1)

                for i in range(0,self.half_batch_size):
                    if done_obs[i]:
                        reward_obs[i] = 10



                # replace the data used in SAC for each gradient steps by observational plus robot data
                replay_data = ReplayBufferSamples(
                    observations=th.cat((h_int, h_obs), dim=0),
                    actions=th.cat((action_int, predicted_obs_action), dim=0),
                    next_observations=th.cat((h_int_next, h_obs_next), dim=0),
                    dones=th.cat((done_int, done_obs), dim=0),
                    rewards=th.cat((reward_int, reward_obs), dim=0)
                )

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                # Select action according to policy
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                # Compute the next Q values: min over all critics targets
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                # add entropy term
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                # td error + entropy term
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = 0.5 * sum([F.mse_loss(current_q, target_q_values) for current_q in current_q_values])
            critic_losses.append(critic_loss.item())

            # Optimize Critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Mean over all critic networks
            q_values_pi = th.cat(self.critic.forward(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            # Optimize Actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)

            # Optimize Inverse Model
            self.inverse_model.optimizer.zero_grad()
            self.inverse_model_loss.backward()
            self.inverse_model.optimizer.step()

            if not self.env_name == 'acrobot_continuous':
                self.train_encoder(observation=state_obs, observation_img=state_obs_img, observation_img_raw=state_obs_img_raw)

        self._n_updates += gradient_steps

        if self.wandb_log:
            logging_parameters = {
                "train/n_updates": self._n_updates,
                "train/ent_coef": np.mean(ent_coefs),
                "train/actor_loss": np.mean(actor_losses),
                "train/critic_loss": np.mean(critic_losses),
            }

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/inverse_model_loss", self.inverse_model_loss.item())

        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
            if self.wandb_log:
                logging_parameters["train/ent_coef_loss"] = np.mean(ent_coef_losses)

        if self.wandb_log:
            wandb.log(logging_parameters)