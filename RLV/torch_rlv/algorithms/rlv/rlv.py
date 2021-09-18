import numpy as np
import torch as th
import torch.nn as nn
import wandb
import pickle

from pathlib import Path

from typing import Any, Dict, Optional, Union
from torch.nn import functional as F
from torch import autograd
from RLV.torch_rlv.models.inverse_model_network import InverseModelNetwork
from RLV.torch_rlv.utils.buffers import ReplayBuffer
from RLV.torch_rlv.algorithms.sac.sac import SAC
from stable_baselines3.common.noise import ActionNoise
from RLV.torch_rlv.utils.type_aliases import ReplayBufferSamples
from RLV.torch_rlv.data.acrobot_paper_data.adapter_acrobot import AcrobotAdapterPaper
from RLV.torch_rlv.data.acrobot_continuous_data.adapter_acrobot import AcrobotAdapter
from stable_baselines3.common.utils import polyak_update
from RLV.torch_rlv.models.convnet import ConvNet
from RLV.torch_rlv.models.discriminator import DiscriminatorNetwork
from RLV.torch_rlv.data.visual_pusher_data.adapter_visual_pusher import AdapterVisualPusher
from RLV.torch_rlv.utils.action_free_buffer import ActionFreeReplayBuffer
from RLV.torch_rlv.utils.paired_buffer import PairedBuffer
from RLV.torch_rlv.data.visual_pusher_data.adapter_paired_data import AdapterPairedData


class RLV(SAC):
    def __init__(self, env_name, total_steps, warmup_steps=1500, beta_inverse_model=0.0003, policy='MlpPolicy',
                 env=None, learning_rate=0.0003, buffer_size=1000000, learning_starts=1000, batch_size=256,
                 tau=0.005, gamma=0.99, train_freq=1, gradient_steps=1, optimize_memory_usage=False, ent_coef='auto',
                 target_update_interval=10, target_entropy='auto', wandb_log=False, project_name='rlv',
                 domain_shift=True, device: Union[th.device, str] = "auto", _init_setup_model: bool = True,
                 wandb_logging_parameters={}, wandb_config={}, verbose=1):
        super(RLV, self).__init__(
            env_name=env_name, total_steps=total_steps, policy=policy, env=env, learning_rate=learning_rate,
            buffer_size=buffer_size, learning_starts=learning_starts, batch_size=batch_size, tau=tau, gamma=gamma,
            train_freq=train_freq, gradient_steps=gradient_steps, optimize_memory_usage=optimize_memory_usage,
            ent_coef=ent_coef, target_update_interval=target_update_interval, wandb_config=wandb_config,
            target_entropy=target_entropy, wandb_log=wandb_log, device=device, _init_setup_model=_init_setup_model,
            verbose=verbose)

        self.half_batch_size = batch_size
        self.target_update_interval = target_update_interval

        self.wandb_log = wandb_log
        self.wandb_logging_parameters = wandb_logging_parameters

        self.inverse_model_loss = 0
        self.encoder_loss = 0
        self.warmup_steps = warmup_steps
        self.beta_inverse_model = beta_inverse_model
        self.inverse_model = InverseModelNetwork(beta=beta_inverse_model, input_dims=env.observation_space.shape[-1] * 2,
                                                 output_dims=env.action_space.shape[-1], fc1_dims=64, fc2_dims=64,
                                                 fc3_dims=64).to(self.device)

        self.domain_shift = domain_shift
        if self.domain_shift:
            self.encoder = ConvNet(output_dims=self.env.observation_space.shape[-1]).to(self.device)
            self.discriminator = DiscriminatorNetwork(input_dims=self.env.observation_space.shape[-1],
                                                      beta=3e-8, fc1_dims=64, fc2_dims=64, fc3_dims=64).to(self.device)
            # Optimizers
            self.encoder_optimizer = th.optim.Adam(self.encoder.parameters(), lr=3e-8)
            self.discriminator_optimizer = th.optim.Adam(self.discriminator.parameters(), lr=3e-8)

            # loss
            self.domain_shift_loss = nn.BCELoss()
            self.paired_loss = nn.MSELoss()

            # data
            simulation_data = AdapterVisualPusher()
            paired_data = AdapterPairedData()
            self.paired_buffer = PairedBuffer(observation=paired_data.observation.to(self.device),
                                              observation_img=paired_data.observation_img.to(self.device),
                                              observation_img_raw=paired_data.observation_img_raw.to(self.device))


        self.action_free_replay_buffer = ReplayBuffer(buffer_size=buffer_size,
                                                      observation_space=env.observation_space,
                                                      action_space=env.action_space, device=self.device, n_envs=1,
                                                      optimize_memory_usage=optimize_memory_usage,
                                                      handle_timeout_termination=False) \
            if self.env_name == 'acrobot_continuous' \
            else ActionFreeReplayBuffer(observation=simulation_data.observation.to(self.device),
                                        observation_img=simulation_data.observation_img.to(self.device),
                                        observation_img_raw=simulation_data.observation_img_raw.to(self.device),
                                        done=simulation_data.done.to(self.device))


    def fill_action_free_buffer_acrobot(self, paper_data=False, num_steps=200000, replay_buffer=None):
        data = AcrobotAdapterPaper() if paper_data else AcrobotAdapter()

        observations = data.observations
        next_observations = data.next_observations
        actions = data.actions
        rewards = data.rewards
        terminals = data.terminals

        for i in range(0, observations.shape[0]):
            self.action_free_replay_buffer.add(obs=observations[i], next_obs=next_observations[i], action=actions[i],
                                               reward=rewards[i], done=terminals[i], infos={'': ''})

    @staticmethod
    def set_reward_acrobot(done_obs):
        reward = 10 if done_obs > 0 else -1
        return reward

    def warmup_inverse_model(self):
        if self.env_name == 'acrobot_continuous':
            for step in range(0, self.warmup_steps):
                obs_data = self.action_free_replay_buffer.sample(batch_size=self.half_batch_size)
                state_obs = obs_data.observations
                target_action = obs_data.actions
                next_state_obs = obs_data.next_observations

                input_inverse_model = th.cat((state_obs, next_state_obs), dim=1)
                action_obs = self.inverse_model(input_inverse_model)

                self.inverse_model_loss = self.inverse_model.criterion(action_obs, target_action)

                # optimize inverse model
                self.inverse_model.optimizer.zero_grad()
                self.inverse_model_loss.backward()
                self.inverse_model.optimizer.step()

                if step % 100 == 0:
                    print(f"Steps {step}, Loss: {self.inverse_model_loss.item()}")
        else:
            model = SAC.load("../data/visual_pusher_data/478666_sac_trained_for_500000_steps")

            env = self.env
            obs = env.reset()

            for step in range(self.warmup_steps * 10):
                action, state_ = model.predict(obs)
                next_obs, _, done, _ = env.step(action)
                target_action = th.from_numpy(action).to(self.device)

                input_inverse_model = th.cat((th.from_numpy(obs).to(self.device),
                                              th.from_numpy(next_obs).to(self.device)), dim=1)
                action_obs = self.inverse_model(input_inverse_model)

                self.inverse_model_loss = self.inverse_model.criterion(action_obs, target_action)

                # optimize inverse model
                self.inverse_model.optimizer.zero_grad()
                self.inverse_model_loss.backward()
                self.inverse_model.optimizer.step()

                obs = next_obs if not done else env.reset()

                if step % 1000 == 0:
                    print(f"Steps {step}, Loss: {self.inverse_model_loss.item()}")

    def warmup_encoder(self):
        for step in range(0, self.warmup_steps):
            observation, _, _, _, _, _, _ = self.action_free_replay_buffer.sample()
            _, state_obs_img, _, _, _, _, _ = self.action_free_replay_buffer.sample()
            self.train_encoder(observation=observation, observation_img=state_obs_img)

            if step % 300 == 0:
                print(f"Warmup Step {step} / {self.warmup_steps} Loss {self.encoder_loss}")


    def train_encoder(self, observation, observation_img):
#        with autograd.detect_anomaly():
        input_image, real_state, true_labels = observation_img, observation, \
                                               th.ones((self.half_batch_size,1), device=self.device)

        fake_state = self.encoder(input_image.float())

        # add paired data // obs = filtered, int = raw
        paired_obs, paired_img_obs, _ = self.paired_buffer.sample()
        paired_loss = self.paired_loss(paired_obs.float(), self.encoder(paired_img_obs.float()))

        # Train the discriminator on the true/generated data
        self.discriminator_optimizer.zero_grad()
        true_discriminator_out = self.discriminator(real_state.float())
        true_discriminator_loss = self.domain_shift_loss(true_discriminator_out.float(), true_labels.float())

        fake_discriminator_out = self.discriminator(fake_state)
        fake_discriminator_loss = self.domain_shift_loss(fake_discriminator_out,
                                                         th.zeros((self.half_batch_size, 1), device=self.device))

        # Optimize Discriminator Loss
        discriminator_loss = ((true_discriminator_loss + fake_discriminator_loss) / 2 + paired_loss) * 0.001
        discriminator_loss.backward(retain_graph=True)
        self.discriminator_optimizer.step()

        output = self.discriminator(fake_state)

        # Train the generator
        self.encoder_optimizer.zero_grad()
        generator_loss = self.domain_shift_loss(output, true_labels)
        generator_loss.backward()
        self.encoder_optimizer.step()

        self.encoder_loss = generator_loss


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
                action_obs = self.inverse_model(input_inverse_model)

                # Compute inverse model loss
                self.inverse_model_loss = self.inverse_model.criterion(action_obs, target_action)

                # set rewards for observational data
                reward_obs = th.zeros((self.half_batch_size, 1), device=self.device)
                for i in range(0, self.half_batch_size):
                    reward_obs[i] = self.set_reward_acrobot(done_obs=done_obs[i])

                # replace the data used in SAC for each gradient steps by observational plus robot data
                replay_data = ReplayBufferSamples(
                    observations=th.cat((data_int.observations, state_obs.detach()), dim=0),
                    actions=th.cat((data_int.actions, action_obs.detach()), dim=0),
                    next_observations=th.cat((data_int.next_observations, next_state_obs.detach()), dim=0),
                    dones=th.cat((data_int.dones, done_obs.detach()), dim=0),
                    rewards=th.cat((data_int.rewards, reward_obs.detach()), dim=0))

            # visual_pusher case


            else:
                state_obs, state_obs_img, state_obs_img_raw, next_state_obs, next_state_obs_img, \
                next_state_obs_img_raw, done_obs = self.action_free_replay_buffer.sample(batch_size=self.half_batch_size)

                obs_int, action_int, next_obs_int, reward_int, done_int = data_int.observations, data_int.actions, \
                                                                          data_int.next_observations, data_int.rewards, \
                                                                          data_int.dones


                # Get domain invariant encodings
                h_int, h_int_next = obs_int, next_obs_int
                h_obs, h_obs_next = self.encoder(state_obs_img.float()), self.encoder(next_state_obs_img.float())


                #Inverse Model
                # inputs
                int_input_inverse_model = th.cat((h_int, h_int_next), dim=1)
                obs_input_inverse_model = th.cat((h_obs, h_obs_next), dim=1)

                #outputs
                predicted_int_action = self.inverse_model(int_input_inverse_model.detach())
                predicted_obs_action = self.inverse_model(obs_input_inverse_model.detach())


                self.inverse_model_loss = self.inverse_model.criterion(predicted_int_action, action_int)
                reward_obs = th.zeros((self.half_batch_size, 1), device=self.device)

                for i in range(0,self.half_batch_size):
                    reward_obs[i] = 100 if done_obs[i] else 1

                # replace the data used in SAC for each gradient steps by observational plus robot data
                replay_data = ReplayBufferSamples(
                    observations=th.cat((h_int.detach(), h_obs.detach()), dim=0),
                    actions=th.cat((action_int, predicted_obs_action.detach()), dim=0),
                    next_observations=th.cat((h_int_next.detach(), h_obs_next.detach()), dim=0),
                    dones=th.cat((done_int, done_obs.detach()), dim=0),
                    rewards=th.cat((reward_int, reward_obs.detach()), dim=0)
                )

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            # Optimize entropy coefficient, also called entropy temperature or alpha in the paper
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

            # Get current Q-values estimates for each critic network - using action from the replay buffer
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = 0.5 * sum([F.mse_loss(current_q, target_q_values) for current_q in current_q_values])
            critic_losses.append(critic_loss.item())

            # Optimize Critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Compute actor loss
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

            if not self.env_name == 'acrobot_continuous':
                self.train_encoder(observation=h_int, observation_img=state_obs_img)

        self._n_updates += gradient_steps

        # logging
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

        if self.wandb_log:
            self.wandb_logging_parameters.update({
                'inverse_model_loss': self.inverse_model_loss.item(),
                'n_updates': self._n_updates,
                'ent_coef': np.mean(ent_coefs),
                'actor_loss': np.mean(actor_losses),
                'critic_loss': np.mean(critic_losses)
            })

        if self.domain_shift:
            self.logger.record("train/domain_shift_loss", self.domain_shift_loss)

        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
            if self.wandb_log:
                self.wandb_logging_parameters.update({"train/ent_coef_loss":np.mean(ent_coef_losses)})
