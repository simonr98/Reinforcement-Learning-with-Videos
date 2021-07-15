from typing import Any

import torch as T
from RLV.torch_rlv.models.inverse_model_network import InverseModelNetwork
from stable_baselines3 import SAC
from RLV.torch_rlv.visualizer.plot import plot_learning_curve, plot_env_step, animate_env_obs
import numpy as np
from datetime import datetime


class InverseModel:
    def __init__(self, beta=0.0003, observation_space_dims=3, action_space_dims=1, batch_size=256, env=None,
                 warmup_steps=500):
        self.beta = beta
        self.observation_space_dims = observation_space_dims
        self.warmup_steps = warmup_steps
        self.action_space_dims = action_space_dims
        self.batch_size = batch_size
        self.env = env
        self.input_dims = observation_space_dims * 2
        self.output_dims = action_space_dims
        self.network = InverseModelNetwork(beta=self.beta, input_dims=self.input_dims,
                                           output_dims=self.action_space_dims)
        self.loss = 0

    def warmup(self):
        obs = np.zeros((self.batch_size, self.observation_space_dims))
        target_action = np.zeros((self.batch_size, self.action_space_dims))
        obs_ = np.zeros((self.batch_size, self.observation_space_dims))
        reward = np.zeros((self.batch_size, 1))
        done = np.zeros((self.batch_size, 1))

        obs[0] = self.env.reset()

        print("Inverse Model Warmup")

        for s in range(0, self.warmup_steps):
            for c in range(0, self.batch_size-1):
                target_action[c] = self.env.action_space.sample()
                obs_[c], reward[c], done[c], info = self.env.step(target_action[c])

                if done[c]:
                    obs[c+1] = self.env.reset()
                else:
                    obs[c+1] = obs_[c]

            input = T.cat((T.from_numpy(obs), T.from_numpy(obs_)), dim=1).float()

            obs_action_t = self.network(input)

            target_action_t = T.from_numpy(target_action).float()

            self.loss = self.calculate_loss(obs_action_t, target_action_t)

            if s % 50 == 0:
                print(f"Step: {s} --- Loss: {self.loss}")

            # Update Inverse Model
            self.update()

    def calculate_loss(self, obs_action, target_act):
        self.loss = self.network.criterion(obs_action, target_act)
        return self.loss

    def update(self):
        self.network.optimizer.zero_grad()
        self.loss.backward()
        self.network.optimizer.step()
