import torch as T
from RLV.torch_rlv.models.inverse_model_network import InverseModelNetwork
from stable_baselines3 import SAC
from RLV.torch_rlv.visualizer.plot import plot_learning_curve, plot_env_step, animate_env_obs
import numpy as np
from datetime import datetime


class InverseModel:
    def __init__(self, beta=0.003, observations_space=None, action_space=None, batch_size=256, env=None):
        self.beta = beta
        self.observation_space = observations_space
        self.action_space = action_space
        self.batch_size = batch_size
        self.env = env
        self.input_dims = observations_space * 2
        self.output_dims = action_space
        self.network = InverseModelNetwork(beta=self.beta, input_dims=self.input_dims)
        self.loss = 0

    def warmup(self, warmup_steps):
        obs = self.env.reset()

        for _ in range(0, warmup_steps):
            target_action = self.env.action_space.sample()
            obs_, reward, done, info = self.env.step(target_action)

            input = T.cat(T.from_numpy(obs), T.from_numpy(obs_), dim=1).float()

            obs_action_t = self.network(input)

            target_action_t = T.from_numpy(target_action).float()

            self.network.optimizer.zero_grad()

            self.loss = self.calculate_loss(obs_action_t, target_action_t)

            if _ % 50 == 0:
                print(f"Warmup Step: {_} - Loss Inverse Model: {loss}")

            # Update Inverse Model
            self.update()

    def calculate_loss(self, obs_action, target_act):
        self.loss = self.network.criterion(obs_action, target_act)
        return self.loss

    def update(self):
        self.loss.backward()
        self.network.optimizer.step()