import numpy as np

class ActionFreeReplayBuffer():
    def __init__(self, observation, observation_img, observation_img_raw, done):
        self.n = len(observation) - 1
        self.observation = observation

        self.observation_img = observation_img

        self.observation_img_raw = observation_img_raw

        self.done = done[:-1]

        self.next_observation = self.observation[1:]
        self.observation = self.observation[:-1]

        self.next_observation_img = self.observation_img[1:]
        self.observation_img = self.observation_img[:-1]

        self.next_observation_img_raw = self.observation_img_raw[1:]
        self.observation_img_raw = self.observation_img_raw[:-1]

    def sample(self, batch_size=256):
        batch = np.random.choice(self.n, batch_size)
        obs = self.observation[batch]
        obs_img = self.observation_img[batch]
        obs_img_raw = self.observation_img_raw[batch]

        next_obs = self.next_observation[batch]
        next_obs_img = self.next_observation_img[batch]
        next_obs_img_raw = self.next_observation_img_raw[batch]
        done = self.done[batch]
        return obs, obs_img, obs_img_raw, next_obs, next_obs_img, next_obs_img_raw, done
