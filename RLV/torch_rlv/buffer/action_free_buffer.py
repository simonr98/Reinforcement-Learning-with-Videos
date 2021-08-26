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
        lower_bound = np.random.randint(0, high=self.n - batch_size, size=None, dtype=int)
        upper_bound = lower_bound + batch_size

        return self.observation[lower_bound:upper_bound], self.observation_img[lower_bound:upper_bound], \
               self.observation_img_raw[lower_bound:upper_bound], self.next_observation[lower_bound:upper_bound], \
               self.next_observation_img[lower_bound:upper_bound], \
               self.next_observation_img_raw[lower_bound:upper_bound],\
               self.done[lower_bound:upper_bound]




