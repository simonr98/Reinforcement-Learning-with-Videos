import numpy as np

class PairedBuffer():
    def __init__(self, observation, observation_img, observation_img_raw):
        self.n = len(observation_img) - 1
        self.observation = observation
        self.observation_img = observation_img
        self.observation_img_raw = observation_img_raw

    def sample(self, batch_size=256):
        batch = np.random.choice(self.n, batch_size)
        return self.observation[batch], self.observation_img[batch], self.observation_img_raw[batch]