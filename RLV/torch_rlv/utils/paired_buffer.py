import numpy as np

class PairedBuffer():
    def __init__(self, observation_img, observation_img_raw):
        self.n = len(observation_img) - 1
        self.observation_img = observation_img
        self.observation_img_raw = observation_img_raw

    def sample(self, batch_size=256):
        lower_bound = np.random.randint(0, high=self.n - batch_size, size=None, dtype=int)
        upper_bound = lower_bound + batch_size

        return self.observation_img[lower_bound:upper_bound], self.observation_img_raw[lower_bound:upper_bound]