import pickle
import pandas as pd
import os
import torch as T
import numpy as np
from PIL import Image


class AdapterVisualPusher:
    def __init__(self):
        current_directory = os.path.dirname(__file__)
        path = os.path.join(current_directory, '3000_SAC_steps_500_samples.pickle')
        self.data = pickle.load(open(path, 'rb'))

        self.n = len(self.data['observation'])

        self.observation = T.from_numpy(np.reshape(np.array(self.data['observation']), (self.n, 20)))
        self.action = T.from_numpy(np.reshape(np.array(self.data['action']), (self.n, 4)))
        self.next_observation = T.from_numpy(np.reshape(np.array(self.data['next_observation']), (self.n, 20)))
        self.reward = T.from_numpy(np.array(self.data["reward"]))
        self.done = T.from_numpy(np.reshape(np.array(self.data["done"]), (self.n, 1)))
        self.observation_img = T.from_numpy(np.reshape(np.array(self.data['observation_img']), (self.n, 3, 120, 120))).float()
        self.observation_img_raw = T.from_numpy(np.reshape(np.array(self.data['observation_img_raw']), (self.n, 3, 120, 120))).float()

if __name__ == '__main__':
    a = AdapterVisualPusher()

    print(a.observation.shape)
    print(a.action.shape)
    print(a.next_observation.shape)
    print(a.reward.shape)
    print(a.done.shape)