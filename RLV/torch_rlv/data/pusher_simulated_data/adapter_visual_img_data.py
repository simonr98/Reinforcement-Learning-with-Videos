import pickle
import pandas as pd
import os
import numpy as np
from PIL import Image


class AdapterVisualImgData:
    def __init__(self):
        current_directory = os.path.dirname(__file__)
        path = os.path.join(current_directory, 'simulated_pusher_data_4000_steps.pickle')

        self.data = pickle.load(open(path, 'rb'))
        self.n = len(self.data['observation'])
        self.observation = np.reshape(np.array(self.data['observation']), (self.n, 32))
        self.observation_img = np.reshape(np.array(self.data['observation_img']), (self.n, 3, 120, 120))
        self.action = np.reshape(np.array(self.data['action']), (self.n, 4))
        self.next_observation = np.reshape(np.array(self.data['next_observation']), (self.n, 32))
        self.reward = np.array(self.data["reward"])
        self.done = np.array(self.data["done"])

if __name__ == '__main__':
    a = AdapterVisualImgData()
    print(a.observation.shape)
    print(a.observation_img.shape)
    print(a.action.shape)
    print(a.next_observation.shape)
    print(a.reward.shape)
    print(a.done.shape)


    x = np.reshape(a.observation_img[0], (120, 120, 3))
    x = Image.fromarray(x)
    x.show()