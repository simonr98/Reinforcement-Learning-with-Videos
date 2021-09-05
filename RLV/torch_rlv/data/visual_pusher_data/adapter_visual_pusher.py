import pickle
import pandas as pd
import os
import torch as T
import numpy as np
from PIL import Image


class AdapterVisualImgData:
    def __init__(self, paired_data=False):
        current_directory = os.path.dirname(__file__)

        self.paired_data = paired_data

        if paired_data:
            path = os.path.join(current_directory, 'pusher_paired_3000_SAC_steps_2000_samples.pickle')
        else:
            path = os.path.join(current_directory, 'pusher_3000_SAC_steps_2000_samples.pickle')

        self.data = pickle.load(open(path, 'rb'))

        self.n = len(self.data['observation'])
        self.observation_img = T.from_numpy(np.reshape(np.array(self.data['observation_img']), (self.n, 3, 120, 120))).float()
        self.observation_img_raw = T.from_numpy(np.reshape(np.array(self.data['observation_img_raw']), (self.n, 3, 120, 120))).float()

        if not paired_data:
            self.observation = T.from_numpy(np.reshape(np.array(self.data['observation']), (self.n, 32)))
            self.action = T.from_numpy(np.reshape(np.array(self.data['action']), (self.n, 4)))
            self.next_observation = T.from_numpy(np.reshape(np.array(self.data['next_observation']), (self.n, 32)))
            self.reward = T.from_numpy(np.array(self.data["reward"]))
            self.done = T.from_numpy(np.reshape(np.array(self.data["done"]), (self.n, 1)))

if __name__ == '__main__':
    a = AdapterVisualImgData()

    print(a.observation_img.shape)
    print(a.observation_img_raw.shape)

    if not a.paired_data:
        print(a.observation.shape)
        print(a.action.shape)
        print(a.next_observation.shape)
        print(a.reward.shape)
        print(a.done.shape)


    # x = np.reshape((a.observation_img[0]).detach().numpy(), (120, 120, 3))
    # x = Image.fromarray(x)
    # x.show()