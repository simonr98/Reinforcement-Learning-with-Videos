import pickle
import pandas as pd
import os
import torch as T
import numpy as np
from PIL import Image


class AdapterPairedData:
    def __init__(self):
        current_directory = os.path.dirname(__file__)

        path = os.path.join(current_directory, 'paired_3000_SAC_steps_500_samples.pickle')
        self.data = pickle.load(open(path, 'rb'))

        self.n = len(self.data['observation_img'])

        self.observation_img = T.from_numpy(np.reshape(np.array(self.data['observation_img']), (self.n, 3, 120, 120))).float()
        self.observation_img_raw = T.from_numpy(np.reshape(np.array(self.data['observation_img_raw']), (self.n, 3, 120, 120))).float()


if __name__ == '__main__':
    a = AdapterPairedData()

    print(a.observation_img_raw.shape)
    print(a.observation_img.shape)