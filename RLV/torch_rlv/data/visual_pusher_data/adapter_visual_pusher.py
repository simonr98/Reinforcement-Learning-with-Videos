import pickle
import pandas as pd
import os
import torch as T
import numpy as np
from PIL import Image
import cv2

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class AdapterVisualPusher:
    def __init__(self):
        current_directory = os.path.dirname(__file__)
        path = os.path.join(current_directory, '500000_SAC_steps_20000_samples.pickle')

        # get data
        data = pickle.load(open(path, 'rb'))
        observation = data['observation']
        observation_img = data['observation_img_raw']
        observation_img_raw = data['observation_img']
        action = data['action']
        next_observation = data['next_observation']
        reward = data['reward']
        done = data['done']

        # convert to numpy array
        observation = np.array(observation)
        observation_img = np.array(observation_img)[:, 0:80,20:100]
        observation_img_raw = np.array(observation_img_raw)[:, 0:80,20:100]
        action = np.array(action)
        next_observation = np.array(next_observation)
        reward = np.array(reward)
        done = np.array(done)

        # store length of data
        self.n = len(observation)

        # numpy reshape operations
        observation = np.reshape(observation, (self.n, observation.shape[1]))
        next_observation = np.reshape(next_observation, (self.n, next_observation.shape[1]))
        action = np.reshape(action, (self.n, action.shape[1]))
        done = np.reshape(done, (self.n, 1))
        observation_img = np.reshape(observation_img, (self.n, 3, 80, 80))
        observation_img_raw = np.reshape(observation_img_raw, (self.n, 3, 80, 80))

        # store tensors
        self.observation = T.from_numpy(observation)
        self.next_observation = T.from_numpy(next_observation)
        self.action = T.from_numpy(action)
        self.reward = T.from_numpy(reward)
        self.done = T.from_numpy(done)

        self.observation_img_raw = observation_img_raw
        self.observation_img = observation_img

        self.observation_img = T.from_numpy(observation_img)
        self.observation_img_raw = T.from_numpy(observation_img_raw)

if __name__ == '__main__':
    a = AdapterVisualPusher()
    img = a.observation_img[0]

    img = np.reshape(img, (80, 80, 3))
    imgplot = plt.imshow(img)
    plt.show()

    img = a.observation_img_raw[0]#[:,0:80,20:100]
    #img = np.reshape(img, (80, 80, 3))

    # print(T.sum(a.reward[:500], axis=0))
    # print(T.sum(a.reward[:5000], axis=0))
    # print(a.observation_img_raw[:100])
    # print(a.observation.shape)
    # print(a.action.shape)
    # print(a.next_observation.shape)
    # print(a.reward.shape)
    # print(a.done.shape)