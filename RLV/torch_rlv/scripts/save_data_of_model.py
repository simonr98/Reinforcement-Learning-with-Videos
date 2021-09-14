import numpy as np
import cv2
import time
import pickle
from RLV.torch_rlv.environments.utils import get_environment
from RLV.torch_rlv.algorithms.sac.sac import SAC
from fastgrab import screenshot

class DatasetCreator():
    def __init__(self, env_name, num_steps=40000, model_path="../output/sac_models/acrobot/trained_for_1000000.zip"):
        self.env = get_environment(env_name)
        self.env_name = env_name
        self.num_steps = num_steps
        self.model = SAC.load(model_path)
        print(self.model.wandb_config)
        self.total_steps = self.model.total_steps

        if self.env_name == 'visual_pusher':
            self.dataset = {'observation': [], 'observation_img': [], 'observation_img_raw': [], 'action': [],
                            'next_observation': [], 'reward': [],  'done': []}
            self.paired_dataset = {'observation_img_raw': [], 'observation_img': []}
        else:
            self.dataset = {'observation': [], 'action': [], 'next_observation': [], 'reward': [], 'done': []}

    def get_image(self, mode="rgb_array", noise=None):
        img_big = cv2.cvtColor(screenshot.Screenshot().capture(), cv2.COLOR_BGRA2RGB)
        img = cv2.resize(img_big, (120,120), interpolation = cv2.INTER_AREA)

        if noise == "gauss":
            img = cv2.GaussianBlur(img, (5, 5), 0)
        if noise == "red":
            red_img = np.full(img.shape, (0, 0, 255), np.uint8)
            img = cv2.addWeighted(img, 0.8, red_img, 0.2, 0)

        if mode == "rgb_array":
            return img


    def save_data_of_model(self):
        global filter
        filter = 'red'

        # random steps to adjust camera
        self.env.reset()
        for j in range(300):
            action = self.env.action_space.sample()
            _, _, done, _ = self.env.step(action)
            if done:
                self.env.reset()

        obs = self.env.reset()

        counter = 0

        for i in range(self.num_steps):
            if self.env_name == 'visual_pusher':
                # get the images with and without noise
                obs_img_raw = self.get_image()
                obs_img = self.get_image(noise=filter)

            action, state_ = self.model.predict(obs)
            next_obs, reward, done, _ = self.env.step(action)
            counter += 1

            # build up the data set
            self.dataset['observation'].append(obs)
            self.dataset['action'].append(action)
            self.dataset['next_observation'].append(next_obs)
            self.dataset['reward'].append(reward)
            self.dataset['done'].append(done)

            # if visual pusher append image data
            if self.env_name == 'visual_pusher':
                self.dataset['observation_img'].append(obs_img)
                self.dataset['observation_img_raw'].append(obs_img_raw)

                # in every 10. step append paired image data
                if i % 10 == 0:
                    self.paired_dataset['observation_img_raw'].append(obs_img_raw)
                    self.paired_dataset['observation_img'].append(obs_img)

            if not done:
                obs = next_obs

            # if not succesfull drop observations
            elif self.env_name == 'visual_pusher' and counter == 300:
                obs = self.env.reset()
                n = len(self.dataset['observation'])
                self.dataset['observation'] = self.dataset['observation'][:n-counter]
                self.dataset['action'] = self.dataset['action'][:n-counter]
                self.dataset['next_observation'] = self.dataset['next_observation'][:n-counter]
                self.dataset['reward'] = dataset['reward'][:n-counter]
                self.dataset['done'] = self.dataset['done'][:n-counter]
                # self.dataset['observation_img'] = self.dataset['observation_img'][:n-counter]
                # self.dataset['observation_img_raw'] = self.dataset['observation_img_raw'][:n-counter]
                if i % 10 == 0:
                    self.paired_dataset['observation_img_raw'] = self.paired_dataset['observation_img_raw'][:n-counter]
                    self.paired_dataset['observation_img'] = self.paired_dataset['observation_img'][:n-counter]
                counter = 0

            else:
                obs = self.env.reset()
                counter = 0
                filter = 'gauss' if filter == 'red' else 'gauss'

            if i % 500 == 0:
                print(f'{(i / self.num_steps * 100)}  % done')

        with open(f'../data/{self.env_name}_data/{self.total_steps}_SAC_steps'
                  f'_{self.num_steps}_samples.pickle', 'w+b') as df:
            pickle.dump(self.dataset, df)

        if self.env_name == 'visual_pusher':
            with open(f'../data/{self.env_name}_data/paired_{self.total_steps}_SAC_steps'
                      f'_{self.num_steps}_samples.pickle', 'w+b') as df:
                pickle.dump(self.paired_dataset, df)


if __name__ == '__main__':
    creator = DatasetCreator(env_name='visual_pusher', num_steps=10000,
                             model_path="../data/visual_pusher_data/478666_sac_trained_for_500000_steps")

    creator.save_data_of_model()

