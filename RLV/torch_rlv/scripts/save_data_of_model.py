import numpy as np
import pickle
from RLV.torch_rlv.environments.utils import get_environment
from RLV.torch_rlv.algorithms.sac.sac import SAC

class DatasetCreator():
    def __init__(self, env_name, num_steps=20000, model_path="../output/sac_models/trained_for_3000"):
        self.env = get_environment(env_name)
        self.env_name = env_name
        self.num_steps = num_steps
        self.model = SAC.load(model_path)
        self.total_steps = self.model.total_steps

        if self.env_name == 'visual_pusher':
            self.dataset = {'observation': [], 'observation_img': [], 'observation_img_raw': [], 'action': [],
                            'next_observation': [], 'reward': [],  'done': []}
            self.paired_dataset = {'raw_img': [], 'filtered_img': []}
        else:
            self.dataset = {'observation': [], 'action': [], 'next_observation': [], 'reward': [], 'done': []}

    def save_data_of_model(self):
        obs = self.env.reset()
        for i in range(self.num_steps):
            action, state_ = self.model.predict(obs)

            if self.env_name == 'visual_pusher':
                obs_img = self.env.get_image()
                obs_img_raw = self.env.get_raw_image()

            next_obs, reward, done, _ = self.env.step(action)

            self.dataset['observation'].append(obs)
            self.dataset['action'].append(action)
            self.dataset['next_observation'].append(next_obs)
            self.dataset['reward'].append(reward)
            self.dataset['done'].append(done)

            if self.env_name == 'visual_pusher':
                self.dataset['observation_img'].append(obs_img)
                self.dataset['observation_img_raw'].append(obs_img_raw)

                if i % 10 == 0:
                    self.paired_dataset['raw_img'].append(obs_img_raw)
                    self.paired_dataset['filtered_img'].append(obs_img)

            if not done:
                obs = next_obs
            else:
                obs = self.env.reset()

            with open(f'../data/{self.env_name}/pusher_{self.total_steps}_SAC_steps'
                      f'_{self.num_steps}_samples.pickle', 'w+b') as df:
                pickle.dump(self.dataset, df)

            if self.env_name == 'visual_pusher':
                with open(f'../data/{self.env_name}/paired_{self.total_steps}_SAC_steps'
                          f'_{self.num_steps}_samples.pickle', 'w+b') as df:
                    pickle.dump(self.paired_dataset, df)


if __name__ == '__main__':
    creator = DatasetCreator(env_name='acrobot_continuous', num_steps=2000,
                             model_path="../output/sac_models/trained_for_3000")

    creator.save_data_of_model()

