import pickle
import pandas as pd
import os
import numpy as np


class Adapter:
    def __init__(self, data_type='unpaired', env_name='Acrobot'):
        current_directory = os.path.dirname(__file__)

        if env_name == 'Acrobot':
            self.data = pd.read_pickle(os.path.join(current_directory, 'Acrobot/acrobot-975-1000.pkl'), 'gzip')
            self.observations = self.data['observations']
            self.next_observations = self.data['next_observations']
            self.rewards = self.data['rewards']
            self.terminals = self.data['terminals']
            self.actions = self.data['actions']

        if env_name == 'Visual_Pusher':
            if data_type == 'unpaired':
                self.data = pd.read_pickle(os.path.join(current_directory, 'Visual_Pusher/hand_july_21_26_keep_all_fixed.pkl'), 'gzip')
            else:
                self.data = pd.read_pickle(os.path.join(current_directory, 'Visual_Pusher/human_paired_oct_10.pkl'), 'gzip')
            self.observations = self.data['observations']
            self.next_observations = self.data['next_observations']
            self.rewards = self.data['rewards']
            self.terminals = self.data['terminals']
            self.actions = self.data['actions']


if __name__ == '__main__':
    a = Adapter()
    print(a.observations.shape)
    print(a.actions)
    print(a.terminals.shape)
