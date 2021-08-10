import pickle
import pandas as pd
import os
import numpy as np


class AcrobotAdapter:
    def __init__(self, data_type='unpaired', env_name='Acrobot'):
        current_directory = os.path.dirname(__file__)

        if env_name == 'Acrobot':
            self.data = pd.read_pickle(os.path.join(current_directory, 'acrobot-975-1000.pkl'), 'gzip')
            self.observations = self.data['observations']
            self.next_observations = self.data['next_observations']
            self.rewards = self.data['rewards']
            self.terminals = self.data['terminals']
            self.actions = self.data['actions']



if __name__ == '__main__':
    a = AcrobotAdapter()
    print(a.observations.shape)
    print(a.next_observations.shape)
    print(a.actions.shape)
    print(a.rewards.shape)
    print(a.terminals.shape)
