import pickle
import pandas as pd
import os
import numpy as np


class AdapterSAC:
    def __init__(self, env_name='Acrobot'):
        current_directory = os.path.dirname(__file__)

        if env_name == 'Acrobot':
            self.data = pd.read_pickle(os.path.join(current_directory,
                                                    'data_from_sac_trained_for_975000_steps.pickle'), 'infer')
            self.observations = np.reshape(self.data['observations'],
                                           (self.data['observations'].shape[0], self.data['observations'].shape[2]))
            self.next_observations = np.reshape(self.data['next_observations'],
                                                (self.data['next_observations'].shape[0],
                                                 self.data['next_observations'].shape[2]))

            self.rewards = self.data['rewards']
            self.terminals = self.data['terminals']
            self.actions = np.reshape(self.data['actions'],
                                      (self.data['actions'].shape[0], self.data['actions'].shape[2]))

        else:
            print('unknown env name')


if __name__ == '__main__':
    a = AdapterSAC()
    print(a.observations.shape)
    print(a.next_observations.shape)
    print(a.actions.shape)
    print(a.rewards.shape)
    print(a.terminals.shape)
