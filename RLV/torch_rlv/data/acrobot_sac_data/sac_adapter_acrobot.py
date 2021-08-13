import pickle
import pandas as pd
import os
import numpy as np


class AdapterSAC:
    def __init__(self):
        current_directory = os.path.dirname(__file__)
        path = os.path.join(current_directory, 'data_from_sac_trained_for_975000_steps.pickle')
        self.data = pd.read_pickle(path, compression='infer')
        self.observations = np.reshape(self.data['observations'],
                                       (self.data['observations'].shape[0],
                                        self.data['observations'].shape[2]))[:975000]

        self.next_observations = np.reshape(self.data['next_observations'],
                                            (self.data['next_observations'].shape[0],
                                             self.data['next_observations'].shape[2]))[:975000]

        self.rewards = self.data['rewards'][:975000]
        self.terminals = self.data['terminals'][:975000]
        self.actions = np.reshape(self.data['actions'],
                                  (self.data['actions'].shape[0],
                                   self.data['actions'].shape[2]))[:975000]

if __name__ == '__main__':
    a = AdapterSAC()
    print(a.observations.shape)
    print(a.next_observations.shape)
    print(a.actions.shape)
    print(a.rewards.shape)
    print(a.terminals.shape)
