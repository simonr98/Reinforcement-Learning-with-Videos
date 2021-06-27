import pickle
import pandas as pd
import os

current_directory = os.path.dirname(__file__)

data = pd.read_pickle(os.path.join(current_directory, 'hand_july_21_26_keep_all_fixed.pkl'), 'gzip')

print(data.keys())
