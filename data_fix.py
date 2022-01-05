import os

import numpy as np

path = '/Users/christiandjurhuus/Documents/DTU/6_semester/ml_ops/dtu_mlops/data/corruptmnist/'
os.chdir(path)

file_list = ["train_0.npz", "train_1.npz", "train_2.npz", "train_3.npz", "train_4.npz", "train_5.npz", "train_6.npz", "train_7.npz"]

data_all = [np.load(fname) for fname in file_list]
merged_data = {}
for data in data_all:
    [merged_data.update({k: v}) for k, v in data.items()]
np.savez('training_data.npz', **merged_data)
