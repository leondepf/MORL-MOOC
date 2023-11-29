import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


## KDD2015 Dataset
# self.x_train, self.x_test, self.y_train, self.y_test = np.load("/mnt/data/pan_feng/EarlyClassificationTimeSeries-MOOC/KDD2015/resampled_X_train_all_timesteps.npy"),\
# np.load("/mnt/data/pan_feng/EarlyClassificationTimeSeries-MOOC/KDD2015/X_test_all_timesteps.npy"),\
# np.load("/mnt/data/pan_feng/EarlyClassificationTimeSeries-MOOC/KDD2015/resampled_C_train_all_timesteps.npy"),\
# np.load("/mnt/data/pan_feng/EarlyClassificationTimeSeries-MOOC/KDD2015/C_test_all_timesteps.npy")

## XuetaingX Dataset
x_train, x_test, y_train, y_test = np.load("/mnt/data/pan_feng/EarlyClassificationTimeSeries-MOOC/XuetangX/resampled_X_train_all_timesteps.npy"),\
np.load("/mnt/data/pan_feng/EarlyClassificationTimeSeries-MOOC/XuetangX/X_test_all_timesteps.npy"),\
np.load("/mnt/data/pan_feng/EarlyClassificationTimeSeries-MOOC/XuetangX/resampled_C_train_all_timesteps.npy"),\
np.load("/mnt/data/pan_feng/EarlyClassificationTimeSeries-MOOC/XuetangX/C_test_all_timesteps.npy")

