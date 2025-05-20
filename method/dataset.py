import numpy as np
from torch.utils.data import Dataset
from sklearn.neighbors import KDTree
from .utils import standardize_ts, hankel_matrix
import torch

class SSRDataset(Dataset):

    def __init__(self, source, target, window_size, horizon):

        super(SSRDataset, self).__init__()
        self.feature = hankel_matrix(source, window_size)
        self.label = hankel_matrix(target[horizon:], window_size)
        self.window_size = window_size
        self.horizon = horizon
        self.sample_len = min(self.feature.shape[0], self.label.shape[0])
    def __getitem__(self, idx):

        feature = self.feature[idx].astype(np.float32)
        y = self.label[idx].astype(np.float32)

        return torch.from_numpy(feature), torch.from_numpy(y)

    def __len__(self):
        return self.sample_len


class NASSRDataset(Dataset):

    def __init__(self, source, target, window_size, forward_horizon, output_len):

        super(NASSRDataset, self).__init__()
        self.feature = hankel_matrix(source, window_size)
        self.label = hankel_matrix(target[forward_horizon:], output_len)
        self.window_size = window_size
        self.horizon = forward_horizon
        self.output_len = output_len
        self.sample_len = min(self.feature.shape[0], self.label.shape[0])
    def __getitem__(self, idx):

        feature = self.feature[idx].astype(np.float32)
        y = self.label[idx].astype(np.float32)

        return torch.from_numpy(feature), torch.from_numpy(y)

    def __len__(self):
        return self.sample_len


