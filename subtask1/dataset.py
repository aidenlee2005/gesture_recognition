import numpy as np
import torch
from torch.utils.data import Dataset

class HandGestureDataset(Dataset):
    def __init__(self, data_path, transform=None):
        data = np.load(data_path)
        self.X = data['X']
        self.y = data['y']
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        sample = self.X[idx]
        label = self.y[idx]
        if self.transform:
            sample = self.transform(sample)
        return torch.tensor(sample, dtype=torch.float32), torch.tensor(label, dtype=torch.long)