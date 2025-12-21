import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class HandGestureDataset(Dataset):
    def __init__(self, data_path, transform=None, normalize=True):
        data = np.load(data_path)
        self.X = data['X']
        self.y = data['y']
        self.transform = transform
        self.normalize = normalize
        
        if self.normalize:
            # 对每个特征维度进行标准化
            # X shape: (samples, seq_len, features)
            original_shape = self.X.shape
            X_reshaped = self.X.reshape(-1, original_shape[-1])  # (samples*seq_len, features)
            self.scaler = StandardScaler()
            X_normalized = self.scaler.fit_transform(X_reshaped)
            self.X = X_normalized.reshape(original_shape)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        sample = self.X[idx]
        label = self.y[idx]
        if self.transform:
            sample = self.transform(sample)
        return torch.tensor(sample, dtype=torch.float32), torch.tensor(label, dtype=torch.long)