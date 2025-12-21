import torch
import torch.nn as nn

class GestureGRU(nn.Module):
    def __init__(self, input_dim=225, hidden_dim=256, num_layers=3, num_classes=10, dropout=0.5):
        super(GestureGRU, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.bn = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        out, _ = self.gru(x)
        out = self.bn(out[:, -1, :])  # BatchNorm on last timestep
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return out