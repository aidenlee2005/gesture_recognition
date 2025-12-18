import torch
import torch.nn as nn

class GestureGRU(nn.Module):
    def __init__(self, input_dim=63, hidden_dim=128, num_layers=1, num_classes=4, dropout=0.2):
        super(GestureGRU, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        out, _ = self.gru(x)
        out = self.dropout(out[:, -1, :])  # 取最后时间步
        out = self.fc(out)
        return out