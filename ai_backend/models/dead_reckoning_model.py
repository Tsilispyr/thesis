import os
import sys

import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_processing.dataset_parser import INPUT_SIZE


class DeadReckoningLSTM(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, hidden_size=64,
                 num_layers=2, output_size=3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, output_size),
        )

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])


if __name__ == '__main__':
    model = DeadReckoningLSTM()
    dummy = torch.randn(16, 10, INPUT_SIZE)
    out   = model(dummy)
    print(f"Output shape: {out.shape}  expected (16, 3)")
