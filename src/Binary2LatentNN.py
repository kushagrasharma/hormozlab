import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class Binary2LatentNN(nn.Module):
    def __init__(self, N_combinations=10):
        super().__init__()
        self.hl1 = nn.Linear(N_combinations, 128)
        self.output_layer = nn.Linear(128, 100)

    def forward(self, features):
        activation = self.hl1(features)
        activation = torch.relu(activation)
        code = self.output_layer(activation)
        code = torch.sigmoid(code)
        return code