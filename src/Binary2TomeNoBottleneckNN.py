import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class Binary2TomeNoBottleneckNN(nn.Module):
    def __init__(self, N_combinations=10):
        super().__init__()
        self.hl1 = nn.Linear(N_combinations, 128)
        self.hl2 = nn.Linear(128, 1024)
        self.hl3 = nn.Linear(1024, 2048)
        self.output_layer = nn.Linear(2048, 9781)

    def forward(self, features):
        activation = self.hl1(features)
        activation = torch.relu(activation)
        activation = self.hl2(activation)
        activation = torch.relu(activation)
        activation = self.hl3(activation)
        activation = torch.relu(activation)
        code = self.output_layer(activation)
        code = torch.sigmoid(code)
        return code