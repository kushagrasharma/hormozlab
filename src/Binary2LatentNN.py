import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class Binary2LatentNN(nn.Module):
    def __init__(self, N_combinations=10, N_output=100):
        super().__init__()
        self.batchNorm1 = nn.BatchNorm1d(128)

        self.hl1 = nn.Linear(N_combinations, 48)
        self.hl2 = nn.Linear(48, 128)
        self.hl3 = nn.Linear(128, 512)
        self.hl4 = nn.Linear(512, 1024)
        self.hl5 = nn.Linear(1024, 2048)
        self.output_layer = nn.Linear(2048, N_output)

    def forward(self, features):
        activation = self.hl1(features)
        activation = torch.relu(activation)


        activation = self.hl2(activation)
        activation = torch.relu(activation)

        activation = self.batchNorm1(activation)

        activation = self.hl3(activation)
        activation = torch.relu(activation)

        activation = self.hl4(activation)
        activation = torch.relu(activation)

        activation = self.hl5(activation)
        activation = torch.relu(activation)

        code = self.output_layer(activation)
        code = torch.tanh(code)
        return code