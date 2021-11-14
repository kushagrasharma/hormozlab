import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class Transcriptome2GaussianNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(0.2)
        self.batchNorm1 = nn.BatchNorm1d(128)
        self.batchNorm2 = nn.BatchNorm1d(2048)

        self.hl1 = nn.Linear(9781, 128)
        self.hl2 = nn.Linear(128, 512)
        self.hl3 = nn.Linear(512, 1024)
        self.hl4 = nn.Linear(1024, 2048)
        self.hl5 = nn.Linear(2048, 4196)
        self.output_layer = nn.Linear(4196, 3595)

    def forward(self, features):
        activation = self.hl1(features)
        activation = torch.relu(activation)
        activation = self.batchNorm1(activation)
        activation = self.hl2(activation)
        activation = torch.relu(activation)
        activation = self.hl3(activation)
        activation = torch.relu(activation)
        activation = self.dropout(activation)
        activation = self.hl4(activation)
        activation = torch.relu(activation)
        activation = self.batchNorm2(activation)
        activation = self.hl5(activation)
        activation = torch.relu(activation)
        code = self.output_layer(activation)
        code = F.log_softmax(code, dim=1)
        return code