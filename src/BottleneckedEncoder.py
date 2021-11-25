import torch
import torch.nn as nn

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class BottleneckedEncoder(nn.Module):
    def __init__(self, N_features=9781, N_latent=100):
        super().__init__()
        self.encoder_hidden_layer1 = nn.Linear(N_features, 32)
        self.encoder_output_layer = nn.Linear(32, N_latent)

    def forward(self, features):
        activation = self.encoder_hidden_layer1(features)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        code = torch.sigmoid(code)
        return code