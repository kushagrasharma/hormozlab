import torch
import torch.nn as nn

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class Encoder(nn.Module):
    def __init__(self, N_features=9781, N_latent=100):
        super().__init__()
        self.encoder_hidden_layer1 = nn.Linear(N_features, 128)
        self.encoder_output_layer = nn.Linear(128, N_latent)

    def forward(self, features):
        activation = self.encoder_hidden_layer1(features)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        code = torch.sigmoid(code)
        return code


class Decoder(nn.Module):
    def __init__(self, N_features=9781, N_latent=100):
        super().__init__()
        self.decoder_hidden_layer1 = nn.Linear(N_latent, 128)
        self.decoder_output_layer = nn.Linear(128, N_features)

    def forward(self, features):
        activation = self.decoder_hidden_layer1(features)
        activation = torch.relu(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.sigmoid(activation)
        return reconstructed


class AE(nn.Module):
    def __init__(self, N_features=9781, N_latent=100):
        super().__init__()
        self.encoder = Encoder(N_features=N_features, N_latent=N_latent)
        self.decoder = Decoder(N_features, N_latent=N_latent)

    def forward(self, features):
        code = self.encoder(features)
        reconstructed = self.decoder(code)
        return reconstructed
