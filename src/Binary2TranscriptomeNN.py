import torch
import torch.nn as nn

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class Binary2TranscriptomeNN(nn.Module):
    def __init__(self, binaryToLatent, decoder):
        super().__init__()
        self.binaryToLatent = binaryToLatent
        self.decoder = decoder

    def forward(self, features):
        code = self.binaryToLatent(features)
        reconstructed = self.decoder(code)
        return reconstructed