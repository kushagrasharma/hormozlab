import itertools

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from Binary2TranscriptomeNN import Binary2TranscriptomeNN
from trainBinaryToEncoded import train_binary_to_latent_with_matrix
from utils import generate_binary_matrix

from dotenv import load_dotenv
load_dotenv()
import os
DATA_DIR = os.environ.get("DATA_DIR")
MODELS_DIR = os.environ.get("MODELS_DIR")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    train_full = pd.read_csv(
        DATA_DIR + 'scvi_train_set_gapdh.csv', header=None).to_numpy()
    valid_full = pd.read_csv(
        DATA_DIR + 'scvi_valid_set_gapdh.csv', header=None).to_numpy()
    valid_tensor = torch.tensor(valid_full).float()

    criterion = nn.MSELoss()

    # Then, we generate a random binary matrix with N genes and use it to generate training features
    N_genes = train_full.shape[1]
    # The number of genes in our binary combination
    N_binary = [1, 2, 5, 10, 20, 40, 50, 75, 100]
    N_combinations = [1, 2, 5, 10, 20, 40, 50, 75, 100]

    losses = []

    for N_b, N_c in itertools.product(N_binary, N_combinations):
        binary_matrix = generate_binary_matrix(N_genes, N_b, N_c)

        model = train_binary_to_latent_with_matrix(
            binary_matrix, train_full, N_c)

        model = Binary2TranscriptomeNN(model, torch.load(MODELS_DIR + 'autoencoder.pt').decoder)

        binary_valid_features = torch.tensor(
            np.matmul(valid_full, binary_matrix)).float()
        binary_valid_labels = valid_tensor
        print(binary_valid_features.shape)
        print(valid_tensor.shape)

        binary_valid = []

        for i in range(binary_valid_features.shape[0]):
            binary_valid.append(
                [binary_valid_features[i], binary_valid_labels[i]])

        valid_dataloader = DataLoader(
            binary_valid, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

        loss = 0
        for batch_features, batch_labels in valid_dataloader:
            batch_features = batch_features.view(-1, N_c).to(device)
            outputs = model(batch_features)
            loss += criterion(outputs, batch_labels)

        loss /= len(valid_dataloader)
        print("validation loss for N_binary={}, N_combinations={} = {:.6f}".format(
            N_b, N_c, loss))
        losses.append([N_b, N_c, loss])

    pd.DataFrame(losses, columns=['N_binary', 'N_combinations', 'valid_loss']).to_csv(
        DATA_DIR + "binaryMatrixParamSearchResults.csv")
