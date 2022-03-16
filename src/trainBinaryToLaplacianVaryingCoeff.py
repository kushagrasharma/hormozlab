import os
import psutil

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

from Binary2LatentNN import Binary2LatentNN
from utils import get_laplacian_coefficients, load_binary_matrix, train_model, load_constrained_binary_matrix

from dotenv import load_dotenv
load_dotenv()
DATA_DIR = os.environ.get("DATA_DIR")
MODELS_DIR = os.environ.get("MODELS_DIR")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_binary_to_laplacian(binary_train_features, binary_valid_features, train_laplacian, valid_laplacian, device, N_coeffs=100):
    train = []
    valid = []

    for i in range(binary_train_features.shape[0]):
        train.append([binary_train_features[i],
                      train_laplacian[i,:N_coeffs]])

    for i in range(binary_valid_features.shape[0]):
        valid.append([binary_valid_features[i], valid_laplacian[i,:N_coeffs]])

    train_dataloader = DataLoader(train,
                                  batch_size=64, shuffle=True, num_workers=1, pin_memory=False)

    valid_dataloader = DataLoader(train,
                                  batch_size=64, shuffle=True, num_workers=1, pin_memory=False)

    model = Binary2LatentNN(N_output=N_coeffs).to(device)

    model = train_model(model, train_dataloader, device,
                        valid_dataloader, N_features=10, epochs=50)

    return model


if __name__ == "__main__":
    train_full = pd.read_csv(
        DATA_DIR + 'scvi_train_set_gapdh.csv', header=None).to_numpy()

    valid_full = pd.read_csv(
        DATA_DIR + 'scvi_valid_set_gapdh.csv', header=None).to_numpy()

    v_l = np.load(DATA_DIR + "laplacian_eigenvectors.npy")

    binary_matrix = load_binary_matrix()
    constrained_binary_matrix = torch.tensor(load_constrained_binary_matrix()).float()

    # N_coeffs_train = [1, 10, 50, 100, 250, 500, 1000, 2000, 3000, 3200, 3595]
    # N_coeffs_train = list(range(30, 150, 10))
    N_coeffs_train = [50, 3595]

    train_laplacian = np.load(DATA_DIR + "train_laplacian_projection.npy")
    valid_laplacian = np.load(DATA_DIR + "valid_laplacian_projection.npy")

    train_laplacian = torch.tensor(train_laplacian).float()
    valid_laplacian = torch.tensor(valid_laplacian).float()

    binary_train_features = torch.tensor(
        np.matmul(train_full, binary_matrix)).float()

    binary_valid_features = torch.tensor(
        np.matmul(valid_full, binary_matrix)).float()

    constrained_binary_train_features = torch.tensor(
        np.matmul(train_full, binary_matrix)).float()

    constrained_binary_valid_features = torch.tensor(
        np.matmul(valid_full, binary_matrix)).float()

    print("DATA LOADED")

    for N_coeffs in N_coeffs_train:
        print("TRAINING MODEL {}".format(N_coeffs))
        model = train_binary_to_laplacian(
            binary_train_features, binary_valid_features, train_laplacian, valid_laplacian, device, N_coeffs=N_coeffs)

        constrained_model = train_binary_to_laplacian(
            constrained_binary_train_features, constrained_binary_valid_features, train_laplacian, valid_laplacian, device, N_coeffs=N_coeffs)

        torch.save(model, MODELS_DIR +
                   'binaryToLaplacian{}Coeffs.pt'.format(N_coeffs))

        torch.save(model, MODELS_DIR +
                   'binaryToLaplacian{}Coeffs_constrained.pt'.format(N_coeffs))
