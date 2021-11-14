import os
from os.path import exists

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR

from BinaryToGaussianNN import BinaryToGaussianNN
from utils import generate_binary_matrix, train_model

from dotenv import load_dotenv
load_dotenv()
DATA_DIR = os.environ.get("DATA_DIR")
MODELS_DIR = os.environ.get("MODELS_DIR")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

criterion = nn.KLDivLoss(reduction='batchmean')


def train_binary_to_gaussian_with_matrix(binary_matrix, train_full, valid_dataloader, N_combinations=10):
    binary_train_features = torch.tensor(
        np.matmul(train_full, binary_matrix)).float()

    binary_train_labels = torch.tensor(
        np.load(DATA_DIR + 'gaussian_train.npy')).float()

    binary_train = []

    for i in range(binary_train_features.shape[0]):
        binary_train.append(
            [binary_train_features[i, :], binary_train_labels[i, :]])

    train_dataloader = DataLoader(binary_train,
                                  batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

    model = BinaryToGaussianNN(N_combinations=N_combinations).to(device)
    model = train_model(model, train_dataloader, device,
                        valid_dataloader=valid_dataloader, N_features=N_combinations, labelled=True, criterion=criterion, epochs=50)

    return model


if __name__ == "__main__":
    train_full = pd.read_csv(
        DATA_DIR + 'scvi_train_set_gapdh.csv', header=None).to_numpy()

    # Then, we generate a random binary matrix with N genes and use it to generate training features
    N_genes = train_full.shape[1]
    N_binary = 50  # The number of genes in our binary combination
    N_combinations = 10

    binary_matrix_filepath = MODELS_DIR + 'binary_matrix.npy'

    if exists(binary_matrix_filepath):
        binary_matrix = np.load(binary_matrix_filepath)
    else:
        binary_matrix = generate_binary_matrix(
            N_genes, N_binary, N_combinations)
        np.save(binary_matrix_filepath, binary_matrix)

    valid_full = pd.read_csv(
        DATA_DIR + 'scvi_valid_set_gapdh.csv', header=None).to_numpy()

    binary_train_labels = torch.tensor(
        np.load(DATA_DIR + 'gaussian_valid.npy')).float()

    binary_valid_features = torch.tensor(
        np.matmul(valid_full, binary_matrix)).float()

    binary_valid = []

    for i in range(binary_valid_features.shape[0]):
        binary_valid.append([binary_valid_features[i], binary_train_labels[i]])

    valid_dataloader = DataLoader(binary_valid,
                                  batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

    model = train_binary_to_gaussian_with_matrix(
        binary_matrix, train_full, valid_dataloader)
    torch.save(model, MODELS_DIR + 'binaryToGaussian.pt')
    # model = torch.load(MODELS_DIR + 'binaryToGaussian.pt')
