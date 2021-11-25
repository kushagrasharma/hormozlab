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

from Binary2TomeBottleneckedNN import Binary2TomeBottleneckedNN
from utils import load_binary_matrix, train_model

from dotenv import load_dotenv
load_dotenv()
DATA_DIR = os.environ.get("DATA_DIR")
MODELS_DIR = os.environ.get("MODELS_DIR")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_binary_to_tome_with_matrix(binary_matrix, train_full, valid_full, N_combinations=10):
    train_tensor = torch.tensor(train_full).float()
    valid_tensor = torch.tensor(valid_full).float()

    binary_train_features = torch.tensor(
        np.matmul(train_full, binary_matrix)).float()

    binary_valid_features = torch.tensor(
        np.matmul(valid_tensor, binary_matrix)).float()

    binary_train = []
    binary_valid = []

    for i in range(binary_train_features.shape[0]):
        binary_train.append([binary_train_features[i], train_tensor[i]])

    for i in range(binary_valid_features.shape[0]):
        binary_train.append([binary_valid_features[i], valid_tensor[i]])

    train_dataloader = DataLoader(binary_train,
                                  batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

    valid_dataloader = DataLoader(binary_train,
                                  batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

    model = Binary2TomeBottleneckedNN(N_combinations=N_combinations).to(device)
    model = train_model(model, train_dataloader, device, valid_dataloader=valid_dataloader,
                        N_features=N_combinations, labelled=True)

    return model


if __name__ == "__main__":
    train_full = pd.read_csv(
        DATA_DIR + 'scvi_train_set_gapdh.csv', header=None).to_numpy()

    valid_full = pd.read_csv(
        DATA_DIR + 'scvi_valid_set_gapdh.csv', header=None).to_numpy()

    binary_matrix = load_binary_matrix()

    model = train_binary_to_tome_with_matrix(binary_matrix, train_full, valid_full)
    torch.save(model, MODELS_DIR + 'binaryToTomeBottlenecked.pt')
