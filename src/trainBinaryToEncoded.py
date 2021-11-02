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

from Binary2LatentNN import Binary2LatentNN
from utils import generate_binary_matrix, train_model

from dotenv import load_dotenv
load_dotenv()
import os
DATA_DIR = os.environ.get("DATA_DIR")
MODELS_DIR = os.environ.get("MODELS_DIR")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_binary_to_latent_with_matrix(binary_matrix, train_full, N_combinations=10):
    encoder = torch.load(MODELS_DIR + 'autoencoder.pt').encoder   
    label_function = lambda data: encoder(data).detach()

    train_tensor = torch.tensor(train_full).float()

    binary_train_features = torch.tensor(np.matmul(train_full, binary_matrix)).float()
    binary_train_labels = label_function(train_tensor)

    binary_train = []

    for i in range(binary_train_features.shape[0]):
        binary_train.append([binary_train_features[i], binary_train_labels[i]])

    train_dataloader = DataLoader(binary_train, 
                                batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

    model = Binary2LatentNN(N_combinations=N_combinations).to(device)
    model = train_model(model, train_dataloader, device, N_combinations, labelled=True)

    return model

if __name__ == "__main__":
    train_full = pd.read_csv(
        DATA_DIR + 'scvi_train_set_gapdh.csv', header=None).to_numpy()

    # Then, we generate a random binary matrix with N genes and use it to generate training features
    N_genes = train_full.shape[1]
    N_binary = 50 # The number of genes in our binary combination
    N_combinations = 10

    binary_matrix_filepath = MODELS_DIR + 'binary_matrix.npy'

    if exists(binary_matrix_filepath):
        binary_matrix = np.load(binary_matrix_filepath)
    else:
        binary_matrix = generate_binary_matrix(N_genes, N_binary, N_combinations)
        np.save(binary_matrix_filepath, binary_matrix)

    model = train_binary_to_latent_with_matrix(binary_matrix, train_full)
    torch.save(model, MODELS_DIR + 'binaryToEncoded.pt')

    encoder = torch.load(MODELS_DIR + 'autoencoder.pt').encoder   
    label_function = lambda data: encoder(data).detach()
    criterion = nn.MSELoss()

    ## Generate validation data
    valid_full = pd.read_csv(DATA_DIR + 'scvi_valid_set_gapdh.csv', header=None).to_numpy()
    valid_tensor = torch.tensor(valid_full).float()
    
    binary_valid_features = torch.tensor(np.matmul(valid_full, binary_matrix)).float()
    binary_valid_labels = label_function(valid_tensor)

    binary_valid = []

    for i in range(binary_valid_features.shape[0]):
        binary_valid.append([binary_valid_features[i], binary_valid_labels[i]])

    valid_dataloader = DataLoader(binary_valid, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

    loss = 0
    for batch_features, batch_labels in valid_dataloader:
        batch_features = batch_features.view(-1, 10).to(device)
        outputs = model(batch_features)
        loss += criterion(outputs, batch_labels)

    loss /= len(valid_dataloader)
    print("validation loss = {:.6f}".format(loss))