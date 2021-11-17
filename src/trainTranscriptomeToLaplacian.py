import os
from os.path import exists

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from AutoEncoder import Encoder
from BottleneckedEncoder import BottleneckedEncoder
from Binary2LatentNN import Binary2LatentNN
from utils import get_laplacian_from_tome_data, get_laplacian_eig_from_laplacian, get_laplacian_coefficients, train_model, generate_binary_matrix

from dotenv import load_dotenv
load_dotenv()
DATA_DIR = os.environ.get("DATA_DIR")
MODELS_DIR = os.environ.get("MODELS_DIR")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    train_full = pd.read_csv(
        DATA_DIR + 'scvi_train_set_gapdh.csv', header=None).to_numpy()

    laplacian = get_laplacian_from_tome_data(train_full)
    lambda_l, v_l = get_laplacian_eig_from_laplacian(laplacian)

    n_coeffs = 100

    indicators = np.identity(train_full.shape[0])
    laplacian_projection = np.apply_along_axis(
        lambda x: get_laplacian_coefficients(x, v_l), 1, indicators)[:, :n_coeffs]

    encoder = Encoder().to(device)
    bottleneckedEncoder = BottleneckedEncoder().to(device)
    binaryEncoder = Binary2LatentNN().to(device)

    binary_matrix_filepath = MODELS_DIR + 'binary_matrix.npy'
    N_genes = train_full.shape[1]
    N_binary = 50  # The number of genes in our binary combination
    N_combinations = 10

    if exists(binary_matrix_filepath):
        binary_matrix = np.load(binary_matrix_filepath)
    else:
        binary_matrix = generate_binary_matrix(
            N_genes, N_binary, N_combinations)
        np.save(binary_matrix_filepath, binary_matrix)

    binary_train_features = torch.tensor(
        np.matmul(train_full, binary_matrix)).float()

    train = []
    binary_train = []

    for i in range(train_full.shape[0]):
        train.append([torch.tensor(train_full[i, :]).float(),
                     torch.tensor(laplacian_projection[i, :]).float()])
        binary_train.append([binary_train_features[i], torch.tensor(
            laplacian_projection[i, :]).float()])

    train_dataloader = DataLoader(train,
                                  batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

    binary_train_dataloader = DataLoader(binary_train,
                                         batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

    encoder = train_model(encoder, train_dataloader, device, 9781)
    bottleneckedEncoder = train_model(
        bottleneckedEncoder, train_dataloader, device, 9781)
    binaryEncoder = train_model(binaryEncoder, binary_train_dataloader, device, 10)

    torch.save(encoder, MODELS_DIR + 'tomeToLaplacian.pt')
    torch.save(bottleneckedEncoder, MODELS_DIR +
               'tomeToLaplacianBottlenecked.pt')
    torch.save(binaryEncoder, MODELS_DIR + 'binaryToLaplacian.pt')
