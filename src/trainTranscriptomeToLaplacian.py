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
from utils import train_model

from dotenv import load_dotenv
load_dotenv()
DATA_DIR = os.environ.get("DATA_DIR")
MODELS_DIR = os.environ.get("MODELS_DIR")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    train_full = pd.read_csv(
        DATA_DIR + 'scvi_train_set_gapdh.csv', header=None).to_numpy()
    valid_full = pd.read_csv(
        DATA_DIR + 'scvi_valid_set_gapdh.csv', header=None).to_numpy()

    v_l = np.load(DATA_DIR + 'laplacian_eigenvectors.npy')

    n_coeffs = 50

    train_laplacian_projection = torch.tensor(
        np.load(DATA_DIR + 'train_laplacian_projection.npy')).float()
    valid_laplacian_projection = torch.tensor(
        np.load(DATA_DIR + 'valid_laplacian_projection.npy')).float()

    encoder = Encoder(N_latent=n_coeffs).to(device)
    bottleneckedEncoder = BottleneckedEncoder(N_latent=n_coeffs).to(device)

    train = []
    valid = []

    for i in range(train_full.shape[0]):
        train.append([torch.tensor(train_full[i, :]).float(),
                     train_laplacian_projection[i, :n_coeffs]])

    for i in range(valid_full.shape[0]):
        valid.append([torch.tensor(valid_full[i, :]).float(),
                     valid_laplacian_projection[i, :n_coeffs]])

    train_dataloader = DataLoader(train,
                                  batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

    valid_dataloader = DataLoader(train,
                                  batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

    encoder = train_model(encoder, train_dataloader, device,
                          valid_dataloader=valid_dataloader, N_features=9781)
    bottleneckedEncoder = train_model(
        bottleneckedEncoder, train_dataloader, device, valid_dataloader=valid_dataloader, N_features=9781)

    torch.save(encoder, MODELS_DIR + 'tomeToLaplacian.pt')
    torch.save(bottleneckedEncoder, MODELS_DIR +
               'tomeToLaplacianBottlenecked.pt')
