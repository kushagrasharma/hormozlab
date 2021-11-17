import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from Laplacian2TranscriptomeNN import Laplacian2TranscriptomeNN

from utils import train_model, get_laplacian_from_tome_data, get_laplacian_eig_from_laplacian, get_laplacian_coefficients

from dotenv import load_dotenv
load_dotenv()
DATA_DIR = os.environ.get("DATA_DIR")
MODELS_DIR = os.environ.get("MODELS_DIR")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    train_full = pd.read_csv(
        DATA_DIR + 'scvi_train_set_gapdh.csv', header=None).to_numpy()

    v_l = np.load(DATA_DIR + 'laplacian_eigenvectors.npy')
    gaussian_train = np.load(DATA_DIR + 'gaussian_train.npy')

    n_coeffs = 3000

    laplacian_projection = np.load(DATA_DIR + 'train_laplacian_projection.npy')

    laplacian_projection = torch.tensor(laplacian_projection).float()

    train_tensor = torch.tensor(train_full).float()

    train = []

    for i in range(train_full.shape[0]):
        train.append([laplacian_projection[i, :], train_tensor[i, :]])

    train_dataloader = DataLoader(train,
                                  batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

    model = Laplacian2TranscriptomeNN(n_coeffs).to(device)
    model = train_model(model, train_dataloader, device,
                        N_features=n_coeffs, labelled=True)

    torch.save(model, 'models/laplacianToTome{}Coeffs.pt'.format(n_coeffs))
