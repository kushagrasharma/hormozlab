import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from AutoEncoder import Decoder

from utils import train_model, get_laplacian_from_tome_data, get_laplacian_eig_from_laplacian, get_laplacian_coefficients

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

    train_tensor = torch.tensor(train_full).float()

    train = []

    for i in range(train_full.shape[0]):
        train.append(
            [torch.tensor(laplacian_projection[i, :]).float(), torch.tensor(indicators[i, :]).float()])

    train_dataloader = DataLoader(train,
                                  batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

    model = Decoder().to(device)
    model = train_model(model, train_dataloader, device,
                        n_coeffs, labelled=True)

    torch.save(model, 'models/laplacianToIndicator.pt')
