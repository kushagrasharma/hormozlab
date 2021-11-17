import os

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
from PCA2TranscriptomeNN import PCA2TranscriptomeNN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from utils import train_model

from dotenv import load_dotenv
load_dotenv()
DATA_DIR = os.environ.get("DATA_DIR")
MODELS_DIR = os.environ.get("MODELS_DIR")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_pca_to_gaussian(train_full, N_components=100):
    train_full = StandardScaler().fit_transform(train_full)

    ## PCA

    pca = PCA(n_components=N_components)
    pca = pca.fit_transform(train_full)
    pca = torch.tensor(pca).float()

    np.save(DATA_DIR + 'train_100_pca.npy', train_full)

    train = []

    train_tensor = torch.tensor(train_full).float()

    for i in range(train_tensor.shape[0]):
        train.append([pca[i,:], train_tensor[i]])

    train_dataloader = DataLoader(train,
                                  batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

    model = PCA2TranscriptomeNN().to(device)
    model = train_model(model, train_dataloader, device,
                        labelled=True, N_features=N_components)

    return model


if __name__ == "__main__":
    train_full = pd.read_csv(
        DATA_DIR + 'scvi_train_set_gapdh.csv', header=None).to_numpy()

    model = train_pca_to_gaussian(train_full)
    torch.save(model, MODELS_DIR + 'PCAToTome.pt')
