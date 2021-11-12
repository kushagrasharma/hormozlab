import os

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
from Transcriptome2GaussianNN import Transcriptome2GaussianNN
from utils import train_model

from dotenv import load_dotenv
load_dotenv()
DATA_DIR = os.environ.get("DATA_DIR")
MODELS_DIR = os.environ.get("MODELS_DIR")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_tome_to_gaussian(train_full):
    train_tensor = torch.tensor(train_full).float()

    binary_train_labels = torch.tensor(
        np.load(DATA_DIR + 'truncated_gaussian_sigma_10thNN.npy')).float()

    train = []

    for i in range(train_tensor.shape[0]):
        train.append([train_tensor[i], binary_train_labels[i]])

    train_dataloader = DataLoader(train,
                                  batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

    model = Transcriptome2GaussianNN().to(device)
    model = train_model(model, train_dataloader, device,
                        labelled=True)

    return model


if __name__ == "__main__":
    train_full = pd.read_csv(
        DATA_DIR + 'scvi_train_set_gapdh.csv', header=None).to_numpy()

    model = train_tome_to_gaussian(train_full)
    torch.save(model, MODELS_DIR + 'tomeToGaussian.pt')
