import os

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from Transcriptome2GaussianNN import Transcriptome2GaussianNN
from utils import train_model

from dotenv import load_dotenv
load_dotenv()
DATA_DIR = os.environ.get("DATA_DIR")
MODELS_DIR = os.environ.get("MODELS_DIR")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.KLDivLoss(reduction='batchmean')


def train_tome_to_gaussian(train_full, valid_full):
    train_tensor = torch.tensor(train_full).float()
    valid_tensor = torch.tensor(valid_full).float()

    train_labels = torch.tensor(
        np.load(DATA_DIR + 'gaussian_train.npy')).float()
    valid_labels = torch.tensor(
        np.load(DATA_DIR + 'gaussian_valid.npy')).float()

    train = []
    valid = []

    for i in range(len(train_tensor)):
        train.append([train_tensor[i], train_labels[i]])

    for i in range(len(valid_tensor)):
        valid.append([valid_tensor[i], valid_labels[i]])

    train_dataloader = DataLoader(train,
                                  batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

    valid_dataloader = DataLoader(valid,
                                  batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

    model = Transcriptome2GaussianNN().to(device)
    model = train_model(model, train_dataloader, device, valid_dataloader=valid_dataloader,
                        labelled=True, criterion=criterion)

    return model


if __name__ == "__main__":
    train_full = pd.read_csv(
        DATA_DIR + 'scvi_train_set_gapdh.csv', header=None).to_numpy()

    valid_full = pd.read_csv(
        DATA_DIR + 'scvi_valid_set_gapdh.csv', header=None).to_numpy()

    model = train_tome_to_gaussian(train_full, valid_full)
    torch.save(model, MODELS_DIR + 'tomeToGaussian.pt')
