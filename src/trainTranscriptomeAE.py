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

from AutoEncoder import AE

from utils import train_model

from dotenv import load_dotenv
load_dotenv()
import os
DATA_DIR = os.environ.get("DATA_DIR")
MODELS_DIR = os.environ.get("MODELS_DIR")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    train_full = pd.read_csv(
        DATA_DIR + 'scvi_train_set_gapdh.csv', header=None).to_numpy()

    train_tensor = torch.tensor(train_full).float()

    train_dataloader = DataLoader(train_tensor,
                                  batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

    model = AE().to(device)
    model = train_model(model, train_dataloader, device, train_full.shape[1], labelled=False)
    torch.save(model, 'models/autoencoder.pt')

    valid_full = pd.read_csv(DATA_DIR + 'scvi_valid_set_gapdh.csv', header=None).to_numpy()
    valid_tensor = torch.tensor(valid_full).float()
    valid_dataloader = DataLoader(valid_tensor.float(), batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

    loss = 0
    criterion = nn.MSELoss()
    for _, batch_features in enumerate(valid_dataloader):
        batch_features = batch_features.view(-1, 9781).to(device)
        outputs = model(batch_features)
        train_loss = criterion(outputs, batch_features)
        loss += train_loss.item()

    loss /= len(valid_dataloader)
    print("validation loss = {:.6f}".format(loss))