import os
import pandas as pd
import numpy as np

from utils import gaussian_centered_on_vertex

from dotenv import load_dotenv
load_dotenv()
DATA_DIR = os.environ.get("DATA_DIR")
np.random.seed(42)

if __name__ == "__main__":
    train_full = pd.read_csv(
        DATA_DIR + 'scvi_train_set_gapdh.csv', header=None).to_numpy()
    precomputed_gaussian = np.apply_along_axis(
        lambda x: gaussian_centered_on_vertex(train_full, x, sigma=5), 1, train_full)

    precomputed_gaussian.tofile(DATA_DIR + 'precomputed_gaussian.npy')
