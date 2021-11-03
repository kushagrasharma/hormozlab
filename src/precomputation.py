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

    valid_full = pd.read_csv(
        DATA_DIR + 'scvi_valid_set_gapdh.csv', header=None).to_numpy()

    gaussian_filepath = DATA_DIR + 'precomputed_gaussian.npy'
    if os.path.exists(gaussian_filepath):
        precomputed_gaussian = np.load(gaussian_filepath)
    else:
        precomputed_gaussian = np.apply_along_axis(
            lambda x: gaussian_centered_on_vertex(train_full, x, sigma=5), 1, train_full)

        np.save(gaussian_filepath, precomputed_gaussian)

    closest_cell_path = DATA_DIR + 'closest_cell_to_valid.npy'
    if os.path.exists(closest_cell_path):
        closest_cell_to_valid = np.load(closest_cell_path)
    else:
        closest_cell_to_valid = np.ones((len(valid_full), 2)) * -1

        for i in range(len(valid_full)):
            for j in range(len(train_full)):
                d = np.linalg.norm(valid_full[i, :] - train_full[j, :])
                if d < closest_cell_to_valid[i, 1] or closest_cell_to_valid[i, 0] < 0:
                    closest_cell_to_valid[i, 0] = j
                    closest_cell_to_valid[i, 1] = d

        np.save(closest_cell_path, closest_cell_to_valid[:, 0].astype(int))
