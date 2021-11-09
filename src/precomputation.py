import os
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

from utils import gaussian_centered_on_vertex

from dotenv import load_dotenv
load_dotenv()
DATA_DIR = os.environ.get("DATA_DIR")
np.random.seed(42)


def precompute_gaussian_sigma(train_full, sigma=5):
    gaussian_filepath = DATA_DIR + 'precomputed_gaussian_sigma_{}.npy'.format(sigma)
    if not os.path.exists(gaussian_filepath):
        precomputed_gaussian = np.apply_along_axis(
            lambda x: gaussian_centered_on_vertex(train_full, x, sigma=5), 1, train_full)

        np.save(gaussian_filepath, precomputed_gaussian)


def precompute_closest_cell_to_validation_set(train_full, valid_full):
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


def precompute_gaussian_sigma_kthNN(train_full, k=10):
    gaussian_filepath = DATA_DIR + \
        'precomputed_gaussian_sigma_{}thNN.npy'.format(k)

    truncated_gaussian_filepath = DATA_DIR + \
        'precomputed_gaussian_sigma_{}thNN_truncated.npy'.format(k)
    if not os.path.exists(gaussian_filepath):
        neigh = NearestNeighbors(n_neighbors=k, n_jobs=-1)
        neigh.fit(train_full)
        distances = neigh.kneighbors(train_full, return_distance=True)[0]
        kth_distance = distances[:, -1]

        gaussian = np.array([gaussian_centered_on_vertex(train_full, v, sigma=kth_distance[i])
                            for i, v in enumerate(train_full)])

        np.save(gaussian_filepath, gaussian)

        truncated_idxs = np.argsort(-gaussian, axis=1)[:,11:]
        truncated_gaussian = np.copy(gaussian)

        for i in range(len(gaussian)):
            truncated_gaussian[i,:][truncated_idxs[i,:]] = 0 
            truncated_gaussian[i,:] /= truncated_gaussian[i,:].sum()

        np.save(truncated_gaussian_filepath, truncated_gaussian)


def truncate_gaussian(k=10):
    gaussian_filepath = DATA_DIR + \
        'precomputed_gaussian_sigma_{}thNN.npy'.format(k)
    gaussian = np.load(gaussian_filepath)


if __name__ == "__main__":
    train_full = pd.read_csv(
        DATA_DIR + 'scvi_train_set_gapdh.csv', header=None).to_numpy()

    valid_full = pd.read_csv(
        DATA_DIR + 'scvi_valid_set_gapdh.csv', header=None).to_numpy()

    precompute_gaussian_sigma(train_full)
    precompute_gaussian_sigma(train_full, sigma=100)
    precompute_closest_cell_to_validation_set(train_full, valid_full)
    precompute_gaussian_sigma_kthNN(train_full)
    precompute_gaussian_sigma_kthNN(train_full, 5)
