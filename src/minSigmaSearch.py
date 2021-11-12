import os
import pandas as pd
import numpy as np

from utils import gaussian_centered_on_vertex

from dotenv import load_dotenv
load_dotenv()
DATA_DIR = os.environ.get("DATA_DIR")
EPSILON_N_P_I = os.environ.get("EPSILON_N_P_I")
N_p_i = lambda x: (x > EPSILON_N_P_I).sum()

np.random.seed(42)

def binary_search_for_sigma(i, train_full, min_N, max_sigma):
    v_i = train_full[i,:]
    sigma = float(max_sigma) / 2.0
    gaussian = gaussian_centered_on_vertex(train_full, v_i, sigma=sigma)
    N = N_p_i(gaussian)
    

if __name__ == "__main__":
    train_full = pd.read_csv(
        DATA_DIR + 'scvi_train_set_gapdh.csv', header=None).to_numpy()

    min_sigma_filepath = DATA_DIR + 'min_sigma.npy'
    if os.path.exists(min_sigma_filepath):
        precomputed_gaussian = np.load(min_sigma_filepath)
    else:
        precomputed_min_sigma = np.apply_along_axis(
            lambda x: gaussian_centered_on_vertex(train_full, x, sigma=5), 1, train_full)

        np.save(min_sigma_filepath, precomputed_min_sigma)
