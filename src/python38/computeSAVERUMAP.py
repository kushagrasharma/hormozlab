from sklearn.neighbors import NearestNeighbors
import umap
import pandas as pd
import numpy as np

import os
os.chdir('/Users/kushagrasharma/coding/hormozlab/src')

DATA_DIR = "/Users/kushagrasharma/coding/hormozlab/data/"

if __name__ == "__main__":
    test_full = pd.read_csv(DATA_DIR + 'scvi_test_set_gapdh.csv', header=None).to_numpy()

    test_umap = np.array(umap.UMAP().fit_transform(test_full))

    np.save(DATA_DIR + "test_saver_coords.npy", test_umap)
