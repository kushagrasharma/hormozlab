from sklearn.neighbors import NearestNeighbors
import umap
import pandas as pd
import numpy as np

import os
os.chdir('/Users/kushagrasharma/coding/hormozlab/src')

DATA_DIR = "/Users/kushagrasharma/coding/hormozlab/data/"

if __name__ == '__main__':
    test_full = pd.read_csv(DATA_DIR + 'scvi_test_set_gapdh.csv', header=None).to_numpy()

    m1_results = np.load(DATA_DIR + 'reconstructed_tome_m1.npy')

    m5_results = np.load(DATA_DIR + 'reconstructed_tome_m5.npy')

    reducer_m1 = umap.UMAP()
    reducer_m5 = umap.UMAP()

    concat_m1 = np.concatenate((test_full, m1_results))
    concat_m5 = np.concatenate((test_full, m1_results))

    umap_m1 = reducer_m1.fit_transform(concat_m1)
    umap_m5 = reducer_m5.fit_transform(concat_m5)

    np.save(DATA_DIR + 'umap_m1.npy', umap_m1)
    np.save(DATA_DIR + 'umap_m5.npy', umap_m5)