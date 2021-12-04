from sklearn.neighbors import NearestNeighbors
import umap
import pandas as pd
import numpy as np

import os
os.chdir('/Users/kushagrasharma/coding/hormozlab/src')

DATA_DIR = "/Users/kushagrasharma/coding/hormozlab/data/"

if __name__ == '__main__':
    test_full = pd.read_csv(
        DATA_DIR + 'scvi_test_set_gapdh.csv', header=None).to_numpy()
    test_full_saver = pd.read_csv(
        DATA_DIR + 'saver_test_set_gapdh.csv', header=None).to_numpy()

    # m1_results = np.load(DATA_DIR + 'reconstructed_tome_m1.npy')
    m1_results_saver = np.load(
        DATA_DIR + 'reconstructed_tome_m1_gapdh_saver.npy')

    # m5_results = np.load(DATA_DIR + 'reconstructed_tome_m5.npy')
    m5_results_saver = np.load(DATA_DIR + 'reconstructed_tome_m5_saver.npy')

    # reducer_m1 = umap.UMAP()
    # reducer_m5 = umap.UMAP()
    reducer_m1_saver = umap.UMAP()
    reducer_m5_saver = umap.UMAP()

    # concat_m1 = np.concatenate((test_full, m1_results))
    # concat_m5 = np.concatenate((test_full, m5_results))
    # concat_m1_saver = np.concatenate((test_full_saver, m1_results_saver))
    # concat_m5_saver = np.concatenate((test_full_saver, m5_results_saver))
    concat_m1_saver_scvi = np.concatenate((test_full, m1_results_saver))
    concat_m5_saver_scvi = np.concatenate((test_full, m5_results_saver))

    # umap_m1 = reducer_m1.fit_transform(concat_m1)
    # umap_m5 = reducer_m5.fit_transform(concat_m5)
    # umap_m1_saver = reducer_m1_saver.fit_transform(concat_m1_saver)
    # umap_m5_saver = reducer_m5_saver.fit_transform(concat_m5_saver)
    umap_m1_saver_scvi = reducer_m1_saver.fit_transform(concat_m1_saver_scvi)
    umap_m5_saver_scvi = reducer_m5_saver.fit_transform(concat_m5_saver_scvi)

    # np.save(DATA_DIR + 'umap_m1.npy', umap_m1)
    # np.save(DATA_DIR + 'umap_m5.npy', umap_m5)
    # np.save(DATA_DIR + 'umap_m1_saver.npy', umap_m1_saver)
    # np.save(DATA_DIR + 'umap_m5_saver.npy', umap_m5_saver)

    np.save(DATA_DIR + 'umap_m1_saver_scvi.npy', umap_m1_saver_scvi)
    np.save(DATA_DIR + 'umap_m5_saver_scvi.npy', umap_m5_saver_scvi)
