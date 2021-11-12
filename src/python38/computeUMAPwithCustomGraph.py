from sklearn.neighbors import NearestNeighbors
import umap
import pandas as pd
import numpy as np

import os
os.chdir('/Users/kushagrasharma/coding/hormozlab/src')

DATA_DIR = "/Users/kushagrasharma/coding/hormozlab/data/"

if __name__ == "__main__":
    train_full = pd.read_csv(
        DATA_DIR + 'scvi_train_set_gapdh.csv', header=None).to_numpy()

    graph = np.load(DATA_DIR + "adjacency_15NN.npy")

    indices = []

    for i in range(len(graph)):
        cur_indices = []
        for j in range(len(graph)):
            if graph[i, j]:
                cur_indices.append(j)
        indices.append(cur_indices)

    distances = []

    for i in range(len(indices)):
        cur_distances = []
        for j in indices[i]:
            cur_distances.append(np.linalg.norm(
                train_full[i, :] - train_full[j, :]))
        distances.append(cur_distances)

    zipped = [zip(distances[i], indices[i]) for i in range(len(indices))]
    sorted_pairs = [sorted(zips) for zips in zipped]

    tuples = [zip(*sorted_pair) for sorted_pair in sorted_pairs]
    unpacked = [[list(tup) for tup in tups] for tups in tuples]

    distances_sorted = [x[0] for x in unpacked]
    idx_sorted = [x[1] for x in unpacked]

    knn_umap = np.array(umap.UMAP(n_neighbors=15).fit_transform(train_full))

    np.save(DATA_DIR + "umapWithCustomGraph.npy", knn_umap)
