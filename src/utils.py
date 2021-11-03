from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import os
import math

import numpy as np
from sklearn.neighbors import kneighbors_graph
from tqdm import tqdm
import torch

from dotenv import load_dotenv
load_dotenv()
DATA_DIR = os.environ.get("DATA_DIR")
MODELS_DIR = os.environ.get("MODELS_DIR")


np.random.seed(42)


def check_symmetric(a, rtol=1e-05, atol=1e-05):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def construct_knn_graph(data, n_neighbors):
    graph = kneighbors_graph(data, n_neighbors, n_jobs=-1).toarray()
    for i in range(graph.shape[0]):
        for j in range(graph.shape[0]):
            v = max(graph[i, j], graph[j, i])
            graph[i, j] = v
            graph[j, i] = v
    return graph


def get_laplacian_from_graph(graph):
    degree = np.diag(np.sum(graph, axis=1))
    return degree - graph


def get_laplacian_from_tome_data(data, n_neighbors=None):
    if not n_neighbors:
        n_neighbors = int(math.sqrt(data.shape[0]))
    graph = construct_knn_graph(data, n_neighbors=n_neighbors)
    laplacian = get_laplacian_from_graph(graph)

    return laplacian


def get_laplacian_eig_from_laplacian(laplacian):
    lambda_l, v_l = np.linalg.eig(laplacian)
    idx = lambda_l.argsort()
    lambda_l = lambda_l[idx]
    v_l = v_l[:, idx]

    return lambda_l, v_l


def get_laplacian_coefficients(f, laplacian_v):
    return get_coefficients_in_basis(f, laplacian_v)


def project_onto_basis(vec, basis, k=None):
    output_vec = np.zeros(vec.shape[0])
    if not k:
        k = basis.shape[1]
    for i in range(k):
        vk = basis[:, i]
        output_vec += np.dot(vk, vec) * vk
    return output_vec


def get_coefficients_in_basis(vec, basis):
    output_vec = np.zeros(basis.shape[1])
    for i in range(basis.shape[1]):
        output_vec[i] = np.dot(basis[:, i], vec)
    return output_vec


def graph_to_tome_space(data, distribution_vec, weighted=False):
    # Takes in data::ndarray (NxN) which is the Tome data, and distribution_vec::ndarray (Nx1) which is a reconstruction of
    # an indicator function over the graph
    ## Returns (max_tome, weighted_avg_tome)
    # max_tome is the most probable cell in the training set, weighted_avg_tome takes a weighted average over possible cells
    distribution_vec /= sum(distribution_vec)
    if weighted:
        v = np.dot(data.T, distribution_vec)
    else:
        imax = distribution_vec.argmax()
        v = data[imax, :]

    return v


def laplacian_coefficients_to_probability(coefficients, laplacian_v):
    function = np.zeros(laplacian_v.shape[0])
    for i in range(len(coefficients)):
        function += coefficients[i] * laplacian_v[:, i]
    function = np.exp(function)
    function /= sum(function)
    return function

# This needs some testing to verify the output is correct


def laplacian_on_weighted_matrix(matrix):
    L = np.zeros(matrix.shape)
    weights = np.sum(matrix, axis=1)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[0]):
            if i == j:
                L[i][j] = weights[i]
            else:
                L[i][j] = -matrix[i][j]
    return L


def is_identity(matrix, tol=1e-06):
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[0]):
            if i == j:
                if abs(matrix[i, j] - 1) > tol:
                    return False
            else:
                if abs(matrix[i, j]) > tol:
                    return False
    return True


def generate_binary_matrix(N_genes, N_binary, N_combinations=10):
    # N_genes is the total number of genes in our sample, N_binary is the number of genes in our binary sample
    # N_combinations is the dimension of the sample, i.e. how many binary lin combs we want
    # Column with exact proportion of genes, but not randomly ordered
    column_unshuffled = np.array([0] * (N_genes - N_binary) + [1] * N_binary)
    binary_matrix = []
    for _ in range(N_combinations):
        np.random.shuffle(column_unshuffled)
        binary_matrix.append(np.copy(column_unshuffled))

    binary_matrix = np.array(binary_matrix).T
    return binary_matrix


"""
Distance Metrics in Transcriptome Space
"""


def euclidean_distance(x, y):
    return np.linalg.norm(x-y) ** 2


def cross_entropy(y, yhat):
    loss = 0
    for i in range(len(y)):
        if math.isclose(y[i], 1, rel_tol=1e-05, abs_tol=1e-06):
            loss = + -math.log(yhat[i])
        else:
            loss += -math.log(1-yhat[i])
    return loss


def cross_entropy_on_matrix(Y, Yhat):
    loss = np.zeros(len(Y))
    for i in range(len(Y)):
        loss[i] = cross_entropy(Y[i, :], Yhat[i, :])
    return loss.mean()


def gaussian_centered_on_vertex(data, vertex, sigma=1, distance_metric=euclidean_distance):
    # $p\propto \exp(\frac{-x^2}{\sigma^2})$ where $x$ is the distance from the vertex
    p = np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        distance = distance_metric(data[i, :], vertex)
        p[i] = math.exp(-1 * distance * (1/(sigma ** 2)))
    p /= sum(p)
    return p


def train_model(model, train_dataloader, device, N_features=9781, labelled=True, epochs=20):
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = ExponentialLR(optimizer, gamma=0.8)
    criterion = nn.MSELoss()

    for epoch in tqdm(range(epochs)):
        loss = 0
        if labelled:
            for batch_features, batch_labels in train_dataloader:
                batch_features = batch_features.view(-1, N_features).to(device)
                optimizer.zero_grad()
                outputs = model(batch_features)
                train_loss = criterion(outputs, batch_labels)
                train_loss.backward()
                optimizer.step()
                loss += train_loss.item()
        else:
            for _, batch_features in enumerate(train_dataloader):
                batch_features = batch_features.view(-1, N_features).to(device)
                optimizer.zero_grad()
                outputs = model(batch_features)
                train_loss = criterion(outputs, batch_features)
                train_loss.backward()
                optimizer.step()
                loss += train_loss.item()
        scheduler.step()
        loss = loss / len(train_dataloader)
        print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))

    return model


def one_at(len_vec, one_idx):
    v = np.zeros(len_vec)
    v[one_idx] = 1
    return v



