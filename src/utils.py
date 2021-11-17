from torch.optim.lr_scheduler import ExponentialLR
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import os
import math
from copy import deepcopy

import numpy as np
from sklearn.neighbors import kneighbors_graph
from tqdm import tqdm

from dotenv import load_dotenv
load_dotenv()
DATA_DIR = os.environ.get("DATA_DIR")
MODELS_DIR = os.environ.get("MODELS_DIR")


np.random.seed(42)

"""
Graph, Laplacian Helper Functions
"""


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


def graph_to_tome_space(data, distribution_vec, weighted=True):
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


def laplacian_coefficients_to_probability(coefficients, laplacian_v, norm_fn=lambda x: x ** 2):
    function = np.zeros(laplacian_v.shape[0])
    for i in range(len(coefficients)):
        function += coefficients[i] * laplacian_v[:, i]
    function = norm_fn(function)
    function /= sum(function)
    return function


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
Distance Metrics / Loss Functions
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


"""
Gaussian Helper Functions
"""


def gaussian_centered_on_vertex(data, vertex, sigma=1, distance_metric=euclidean_distance):
    # $p\propto \exp(\frac{-x^2}{\sigma^2})$ where $x$ is the distance from the vertex
    p = np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        distance = distance_metric(data[i, :], vertex)
        p[i] = math.exp(-1 * distance * (1/(sigma ** 2)))
    p /= sum(p)
    return p


"""
Model Functions
"""


def get_validation_loss(model, valid_dataloader, device, N_features=10, criterion=nn.MSELoss(), labelled=True):
    loss = 0
    if labelled:
        for batch_features, batch_labels in valid_dataloader:
            batch_features = batch_features.view(-1,
                                                 N_features).to(device)
            outputs = model(batch_features)
            valid_loss = criterion(outputs, batch_labels)
            loss += valid_loss.item()
    else:
        for _, batch_features in enumerate(valid_dataloader):
            batch_features = batch_features.view(-1,
                                                 N_features).to(device)
            outputs = model(batch_features)
            valid_loss = criterion(outputs, batch_features)
            loss += valid_loss.item()

    loss /= len(valid_dataloader)
    return loss


def train_model(model, train_dataloader, device, valid_dataloader=None, N_features=9781, labelled=True, epochs=20, criterion=nn.MSELoss(), patience=4):
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = ExponentialLR(optimizer, gamma=0.8)
    train_losses = []
    valid_losses = []

    valid_loss_nondecreasing_epochs = 0
    best_valid_loss = math.inf
    best_model = None

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
        train_losses.append(loss)
        if valid_dataloader:
            valid_loss = get_validation_loss(
                model, valid_dataloader, device, N_features=N_features, criterion=criterion, labelled=labelled)
            valid_losses.append(valid_loss)

            print("epoch : {}/{}, train loss = {:.6f}, valid loss = {:.6f}".format(
                epoch + 1, epochs, loss, valid_loss))

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_model = deepcopy(model)

            if epoch > 0:
                if valid_loss >= valid_losses[-2]-1e-6:
                    valid_loss_nondecreasing_epochs += 1
                else:
                    valid_loss_nondecreasing_epochs = 0
                if valid_loss_nondecreasing_epochs > patience:
                    print("EARLY STOP, NONDECREASING VALID LOSS")
                    break
        else:
            print("epoch : {}/{}, train loss = {:.6f}".format(epoch + 1, epochs, loss))

    if best_model:
        print("Returning best model with validation loss {}".format(best_valid_loss))
        return best_model
    return model


def one_at(len_vec, one_idx):
    v = np.zeros(len_vec)
    v[one_idx] = 1
    return v


def normalize(X):
    if torch.is_tensor(X):
        X = X.detach().numpy()
    row_sums = X.sum(axis=1)
    zero_indexes = np.argwhere(row_sums == 0)
    for i in zero_indexes:
        X[i, :] = np.ones(len(X[i, :]))
    row_sums = X.sum(axis=1)
    X = X / row_sums[:, np.newaxis]
    return X


def transform(X, transforms):
    d = torch.clone(X)
    for transform in transforms:
        d = transform(d)
    return d


def transform_and_compute_error(X, Y, transforms, error):
    # Returns the final reconstruction and the error
    d = transform(X, transforms)
    return d, error(Y, d)


def load_binary_matrix(N_genes=9781, N_binary=50, N_combinations=10):
    binary_matrix_filepath = MODELS_DIR + 'binary_matrix.npy'

    if os.path.exists(binary_matrix_filepath):
        binary_matrix = np.load(binary_matrix_filepath)
    else:
        binary_matrix = generate_binary_matrix(
            N_genes, N_binary, N_combinations)
        np.save(binary_matrix_filepath, binary_matrix)

    return binary_matrix
