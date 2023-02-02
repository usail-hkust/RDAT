import logging
import os
import numpy as np
import torch
import random
import datetime
import csv
from mmcv.runner import get_dist_info
from mmcv import mkdir_or_exist
np.seterr(divide='ignore', invalid='ignore')
import itertools
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import os
import zipfile
import numpy as np
import torch
from six.moves import urllib
from scipy.sparse import coo_matrix
import scipy.sparse as sp
#from torch_geometric.utils import dense_to_sparse
def _download_url(url, save_path):  # pragma: no cover
    with urllib.request.urlopen(url) as dl_file:
        with open(save_path, "wb") as out_file:
            out_file.write(dl_file.read())


def load_PEMS_BAY_data():

    url = "https://graphmining.ai/temporal_datasets/PEMS-BAY.zip"

    # Check if zip file is in data folder from working directory, otherwise download
    if not os.path.isfile(
            os.path.join('../data/', "PEMS-BAY.zip")
    ):  # pragma: no cover
        if not os.path.exists('../data/'):
            os.makedirs('../data/')
        _download_url(url, os.path.join('../data/', "PEMS-BAY.zip"))

    if not os.path.isfile(
            os.path.join('../data/', "pems_adj_mat.npy")
    ) or not os.path.isfile(
        os.path.join('../data/', "pems_node_values.npy")
    ):  # pragma: no cover
        with zipfile.ZipFile(
                os.path.join('../data/', "PEMS-BAY.zip"), "r"
        ) as zip_fh:
            zip_fh.extractall('../data/')

    A = np.load(os.path.join('../data/', "pems_adj_mat.npy"))
    X = np.load(os.path.join('../data/', "pems_node_values.npy")).transpose(
        (1, 2, 0)
    )
    X = X.astype(np.float32)
    #print("X shape ", X.shape)
    # data shape (node dim time) (325, 2, 52105)
    # Normalise as in DCRNN paper (via Z-Score Method)
    means = np.mean(X, axis=(0, 2))

    X = X #- means.reshape(1, -1, 1)
    stds = np.std(X, axis=(0, 2))
    X = X #/ stds.reshape(1, -1, 1)

    return A, X, means, stds



def get_adjacent_matrix(distance_file: str, num_nodes: int, id_file: str = None, graph_type="connect") -> np.array:
    """
    :param distance_file: str, path of csv file to save the distances between nodes.
    :param num_nodes: int, number of nodes in the graph
    :param id_file: str, path of txt file to save the order of the nodes.就是排序节点的绝对编号所用到的，这里排好了，不需要
    :param graph_type: str, ["connect", "distance"]，这个就是考不考虑节点之间的距离
    :return:
        np.array(N, N)
    """
    A = np.zeros([int(num_nodes), int(num_nodes)])  # 构造全0的邻接矩阵

    if id_file:  # 就是给节点排序的绝对文件，这里是None，则表示不需要
        with open(id_file, "r") as f_id:
            # 将绝对编号用enumerate()函数打包成一个索引序列，然后用node_id这个绝对编号做key，用idx这个索引做value
            node_id_dict = {int(node_id): idx for idx, node_id in enumerate(f_id.read().strip().split("\n"))}

            with open(distance_file, "r") as f_d:
                f_d.readline() # 表头，跳过第一行.
                reader = csv.reader(f_d) # 读取.csv文件.
                for item in reader:   # 将一行给item组成列表
                    if len(item) != 3: # 长度应为3，不为3则数据有问题，跳过
                        continue
                    i, j, distance = int(item[0]), int(item[1]), float(item[2]) # 节点i，节点j，距离distance
                    if graph_type == "connect":  # 这个就是将两个节点的权重都设为1，也就相当于不要权重
                        A[node_id_dict[i], node_id_dict[j]] = 1.
                        A[node_id_dict[j], node_id_dict[i]] = 1.
                    elif graph_type == "distance":  # 这个是有权重，下面是权重计算方法
                        A[node_id_dict[i], node_id_dict[j]] = 1. / distance
                        A[node_id_dict[j], node_id_dict[i]] = 1. / distance
                    else:
                        raise ValueError("graph type is not correct (connect or distance)")
        return A

    with open(distance_file, "r") as f_d:
        f_d.readline()  # 表头，跳过第一行.
        reader = csv.reader(f_d)  # 读取.csv文件.
        for item in reader:
            if len(item) != 3:
                continue
            i, j, distance = int(item[0]), int(item[1]), float(item[2])

            if graph_type == "connect":
                A[i, j], A[j, i] = 1., 1.
            elif graph_type == "distance":
                A[i, j] = 1. / distance
                A[j, i] = 1. / distance
            else:
                raise ValueError("graph type is not correct (connect or distance)")

    return A

def load_PEMS_D4_data():
    data = np.load("../data/PeMS_04/PeMS04.npz")
    A = get_adjacent_matrix(distance_file="../data/PeMS_04/PeMS04.csv", num_nodes=307)
    X = data['data'].transpose([1, 0, 2])[:, :, 0][:, :,np.newaxis]


    X = X.transpose([0, 2, 1])
    X = X.astype(np.float32)

    # Normalise as in DCRNN paper (via Z-Score Method)
    means = np.mean(X, axis=(0, 2))

    X = X #- means.reshape(1, -1, 1)
    stds = np.std(X, axis=(0, 2))
    X = X #/ stds.reshape(1, -1, 1)

    return A, X, means, stds


def load_METR_LA_data():
    url = "https://graphmining.ai/temporal_datasets/METR-LA.zip"

    # Check if zip file is in data folder from working directory, otherwise download
    if not os.path.isfile(
            os.path.join('../data/METRLA/', "METR-LA.zip")
    ):  # pragma: no cover
        if not os.path.exists('../data/METRLA/'):
            os.makedirs('../data/METRLA/')
        _download_url(url, os.path.join('../data/METRLA/', "METR-LA.zip"))

    if not os.path.isfile(
            os.path.join('../data/METRLA/', "adj_mat.npy")
    ) or not os.path.isfile(
        os.path.join('../data/METRLA/', "node_values.npy")
    ):  # pragma: no cover
        with zipfile.ZipFile(
                os.path.join('../data/METRLA/', "METR-LA.zip"), "r"
        ) as zip_fh:
            zip_fh.extractall('../data/METRLA/')

    A = np.load(os.path.join('../data/METRLA/', "adj_mat.npy"))
    X = np.load(os.path.join('../data/METRLA/', "node_values.npy")).transpose(
        (1, 2, 0)
    )
    X = X.astype(np.float32)

    # Normalise as in DCRNN paper (via Z-Score Method)
    means = np.mean(X, axis=(0, 2))

    X = X #- means.reshape(1, -1, 1)
    stds = np.std(X, axis=(0, 2))
    X = X #/ stds.reshape(1, -1, 1)

    return A, X, means, stds







def load_los_data():
    los_adj = pd.read_csv(r'data/los_adj.csv', header=None)
    adj = np.mat(los_adj)
    los_tf = pd.read_csv(r'data/los_speed.csv')
    return los_tf, adj

def load_hk_data():
    A = pd.read_csv(r'data/hk_adj.csv', header=None)
    A = np.mat(A)
    X = pd.read_csv(r'data/hk_speed.csv')
    X = X.astype(np.float32)
    return A, np.array(X).astype(np.float32)

def load_los_locations():
    los_locations = pd.read_csv(r'data/los_locations.csv', header=None)
    return np.array(los_locations)

def load_hk_locations():
    hk_locations = pd.read_csv(r'data/hk_locations.csv', header=None)
    return np.array(hk_locations)

def load_la_locations():
    la_locations = pd.read_csv(r'data/la_locations.csv', header=None)
    return np.array(la_locations)










def load_metr_la_data():
    if (not os.path.isfile("data/adj_mat.npy")
            or not os.path.isfile("data/node_values.npy")):
        with zipfile.ZipFile("data/METR-LA.zip", 'r') as zip_ref:
            zip_ref.extractall("data/")

    A = np.load("data/adj_mat.npy")
    X = np.load("data/node_values.npy").transpose((1, 2, 0))
    X = X.astype(np.float32)


    means = np.mean(X, axis=(0, 2))

    mean = np.mean(X)
    #X = X - means.reshape(1, -1, 1)

    stds = np.std(X, axis=(0, 2))
    #X = X / stds.reshape(1, -1, 1)


    return A, X, means, stds

def z_inverse_metla(predict, target, means, stds):
    # metr la data only predict dim 0 (feature)
    predict = predict*stds[0] + means[0]
    target = target*stds[0] + means[0]
    return predict, target

def mae(pred, true):
    MAE = np.mean(np.absolute(pred - true))
    return MAE
def rmse(pred, true):

    RMSE = np.sqrt(np.mean(np.square(pred-true)))
    return RMSE
def mape(pred, true):
    y_true, y_pred = np.array(true), np.array(pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))
#Root Relative Squared Error

def rrse(pred, true):
    mean = true.mean()
    return np.divide(np.sqrt(np.sum((pred-true) ** 2)), np.sqrt(np.sum((true-mean) ** 2)))

def All_Metrics(pred, true):
    # np.array
    assert type(pred) == type(true)
    MAE = mae(pred, true)
    RMSE = rmse(pred, true)
    RRSE = rrse(pred, true)

    return MAE, RMSE, RRSE

def local_mae(pred, true):
    local_MAE = np.mean(np.absolute(pred - true))

    return local_MAE
def local_rmse(pred, true):

    RMSE = np.sqrt(np.mean(np.square(pred-true)) )
    return RMSE

def All_Local_Metrics(pred, true):
    # np.array
    assert type(pred) == type(true)
    MAE = local_mae(pred, true)
    RMSE = local_rmse(pred, true)


    return MAE, RMSE

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx



def get_normalized_adj(A):
    """
    Returns the degree normalized adjacency matrix.
    """
    A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_wave


def generate_dataset(X, num_timesteps_input, num_timesteps_output):
    """
    Takes node features for the graph and divides them into multiple samples
    along the time-axis by sliding a window of size (num_timesteps_input+
    num_timesteps_output) across it in steps of 1.
    :param X: Node features of shape (num_vertices, num_features,
    num_timesteps)
    :return:
        - Node features divided into multiple samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_input).
        - Node targets for the samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_output).
    """
    # Generate the beginning index and the ending index of a sample, which
    # contains (num_points_for_training + num_points_for_predicting) points
    indices = [(i, i + (num_timesteps_input + num_timesteps_output)) for i
               in range(X.shape[2] - (
                num_timesteps_input + num_timesteps_output) + 1)]

    # Save samples

    features, target = [], []
    for i, j in indices:
        features.append(
            X[:, :, i: i + num_timesteps_input])#.transpose((1,0, 2))


        target.append(X[:, 0, i + num_timesteps_input: j])
        #print('======================preparing data sets======================')
        # features.append(X[:, :, i: i + num_timesteps_input].transpose((0, 2, 1)))
        #target.append(X[:, :, i + num_timesteps_input: j].transpose((0, 2, 1)))
        #print('features shape', np.array(features).shape)
    #print('Finished!!!')
    return torch.from_numpy(np.array(features)), \
           torch.from_numpy(np.array(target))




def _savefig(name='./image.jpg', dpi=300):
    plt.savefig(name, dpi=dpi,bbox_inches = 'tight')
    plt.close()
    print('Image saved at {}'.format(name))




def save_csv(name, save_list, root='./data_ana', msg=True, devide=True):
    mkdir_or_exist(root)
    name = os.path.join(root, name)
    one_line = []
    for save_line in save_list:
        assert isinstance(save_line, list)
        one_line.extend(save_line)
        if devide:
            one_line.append(' ')
    with open(name, 'a+', newline='') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(one_line)
    if msg:
        print("Date written to {}".format(name))

class CountMeter(object):
    def __init__(self, num_classes, non_zero=True, save_raw=False):
        self.num_classes = num_classes
        self.non_zero = non_zero
        self.save_raw = save_raw
        self.reset()

    def reset(self):
        self.n = 0
        # count non-zero
        self.n_per_class = np.zeros(self.num_classes)
        self.sum_values = np.zeros(self.num_classes)
        self.avg_values = np.zeros(self.num_classes)
        if self.save_raw:
            self.raw_values = None

    def update(self, data, target=None):
        self.n += data.shape[0]

        # data (n,), target=(n,)
        if len(data.shape) == 1:
            assert target is not None
            self.sum_values[target] += data
            for i in range(self.num_classes):
                self.n_per_class[i] += (target == i).sum()

        # data (n, dim), target=(n, dim) / None
        else:
            self.sum_values += np.sum(data, dim=0)
            if target is None:
                self.n_per_class += np.sum(data>0, dim=0)
            else:
                self.n_per_class += np.sum(target, dim=0)

        if self.non_zero:
            self.avg_values = self.sum_values / self.n_per_class
        else:
            self.avg_values = self.sum_values / self.n

        if self.save_raw:
            if self.raw_values is None:
                self.raw_values = data
            else:
                self.raw_values = np.vstack((self.raw_values, data))

    def save_data(self):
        raise NotImplementedError


if __name__ == '__main__':
    header = ['model_dir', 'nat_acc', 'rob_acc',' ']
    class_no = np.arange(100).tolist()
    header.extend(class_no)
    header.append(' ')
    header.extend(class_no)
    save_csv('./CIFAR100_all_results.csv',  header)
