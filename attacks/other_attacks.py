from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import  random
from utils.data_utils import save_csv
import networkx as nx
from utils.data_utils import get_normalized_adj
from torch.nn.functional import normalize
from scipy.special import erf
def n_neighbor(G, id, n_hop):
    node = [id]
    node_visited = set()
    neighbors = []

    while n_hop != 0:
        neighbors = []
        for node_id in node:
            node_visited.add(node_id)
            neighbors += [id for id in G.neighbors(node_id) if id not in node_visited]
        node = neighbors
        n_hop -= 1

        if len(node) == 0:
            return neighbors

    return list(set(neighbors))


def attack_set_by_InfMax_Unif(alpha, adj, attack_nodes,  K_Random_Walk):
    '''
    New sort method
    :param alpha: an int as the threshold in cutting too large element
    :param M: M is typically the original random walk M
    :param limit: limit is typically the args.num_node
    :param bar: an int used to set the threshold of degree that can be chosen to attack
    :param g: the graph, used to calculate the out_degree of an node
    :return: a list contains the indexs of nodes that needed to be attacked.
    '''
    data_size = len(adj)
    G = nx.from_numpy_matrix(adj.cpu().detach().numpy())
    degree =   G.degree()
    pro = normalize(torch.FloatTensor(adj), p=1, dim=1)
    Random_mat = pro
    Cand_degree = sorted([(degree[i], i) for i in range(data_size)], reverse=True)
    threshold = int(data_size * 0.1)
    bar, _ = Cand_degree[threshold]
    for i in range(K_Random_Walk - 1):
        Random_mat= torch.sparse.mm(Random_mat, pro)

    Random_mat = Random_mat.cpu().detach().numpy()
    s = np.zeros((Random_mat.shape[0], 1))  # zero vector
    res = []  # res vector

    # make those i has larger degree to -inf
    for i in range(Random_mat.shape[0]):
        if degree(i) > bar:
            Random_mat[:, i] = -float("inf")

    # debug
    # print("New_sort(debug): alpha = ", alpha)

    # Greedyly choose the point
    for _ in range(attack_nodes):
        L = np.minimum(s + Random_mat, alpha)
        L = L.sum(axis=0)
        i = np.argmax(L)
        res.append(i)
        s = s + Random_mat[:, i].reshape(Random_mat.shape[0], 1)
        Random_mat[:, i] = -float("inf")
        # delete neighbour
        negibours = n_neighbor(G, i, n_hop=1)
        for neighbor in negibours:
            Random_mat[:, neighbor] = -float("inf")
    return res


def attack_set_by_InfMax_Norm(sigma, attack_nodes, adj,K_Random_Walk ):
    '''
    New sort method
    :param alpha: an int as the threshold in cutting too large element
    :param M: M is typically the original random walk M
    :param limit: limit is typically the args.num_node
    :param bar: an int used to set the threshold of degree that can be chosen to attack
    :param g: the graph, used to calculate the out_degree of an node
    :return: a list contains the indexs of nodes that needed to be attacked.
    '''
    data_size = len(adj)
    G = nx.from_numpy_matrix(adj.cpu().detach().numpy())
    degree =   G.degree()
    pro = normalize(torch.FloatTensor(adj), p=1, dim=1)
    Random_mat = pro
    Cand_degree = sorted([(degree[i], i) for i in range(data_size)], reverse=True)
    threshold = int(data_size * 0.1)
    bar, _ = Cand_degree[threshold]

    for i in range(K_Random_Walk - 1):
        Random_mat= torch.sparse.mm(Random_mat, pro)
    Random_mat = Random_mat.cpu().detach().numpy()
    s = np.zeros((Random_mat.shape[0], 1))  # zero vector
    res = []  # res vector

    # make those i has larger degree to -inf
    for i in range(Random_mat.shape[0]):
        if  degree(i) > bar:
            Random_mat[:, i] = -float("inf")

    # debug
    # print("New_sort(debug): sigma = ", sigma)

    # Greedyly choose the point
    for _ in range(attack_nodes):
        L = erf((s + Random_mat) / (sigma * (2 ** 0.5)))
        L = L.sum(axis=0)
        i = np.argmax(L)
        res.append(i)
        s = s + Random_mat[:, i].reshape(Random_mat.shape[0], 1)
        Random_mat[:, i] = -float("inf")
        # delete neighbour
        negibours = n_neighbor(G, i, n_hop=1)
        for neighbor in negibours:
            Random_mat[:, neighbor] = -float("inf")
    return res


def attack_set_by_RWCS(K_Random_Walk,adj, attack_nodes):
    pro = normalize(torch.FloatTensor(adj), p=1, dim=1)
    Random_mat = pro
    for i in range(K_Random_Walk - 1):
        Random_mat= torch.sparse.mm(Random_mat, pro)
    score = Random_mat.sum(dim=0).cpu().detach().numpy()
    Dsort = score.argsort()[::-1]
    l = Dsort
    chosen_nodes = [l[i] for i in range(attack_nodes)]
    return chosen_nodes

def attack_set_by_GC_RWCS(adj,K_Random_Walk,  attack_nodes, beta):
    data_size = len(adj)
    G = nx.from_numpy_matrix(adj.cpu().detach().numpy())

    degree =   G.degree()



    Cand_degree = sorted([(degree[i], i) for i in range(data_size)], reverse=True)
    threshold = int(data_size * 0.1)
    bar, _ = Cand_degree[threshold]

    pro = normalize(torch.FloatTensor(adj), p=1, dim=1)

    Random_mat = pro
    for i in range(K_Random_Walk - 1):
        Random_mat = torch.sparse.mm(Random_mat, pro)
    W = torch.zeros(data_size, data_size)
    for i in range(data_size):
        value, index = torch.topk(Random_mat[i], beta)
        for j, ind in zip(value, index):
            if j != 0:
                W[i, ind] = 1
    SCORE = W.sum(dim=0)
    ind = []
    l = [i for i in range(data_size) if degree(i) <= bar]
    for _ in range(attack_nodes):
        cand = [(SCORE[i], i) for i in l]
        best = max(cand)[1]
        negibours = n_neighbor(G, best, n_hop=1)
        for neighbor in negibours:
            if neighbor in l:
                l.remove(neighbor)
        ind.append(best)
        for i in l:
            W[:, i] -= (W[:, best] > 0) * 1.0
        SCORE = torch.sum(W > 0, dim=0)
    return np.array(ind)



def attack_set_by_degree(adj, attack_nodes):
    G = nx.from_numpy_matrix(adj)
    D = G.degree()
    Degree = np.zeros(adj.shape[0])
    for i in range(adj.shape[0]):
        Degree[i] = D[i]
    # print(Degree)
    Dsort = Degree.argsort()[::-1]
    l = Dsort
    chosen_nodes = [l[i] for i in range(attack_nodes)]
    return chosen_nodes

def attack_set_by_pagerank(adj, attack_nodes):
    G = nx.from_numpy_matrix(adj)
    result = nx.pagerank(G)
    d_order = sorted(result.items(), key=lambda x: x[1], reverse=True)
    l = [x[0] for x in d_order]  # The sequence produced by pagerank algorithm
    chosen_nodes = [l[i] for i in range(attack_nodes)]
    return chosen_nodes
def attack_set_by_betweenness(adj, attack_nodes):
    G = nx.from_numpy_matrix(adj)
    result = nx.betweenness_centrality(G)
    # print(result)
    d_order = sorted(result.items(), key=lambda x: x[1], reverse=True)
    l = [x[0] for x in d_order]
    chosen_nodes = [l[i] for i in range(attack_nodes)]
    return chosen_nodes

def batch_saliency_map(input_grads):

    input_grads = input_grads.mean(dim=0)
    node_saliency_map = []
    for n in range(input_grads.shape[0]): # nth node
        node_grads = input_grads[n,:]
        node_saliency = torch.norm(F.relu(node_grads)).item()
        node_saliency_map.append(node_saliency)
    sorted_id = sorted(range(len(node_saliency_map)), key=lambda k: node_saliency_map[k], reverse=True)
    return node_saliency_map, sorted_id

def attack_set_by_saliency_map(input_grads, attack_nodes):
    node_saliency_map, sorted_id = batch_saliency_map(input_grads)
    chosen_nodes = [sorted_id[i] for i in range(attack_nodes)]
    return  chosen_nodes









def one_hot_tensor(y_batch_tensor, num_classes, device):
    y_tensor = torch.cuda.FloatTensor(y_batch_tensor.size(0),
                                      num_classes).fill_(0)
    y_tensor[np.arange(len(y_batch_tensor)), y_batch_tensor] = 1.0
    return y_tensor









def _ST_pgd_whitebox(model,
                  X,
                  y,
                  A_wave,
                  A,
                  edges,
                  edge_weights,
                  K,
                  epsilon,
                  num_steps,
                  Random,
                  step_size,
                  find_type,
                  device,
                  **kwargs):

    X_pgd = Variable(X.data, requires_grad=True)


    # first choose the nodes: randomly choose K nodes from total nodes
    batch_size_x, num_nodes, num_features, steps_length = X.size()
    ones_mat = torch.ones(1, 1, num_features, steps_length).to(device)  # [1,1 ,num of channel, number of time length]


    if find_type == 'random':
        list = [i for i in range(num_nodes)]
        index = random.sample(list, K)
    elif find_type == 'pagerank':
        index = attack_set_by_pagerank(A.cpu().detach().numpy(), K)
    elif find_type == 'betweeness':
        index = attack_set_by_betweenness(A.cpu().detach().numpy(), K)

    elif find_type == 'degree':
        index = attack_set_by_degree(A.cpu().detach().numpy(), K)

    elif find_type == 'saliency':
        X_saliency = Variable(X.data, requires_grad=True)
        saliency_steps = 5
        for _ in range(saliency_steps):

            if Random:
                random_noise = torch.FloatTensor(*X_saliency.shape).uniform_(-epsilon / 10, epsilon / 10).to(device)
                X_saliency = Variable(X_saliency.data + random_noise, requires_grad=True)

            opt_saliency = optim.SGD([X_saliency], lr=1e-3)
            opt_saliency.zero_grad()
            with torch.enable_grad():
                loss_saliency = nn.MSELoss()(model(X_saliency, A_wave, edges, edge_weights), y)
            loss_saliency.backward()

            eta = step_size * X_saliency.grad.data.sign()
            inputs_grad = X_saliency.grad.data

            X_saliency = Variable(X_saliency.data + eta, requires_grad=True)
            eta = torch.clamp(X_saliency.data - X.data, -epsilon, epsilon)
            X_saliency = Variable(X.data + eta, requires_grad=True)
            X_saliency = Variable(torch.clamp(X_saliency, 0, 1.0), requires_grad=True)

        index = attack_set_by_saliency_map(inputs_grad, K)
    else:
        raise  NameError

    chosen_attack_nodes = torch.zeros_like(X)
    chosen_attack_nodes[:, index, :, :] = ones_mat
    if Random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon/10, epsilon/10).to(device)
        X_pgd = Variable(X_pgd.data + chosen_attack_nodes * random_noise, requires_grad=True)




    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.MSELoss()(model(X_pgd,A_wave, edges, edge_weights), y)
        loss.backward()



        # second clamp the value according to  the neighbourhood value [min, max]
        # define the epsilon: stds: parameter free
        #X_fgsm = X_fgsm.data + epsilon * chosen_attack_nodes * X_fgsm.grad.data.sign()


        eta = step_size * chosen_attack_nodes * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)



    # print('err pgd (white-box): ', err_pgd)
    return X, X_pgd, index
def _ST_gpgd_whitebox(model,
                  X,
                  y,
                  A_wave,
                  A,
                  edges,
                  edge_weights,
                  K,
                  epsilon,
                  num_steps,
                  Random,
                  step_size,
                  find_type,
                  device,
                  **kwargs):

    X_pgd = Variable(X.data, requires_grad=True)


    # first choose the nodes: randomly choose K nodes from total nodes
    batch_size_x, num_nodes, num_features, steps_length = X.size()
    ones_mat = torch.ones(1, 1, num_features, steps_length).to(device)  # [1,1 ,num of channel, number of time length]


    if find_type == 'random':
        list = [i for i in range(num_nodes)]
        index = random.sample(list, K)
    elif find_type == 'pagerank':
        index = attack_set_by_pagerank(A.cpu().detach().numpy(), K)
    elif find_type == 'betweeness':
        index = attack_set_by_betweenness(A.cpu().detach().numpy(), K)

    elif find_type == 'degree':
        index = attack_set_by_degree(A.cpu().detach().numpy(), K)

    elif find_type == 'saliency':
        X_saliency = Variable(X.data, requires_grad=True)
        saliency_steps = 5
        for _ in range(saliency_steps):

            if Random:
                random_noise = torch.FloatTensor(*X_saliency.shape).uniform_(-epsilon / 10, epsilon / 10).to(device)
                X_saliency = Variable(X_saliency.data + random_noise, requires_grad=True)

            opt_saliency = optim.SGD([X_saliency], lr=1e-3)
            opt_saliency.zero_grad()
            with torch.enable_grad():
                loss_saliency = nn.MSELoss()(model(X_saliency, A_wave, edges, edge_weights), y)
            loss_saliency.backward()

            eta = step_size * X_saliency.grad.data.sign()
            inputs_grad = X_saliency.grad.data

            X_saliency = Variable(X_saliency.data + eta, requires_grad=True)
            eta = torch.clamp(X_saliency.data - X.data, -epsilon, epsilon)
            X_saliency = Variable(X.data + eta, requires_grad=True)
            X_saliency = Variable(torch.clamp(X_saliency, 0, 1.0), requires_grad=True)

        index = attack_set_by_saliency_map(inputs_grad, K)
    else:
        raise  NameError

    chosen_attack_nodes = torch.zeros_like(X)
    chosen_attack_nodes[:, index, :, :] = ones_mat
    if Random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon/10, epsilon/10).to(device)
        X_pgd = Variable(X_pgd.data + chosen_attack_nodes * random_noise, requires_grad=True)




    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.MSELoss()(model(X_pgd,A_wave, edges, edge_weights), y)
        loss.backward()



        # second clamp the value according to  the neighbourhood value [min, max]
        # define the epsilon: stds: parameter free
        #X_fgsm = X_fgsm.data + epsilon * chosen_attack_nodes * X_fgsm.grad.data.sign()


        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)

    eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon) * chosen_attack_nodes
    X_pgd = Variable(X.data + eta, requires_grad=True)
    X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)

    # print('err pgd (white-box): ', err_pgd)
    return X, X_pgd, index

