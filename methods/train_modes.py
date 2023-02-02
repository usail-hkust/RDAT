import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import  random
import networkx as nx

import copy
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
    # print('saliency_map')
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


def policy_decisions(sampled_actions, topk):
    node_selection_pro = sampled_actions
    sorted_id = sorted(range(len(node_selection_pro)), key=lambda k: node_selection_pro[k], reverse=True)
    decisions = [sorted_id[i] for i in range(topk)]
    return decisions



# natural training
def plain_train(model, x_natural, A_wave, edges, edge_weights,y, **kwargs):

    outs = model(x_natural, A_wave, edges, edge_weights)
    loss_criterion = torch.nn.MSELoss()
    loss = loss_criterion(outs, y)

    return loss










def compute_cost(x_natural,
                 y,
                 index,
                 ones_mat,
                 rand_start_mode,
                 rand_start_step,
                 epsilon,
                 distance,
                 perturb_steps,
                 model,
                 A_wave,
                 edges,
                 edge_weights,
                 step_size,
                 **kwargs):

    model.eval()
    chosen_attack_nodes = torch.zeros_like(x_natural)
    chosen_attack_nodes[:, index, :, :] = ones_mat

    if rand_start_mode == 'gaussian':
        x_adv = x_natural.detach() + chosen_attack_nodes * rand_start_step * 0.001 * torch.randn(
            x_natural.shape).cuda().detach()
    elif rand_start_mode == 'uniform':
        x_adv = x_natural.detach() + chosen_attack_nodes * rand_start_step * (epsilon / 10) * torch.rand(
            x_natural.shape).cuda().detach()
    else:
        raise NameError

    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()

            with torch.enable_grad():
                loss_mse = nn.MSELoss()(model(x_adv, A_wave, edges, edge_weights), y)
            grad = torch.autograd.grad(loss_mse, [x_adv])[0]
            x_adv = x_adv.detach() + chosen_attack_nodes * step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        raise NotImplementedError
    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)

    outs = model(x_adv, A_wave, edges, edge_weights)
    loss_criterion = torch.nn.MSELoss()
    cost = loss_criterion(outs, y)

    torch.cuda.empty_cache()
    return cost


def compute_baseline_cost(x_natural,
                 y,
                 index,
                 rand_start_mode,
                 epsilon,
                 model,
                 A_wave,
                 edges,
                 edge_weights,
                 **kwargs):



    # first choose the nodes: randomly choose K nodes from total nodes
    batch_size_x, num_nodes, num_features, steps_length = x_natural.size()
    ones_mat = torch.ones(1, 1, num_features, steps_length).cuda()  # [1,1 ,num of channel, number of time length]
    chosen_attack_nodes = torch.zeros_like(x_natural)
    chosen_attack_nodes[:, index, :, :] = ones_mat


    model.eval()

    with torch.no_grad():

        if rand_start_mode == 'gaussian':
            x_adv = x_natural.detach() + chosen_attack_nodes  * epsilon  * torch.randn(
                x_natural.shape).cuda().detach()
        elif rand_start_mode == 'uniform':
            x_adv = x_natural.detach() + chosen_attack_nodes  * epsilon  * torch.rand(
                x_natural.shape).cuda().detach()
        else:
            raise NameError

        x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)

        outs = model(x_adv, A_wave, edges, edge_weights)
        loss_criterion = torch.nn.MSELoss(reduction="none")
        cost = loss_criterion(outs, y)

    return cost.mean(dim=-1).mean(dim=-1)

def compute_policy_cost(device,x_natural,
                 y,
                 tours,
                 rand_start_mode,
                 epsilon,
                 model,
                 A_wave,
                 edges,
                 edge_weights,
                 **kwargs):

    # first choose the nodes: randomly choose K nodes from total nodes
    batch_size_x, num_nodes, num_features, steps_length = x_natural.size()
    tours = tours[:,:,None].repeat(1,1,num_features * steps_length)
    mask_matrix = torch.zeros(batch_size_x, num_nodes, num_features * steps_length).to(device)
    mask_matrix = mask_matrix.scatter(dim=1, index=tours.long(), value=torch.tensor(1))
    mask_matrix = mask_matrix.reshape(batch_size_x, num_nodes, num_features, steps_length)


    model.eval()

    with torch.no_grad():

        if rand_start_mode == 'gaussian':
            x_adv = x_natural.detach() + mask_matrix  * epsilon  * torch.randn(
                x_natural.shape).cuda().detach()
        elif rand_start_mode == 'uniform':
            x_adv = x_natural.detach() + mask_matrix  * epsilon  * torch.rand(
                x_natural.shape).cuda().detach()
        else:
            raise NameError

        x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)

        outs = model(x_adv, A_wave, edges, edge_weights)
        loss_criterion = torch.nn.MSELoss(reduction="none")
        cost = loss_criterion(outs, y)

    return cost.mean(dim=-1).mean(dim=-1)


def rollout_baseline(model,
                     x_natural,
                     A_wave,
                     edges,
                     edge_weights,
                     y,
                     step_size=0.1,
                     epsilon=0.5,
                     K=21,
                     Random=True,
                     saliency_steps=1,
                     **kwargs):
    model.eval()

    X_saliency = Variable(x_natural.data, requires_grad=True)

    for _ in range(saliency_steps):

        if Random:
            random_noise = torch.FloatTensor(*X_saliency.shape).uniform_(-epsilon / 10, epsilon / 10).cuda()
            X_saliency = Variable(X_saliency.data + random_noise, requires_grad=True)

        opt_saliency = optim.SGD([X_saliency], lr=1e-3)
        opt_saliency.zero_grad()
        with torch.enable_grad():
            loss_saliency = nn.MSELoss()(model(X_saliency, A_wave, edges, edge_weights), y)
        loss_saliency.backward()

        eta = step_size * X_saliency.grad.data.sign()
        inputs_grad = X_saliency.grad.data

        X_saliency = Variable(X_saliency.data + eta, requires_grad=True)
        eta = torch.clamp(X_saliency.data - x_natural.data, -epsilon, epsilon)
        X_saliency = Variable(x_natural.data + eta, requires_grad=True)
        X_saliency = Variable(torch.clamp(X_saliency, 0, 1.0), requires_grad=True)
        # print("X_saliency", X_saliency)
    index = attack_set_by_saliency_map(inputs_grad, K)

    return index


def ST_pgd_adv_policy_Atten_train(model,
                            x_natural,
                            A_wave,
                            edges,
                            edge_weights,
                            y,
                            optimizer,
                            policynet,
                            policy_optimizer,
                            num_samples,
                            device,
                            constant,
                            is_known_first_node,
                            step_size=0.1,
                            epsilon=0.5,
                            perturb_steps=5,
                            distance='l_inf',
                            rand_start_mode='uniform',
                            rand_start_step=1,
                            K=21,
                            baseline = "random",
                            **kwargs):

    batch_size_x, num_nodes, num_features, steps_length = x_natural.size()
    model.eval()
    if baseline == "saliency":
        baseline_params = dict(model = model,x_natural = x_natural,A_wave = A_wave,
            edges = edges,edge_weights = edge_weights,y= y,step_size= step_size,epsilon= epsilon,K= K,
            Random= True,saliency_steps=1)
        index = rollout_baseline(**baseline_params)
    elif baseline == "random":
        list_random = [i for i in range(num_nodes)]
        index = random.sample(list_random, K)
    elif baseline == "mix_search":
        beta_search = 0.2
        baseline_params = dict(model = model,x_natural = x_natural,A_wave = A_wave,
            edges = edges,edge_weights = edge_weights,y= y,step_size= step_size,epsilon= epsilon,K= K - int(beta_search * K),
            Random= True,saliency_steps=1)
        index = rollout_baseline(**baseline_params)
        list_random = [i for i in range(num_nodes)]
        index_random = random.sample(list_random, K)
        intersection_index = list(set(index).intersection(set(index_random)))
        difference_index = list(set(index_random).difference(set(intersection_index)))
        index = list(set(index).union(set(difference_index[0:int(beta_search * K)])))
    else:
        raise NameError


    if is_known_first_node:
        first_node_id = index[0]
    else:
        first_node_id = None
    baseline_cost_params = dict(x_natural =x_natural,y = y,index = index,
                 rand_start_mode = rand_start_mode,epsilon = epsilon,model = model,A_wave = A_wave,
                 edges = edges,edge_weights = edge_weights)

    baseline_cost = compute_baseline_cost(**baseline_cost_params)




    policynet.train()

    rewarding = []
    #constant = 1e-03
    for i in range(num_samples):
        ll, tours = policynet(x_natural,first_node_id)

        compute_policy_cost_params = dict(device = device, x_natural = x_natural,y = y, tours = tours,rand_start_mode = rand_start_mode,epsilon = epsilon,model = model,
                 A_wave = A_wave,edges = edges,edge_weights = edge_weights)

        policy_cost = compute_policy_cost(**compute_policy_cost_params)
        policy_optimizer.zero_grad()
        policy_gradicdents = ((policy_cost.to(device) - baseline_cost.to(device) + constant) * ll).mean()
        policy_gradicdents.backward()
        nn.utils.clip_grad_norm_(policynet.parameters(), max_norm = 1.0, norm_type = 2)
        policy_optimizer.step()
        #print("reward",(policy_cost - baseline_cost+constant).detach().cpu().numpy())
        rewarding.append((policy_cost - baseline_cost + constant).sum().detach().cpu().numpy())
        print("reward:", (policy_cost - baseline_cost+ constant).sum().detach().cpu().numpy())


    tours = tours[:,:,None].repeat(1,1,num_features * steps_length)
    mask_matrix = torch.zeros(batch_size_x, num_nodes, num_features * steps_length).to(device)
    mask_matrix = mask_matrix.scatter(dim=1, index=tours.long(), value=torch.tensor(1))
    mask_matrix = mask_matrix.reshape(batch_size_x, num_nodes, num_features, steps_length)



    if rand_start_mode == 'gaussian':
        x_adv = x_natural.detach() + mask_matrix * rand_start_step * 0.001 * torch.randn(
            x_natural.shape).cuda().detach()
    elif rand_start_mode == 'uniform':
        x_adv = x_natural.detach() + mask_matrix * rand_start_step * (epsilon / 10) * torch.rand(
            x_natural.shape).cuda().detach()
    else:
        raise NameError

    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()

            with torch.enable_grad():
                loss_mse = nn.MSELoss()(model(x_adv, A_wave, edges, edge_weights), y)
            grad = torch.autograd.grad(loss_mse, [x_adv])[0]
            x_adv = x_adv.detach() + mask_matrix * step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        raise NotImplementedError

    model.train()
    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss

    outs = model(x_adv, A_wave, edges, edge_weights)
    loss_criterion = torch.nn.MSELoss()
    loss = loss_criterion(outs, y)



    return loss, np.mean(np.stack(rewarding))



def ST_pgd_adv_policy_Atten_dist_offline_train(model,
                            x_natural,
                            A_wave,
                            edges,
                            edge_weights,
                            y,
                            optimizer,
                            teacher_model,
                            epoch,
                            policynet,
                            device,
                            step_size=0.1,
                            epsilon=0.5,
                            perturb_steps=5,
                            distance='l_inf',
                            rand_start_mode='uniform',
                            rand_start_step=1,
                            alpha_reg = 0.2,
                            **kwargs):
    model.eval()

    batch_size_x, num_nodes, num_features, steps_length = x_natural.size()


    first_node_id = None
    policynet.eval()



    _, tours = policynet(x_natural,first_node_id)


    tours = tours[:,:,None].repeat(1,1,num_features * steps_length)
    mask_matrix = torch.zeros(batch_size_x, num_nodes, num_features * steps_length).to(device)
    mask_matrix = mask_matrix.scatter(dim=1, index=tours.long(), value=torch.tensor(1))
    mask_matrix = mask_matrix.reshape(batch_size_x, num_nodes, num_features, steps_length)



    if rand_start_mode == 'gaussian':
        x_adv = x_natural.detach() + mask_matrix * rand_start_step * 0.001 * torch.randn(
            x_natural.shape).cuda().detach()
    elif rand_start_mode == 'uniform':
        x_adv = x_natural.detach() + mask_matrix * rand_start_step * (epsilon / 10) * torch.rand(
            x_natural.shape).cuda().detach()
    else:
        raise NameError

    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()

            with torch.enable_grad():
                loss_mse = nn.MSELoss()(model(x_adv, A_wave, edges, edge_weights), y)
            grad = torch.autograd.grad(loss_mse, [x_adv])[0]
            x_adv = x_adv.detach() + mask_matrix * step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        raise NotImplementedError

    model.train()
    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss

    adv_outs = model(x_adv, A_wave, edges, edge_weights)
    loss_criterion = torch.nn.MSELoss()
    loss = loss_criterion(adv_outs, y)

    if epoch > 1:
        clean_teacher_outs = teacher_model(x_natural, A_wave, edges, edge_weights)
        loss += alpha_reg * loss_criterion(adv_outs, clean_teacher_outs)

    return loss










def ST_pgd_adv_policy_Atten_offine_train(model,
                            x_natural,
                            A_wave,
                            edges,
                            edge_weights,
                            y,
                            optimizer,
                            policynet,
                            device,
                            step_size=0.1,
                            epsilon=0.5,
                            perturb_steps=5,
                            distance='l_inf',
                            rand_start_mode='uniform',
                            rand_start_step=1,
                            **kwargs):
    model.eval()
    """
    baseline_params = dict(model = model,x_natural = x_natural,A_wave = A_wave,
        edges = edges,edge_weights = edge_weights,y= y,step_size= step_size,epsilon= epsilon,K= K,
        Random= True,saliency_steps=1)
    index = rollout_baseline(**baseline_params)
    if is_known_first_node:
        first_node_id = index[0]
    else:
        first_node_id = None
    """
    first_node_id = None
    policynet.eval()



    ll, tours = policynet(x_natural,first_node_id)


    batch_size_x, num_nodes, num_features, steps_length = x_natural.size()
    tours = tours[:,:,None].repeat(1,1,num_features * steps_length)
    mask_matrix = torch.zeros(batch_size_x, num_nodes, num_features * steps_length).to(device)
    mask_matrix = mask_matrix.scatter(dim=1, index=tours.long(), value=torch.tensor(1))
    mask_matrix = mask_matrix.reshape(batch_size_x, num_nodes, num_features, steps_length)



    if rand_start_mode == 'gaussian':
        x_adv = x_natural.detach() + mask_matrix * rand_start_step * 0.001 * torch.randn(
            x_natural.shape).cuda().detach()
    elif rand_start_mode == 'uniform':
        x_adv = x_natural.detach() + mask_matrix * rand_start_step * (epsilon / 10) * torch.rand(
            x_natural.shape).cuda().detach()
    else:
        raise NameError

    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()

            with torch.enable_grad():
                loss_mse = nn.MSELoss()(model(x_adv, A_wave, edges, edge_weights), y)
            grad = torch.autograd.grad(loss_mse, [x_adv])[0]
            x_adv = x_adv.detach() + mask_matrix * step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        raise NotImplementedError

    model.train()
    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss

    outs = model(x_adv, A_wave, edges, edge_weights)
    loss_criterion = torch.nn.MSELoss()
    loss = loss_criterion(outs, y)



    return loss



