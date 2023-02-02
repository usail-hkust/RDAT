import os
import argparse
import shutil
import yaml
from mmcv import Config
import pickle as pk
import numpy as np

import torch
import torch.nn as nn
import time
from utils.data_utils import  mae

from models.astgcn import  ASTGCN
from models.stgcn import  STGCN

from models.GraphWaveNet import gwnet

from models.PolicyAttentionNet import STPolicyAttentionNet

from attacks.other_attacks import  _ST_pgd_whitebox
import copy
from utils.env import get_root_logger, set_random_seed, set_default_configs, \
     init_dist, logger_info
from datasets.datasets import METRLA, HKSPEED,PeMS, PeMSD4
from torch.utils.data import DataLoader
from  datasets.datasets import  DataLoaderX

from methods.train_modes import  plain_train, ST_pgd_adv_policy_Atten_train, ST_pgd_adv_policy_Atten_offine_train, \
ST_pgd_adv_policy_Atten_dist_offline_train

parser = argparse.ArgumentParser(description='STGCN')
parser.add_argument('--enable-cuda', action='store_true',
                    help='Enable CUDA')
parser.add_argument('--gpu', default=6, type=int,
                    help='which gpu to use')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default= 24, metavar='S',
                    help='random seed (default: 24)')
parser.add_argument('--rename', '-r', action='store_true', default=False,
                    help='whether allow renaming the checkpoints parameter to match')
parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'],
                    default='none', help='job launcher')
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--use_gpu',action='store_true',default=True,
                    help='ables cCUDA training')
parser.add_argument('config',
                    default='E:/Project/Robust_STGCN/configs/METRLA/METRLA-train0.6-val0.2-test0.2-standard-stgcn.yaml',
                    help='path to config file')
parser.add_argument('--mode', '-a', default='TRAIN', # ['TRAIN', 'TEST']
                    help='which attack to perform')
parser.add_argument('--device_id',  '-d', default= 2, type=int,# ['TRAIN', 'TEST']
                    help='device ID')
import os


args = parser.parse_args()
set_random_seed(args.seed)

args.device = None
if args.enable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')
# init distributed env first, since logger depends on the dist info.
if args.launcher == 'none':
    distributed = False
    device = torch.device("cuda")
else:
    distributed = True
    init_dist(args.launcher)
    local_rank = torch.distributed.get_rank()
    # torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
print("Using", torch.cuda.device_count(), "GPUs.")
args.device = torch.device('cuda')
print("Using ", args.device)
with open(args.config) as cf:
    cfgs = Config(yaml.safe_load(cf))
if not os.path.exists(cfgs.model_dir):
    os.makedirs(cfgs.model_dir)
torch.backends.cudnn.benchmark = True
torch.cuda.set_device(args.device_id)
shutil.copyfile(args.config, os.path.join(cfgs.model_dir, args.config.split('/')[-1]))
set_default_configs(cfgs)
# setup logger
logger = get_root_logger(cfgs.log_level, cfgs.model_dir)
logger_info(logger, distributed, "Loading config file from {}".format(args.config))
logger_info(logger, distributed, "Models saved at {}".format(cfgs.model_dir))

if cfgs.dataset == 'METRLA':

    train_data = METRLA(mode='train',
                        split_train=cfgs.split_train,
                        split_val=cfgs.split_val,
                        num_timesteps_input=cfgs.num_timesteps_input,
                        num_timesteps_output=cfgs.num_timesteps_output)
    val_data = METRLA(mode='val',
                      split_train=cfgs.split_train,
                      split_val=cfgs.split_val,
                      num_timesteps_input=cfgs.num_timesteps_input,
                      num_timesteps_output=cfgs.num_timesteps_output)
    test_data = METRLA(mode='test',
                       split_train=cfgs.split_train,
                       split_val=cfgs.split_val,
                       num_timesteps_input=cfgs.num_timesteps_input,
                       num_timesteps_output=cfgs.num_timesteps_output)

elif cfgs.dataset == 'HKSPEED':

    train_data = HKSPEED(mode='train',
                        split_train=cfgs.split_train,
                        split_val=cfgs.split_val,
                        num_timesteps_input=cfgs.num_timesteps_input,
                        num_timesteps_output=cfgs.num_timesteps_output)
    val_data = HKSPEED(mode='val',
                      split_train=cfgs.split_train,
                      split_val=cfgs.split_val,
                      num_timesteps_input=cfgs.num_timesteps_input,
                      num_timesteps_output=cfgs.num_timesteps_output)
    test_data = HKSPEED(mode='test',
                       split_train=cfgs.split_train,
                       split_val=cfgs.split_val,
                       num_timesteps_input=cfgs.num_timesteps_input,
                       num_timesteps_output=cfgs.num_timesteps_output)
elif cfgs.dataset == 'PeMS':

    train_data = PeMS(mode='train',
                        split_train=cfgs.split_train,
                        split_val=cfgs.split_val,
                        num_timesteps_input=cfgs.num_timesteps_input,
                        num_timesteps_output=cfgs.num_timesteps_output)
    val_data = PeMS(mode='val',
                      split_train=cfgs.split_train,
                      split_val=cfgs.split_val,
                      num_timesteps_input=cfgs.num_timesteps_input,
                      num_timesteps_output=cfgs.num_timesteps_output)
    test_data = PeMS(mode='test',
                       split_train=cfgs.split_train,
                       split_val=cfgs.split_val,
                       num_timesteps_input=cfgs.num_timesteps_input,
                       num_timesteps_output=cfgs.num_timesteps_output)
elif cfgs.dataset == 'PeMSD4':

    train_data = PeMSD4(mode='train',
                        split_train=cfgs.split_train,
                        split_val=cfgs.split_val,
                        num_timesteps_input=cfgs.num_timesteps_input,
                        num_timesteps_output=cfgs.num_timesteps_output)
    val_data = PeMSD4(mode='val',
                      split_train=cfgs.split_train,
                      split_val=cfgs.split_val,
                      num_timesteps_input=cfgs.num_timesteps_input,
                      num_timesteps_output=cfgs.num_timesteps_output)
    test_data = PeMSD4(mode='test',
                       split_train=cfgs.split_train,
                       split_val=cfgs.split_val,
                       num_timesteps_input=cfgs.num_timesteps_input,
                       num_timesteps_output=cfgs.num_timesteps_output)
else:
    raise  NameError






train_loader = DataLoaderX(train_data, batch_size=cfgs.batch_size, shuffle=True, num_workers =8, pin_memory=True)
val_loader = DataLoaderX(val_data, batch_size=cfgs.batch_size, shuffle=False, num_workers =8, pin_memory=True)
test_loader = DataLoaderX(test_data, batch_size=cfgs.batch_size, shuffle=False)

def train(epoch, logger,train_loader, val_loader, net,optimizer,policynet, policy_optimizer, A_wave, A, loss_criterion, max_speed, edges, edge_weights):
    """
    Trains one epoch with the given data.
    :param training_input: Training inputs of shape (num_samples, num_nodes,
    num_timesteps_train, num_features).
    :param training_target: Training targets of shape (num_samples, num_nodes,
    num_timesteps_predict).
    :param batch_size: Batch size to use during training.
    :return: Average loss for this epoch.
    """
    start = time.time()
    samples_total = len(train_loader) * cfgs.batch_size
    epoch_training_losses = 0
    epoch_training_losses_len = 0
    epoch_rewarding_losses = 0
    epoch_rewarding_losses_len = 0
    save_checkpoint_methods = ["AT_policy_atten_dist_offline"]
    if cfgs.train_mode in save_checkpoint_methods:
        if epoch > 1:
            # load last epoch as the student model
            teacher_model_path = cfgs.model_dir + '/epoch{}.pt'.format(epoch - 1)
            logger.info('Loading teacher model from checkpoint from %s', teacher_model_path)
            teacher_model = copy.deepcopy(net).to(device)
            teacher_model.load_state_dict(torch.load(teacher_model_path))
            teacher_model.eval()
        else:
            teacher_model = None

    for batch_idx, (data, target) in enumerate(train_loader):

        net.train()
        optimizer.zero_grad()


        X_batch, y_batch = data, target
        X_batch = X_batch.to(device=args.device)
        y_batch = y_batch.to(device=args.device)


        loss_params = dict(
            model=net, x_natural=X_batch ,A_wave=A_wave, edges = edges, edge_weights= edge_weights, y=y_batch ,
            optimizer=optimizer,step_size=cfgs.train_step_size, epsilon=cfgs.train_epsilon,
            perturb_steps=cfgs.train_num_steps, distance=cfgs.distance,
            rand_start_mode=cfgs.rand_start_mode,rand_start_step=cfgs.rand_start_step,
            K = int(cfgs.train_attack_nodes * A_wave.size(0)),
            find_type = cfgs.find_type
        )
        if 'plain' == cfgs.train_mode:
            loss = plain_train(**loss_params)
        elif "AT_policy_atten" == cfgs.train_mode:
            extr_params = dict(
                policynet=policynet,
                policy_optimizer = policy_optimizer,
                device = device,
                num_samples = cfgs.num_samples,
                constant = cfgs.constant,
                is_known_first_node = cfgs.is_known_first_node,
                baseline = cfgs.baseline
            )
            loss,reward = ST_pgd_adv_policy_Atten_train(**loss_params,**extr_params)
            epoch_rewarding_losses += reward
            epoch_rewarding_losses_len += 1
            train_reward = epoch_rewarding_losses / epoch_rewarding_losses_len
        elif "AT_policy_atten_dist_offline" == cfgs.train_mode:
            extr_params = dict(
                policynet=policynet,
                device = device,
                alpha_reg = cfgs.alpha_reg,
                teacher_model = teacher_model,
                epoch = epoch
            )
            loss = ST_pgd_adv_policy_Atten_dist_offline_train(**loss_params,**extr_params)

        elif "AT_policy_atten_offine" == cfgs.train_mode:
            extr_params = dict(
                policynet=policynet,
                device = device,
            )
            loss = ST_pgd_adv_policy_Atten_offine_train(**loss_params,**extr_params)
        else:
            raise  NameError

        loss.backward()
        optimizer.step()



        epoch_training_losses += loss.detach().cpu().numpy()
        epoch_training_losses_len += 1
        train_loss = epoch_training_losses / epoch_training_losses_len


        # print progress
        if batch_idx % cfgs.log_interval == 0:
            logger_info(logger, distributed,
                        'Epoch: {} [{}/{} ({:.0f}%)] Train Loss: {:.4f}   time:{:.3f}'.format(
                            epoch, batch_idx * len(data), samples_total,
                                   100. * batch_idx / len(train_loader),
                                   train_loss,
                            time.time() - start))





    if cfgs.train_mode in save_checkpoint_methods:
        torch.save(net.state_dict(),
                   os.path.join(cfgs.model_dir, 'epoch{}.pt'.format(epoch)))
    save_checkpoint_methods_policy = ["AT_policy_atten"]
    if cfgs.train_mode in save_checkpoint_methods_policy:
        torch.save(policynet.state_dict(),
                   os.path.join(cfgs.model_dir, 'policy_epoch{}.pt'.format(epoch)))
    else:
        train_reward = None
    val_loss, val_predict, val_target = eval_val(
                                                  val_loader,
                                                  net, A_wave,A,
                                                  edges,
                                                  edge_weights,
                                                  loss_criterion)
    val_predict, val_target = val_predict * max_speed, val_target * max_speed

    mae_score = mae(val_predict, val_target)
    # print progress
    if batch_idx % cfgs.log_interval == 0:
        logger_info(logger, distributed,
                    'Epoch: {} Train Loss: {:.4f}  Val Loss: {:.4f} MAE: {:.4f}  time:{:.3f}'.format(epoch,
                        train_loss,
                        val_loss,
                        mae_score,
                        time.time() - start))

    return train_loss, val_loss, mae_score, train_reward

def eval_val(val_loader,net, A_wave, A, edges, edge_weights, loss_criterion):
    """
    Trains one epoch with the given data.
    :param training_input: Training inputs of shape (num_samples, num_nodes,
    num_timesteps_train, num_features).
    :param training_target: Training targets of shape (num_samples, num_nodes,
    num_timesteps_predict).
    :param batch_size: Batch size to use during training.
    :return: Average loss for this epoch.
    """

    net.eval()
    with torch.no_grad():
        val_predict = []
        val_target = []
        epoch_val_losses = []
        for batch_idx, (data, target) in enumerate(val_loader):
            X_batch, y_batch = data, target
            X_batch = X_batch.to(device=args.device)
            y_batch = y_batch.to(device=args.device)

            if cfgs.backbone == "RobustGWNET":
                out, _ = net(X_batch, A_wave, edges, edge_weights)
            else:
                out = net(X_batch,A_wave, edges, edge_weights)


            loss = loss_criterion(out, y_batch)
            epoch_val_losses.append(loss.detach().cpu().numpy())
            val_predict.append(out.detach().cpu().numpy())
            val_target.append(y_batch.detach().cpu().numpy())

        val_predict = np.vstack(val_predict)
        val_target = np.vstack(val_target)
        return sum(epoch_val_losses)/len(epoch_val_losses), val_predict, val_target

def eval_val_pgd(val_loader,net, A_wave, A, edges, edge_weights, loss_criterion, max_speed, cfgs):
    """
    Trains one epoch with the given data.
    :param training_input: Training inputs of shape (num_samples, num_nodes,
    num_timesteps_train, num_features).
    :param training_target: Training targets of shape (num_samples, num_nodes,
    num_timesteps_predict).
    :param batch_size: Batch size to use during training.
    :return: Average loss for this epoch.
    """
    train_attack_nodes = int(cfgs.train_attack_nodes * len(A_wave))
    net.eval()
    with torch.no_grad():
        val_predict = []
        val_target = []
        epoch_val_losses = []
        adv_pgd_val_predict = []
        for batch_idx, (data, target) in enumerate(val_loader):
            X_batch, y_batch = data, target
            X_batch = X_batch.to(device=args.device)
            y_batch = y_batch.to(device=args.device)


            out = net(X_batch,A_wave, edges, edge_weights)

            _, X_pgd, index = _ST_pgd_whitebox(net,
                             X_batch,
                             y_batch,
                             A_wave,
                             A,
                             edges,
                             edge_weights,
                             train_attack_nodes,
                             cfgs.train_epsilon,
                             cfgs.train_num_steps,
                             cfgs.random,
                             cfgs.train_step_size,
                             find_type='random')

            adv_pgd_out = net(X_pgd, A_wave, edges, edge_weights)


            loss = loss_criterion(out, y_batch)
            epoch_val_losses.append(loss.detach().cpu().numpy())
            val_predict.append(out.detach().cpu().numpy())
            val_target.append(y_batch.detach().cpu().numpy())
            adv_pgd_val_predict.append(adv_pgd_out.detach().cpu().numpy())

        val_predict = np.vstack(val_predict)
        val_target = np.vstack(val_target)
        adv_pgd_val_predict = np.vstack(adv_pgd_val_predict)

        return sum(epoch_val_losses)/len(epoch_val_losses), val_predict, val_target, adv_pgd_val_predict, index

def main():
    adj = train_data.A
    A = train_data.A.to(device=args.device)
    A_wave = train_data.A_wave.to(device=args.device)

    edges = train_data.edges.to(device=args.device)
    edge_weights = train_data.edge_weights.to(device=args.device)
    max_speed =  train_data.max_speed



    if cfgs.backbone == 'STGCN':
        model = STGCN(A_wave.shape[0],
                    cfgs.num_features,
                    cfgs.num_timesteps_input,
                    cfgs.num_timesteps_output).to(device=args.device)
    elif cfgs.backbone == 'ASTGCN':
        model_params = {
            'nb_time_strides': 1,
            'nb_block': 2,
            'K': 3,
            'nb_chev_filter': 64,
            'nb_time_filter': 64}

        model = ASTGCN(
        nb_block= model_params['nb_block'],
        in_channels = cfgs.num_features,
        K = model_params['K'],
        nb_chev_filter = model_params['nb_chev_filter'],
        nb_time_filter = model_params['nb_time_filter'],
        time_strides = model_params['nb_time_strides'],
        num_for_predict = cfgs.num_timesteps_output,
        len_input = cfgs.num_timesteps_input,
        num_of_vertices = A_wave.shape[0],
        normalization = None,
        bias = True,
        ).to(device=args.device)
    elif cfgs.backbone == 'GWNET':
        dropout = 0.3
        supports = None
        gcn_bool = True
        addaptadj = True
        aptinit = None
        nhid = 32
        model = gwnet(device, num_nodes=cfgs.num_nodes, dropout=dropout, supports=supports, gcn_bool=gcn_bool,
                      addaptadj=addaptadj, aptinit=aptinit, in_dim=cfgs.num_features, out_dim=cfgs.num_timesteps_output,
                      residual_channels=nhid, dilation_channels=nhid, skip_channels=nhid * 8,
                      end_channels=nhid * 16).to(device=args.device)

    else:
        raise NameError

    policy_online_method_lists = ["AT_policy_atten"]
    policy_offine_method_lists = ["AT_policy_atten_offine", "AT_policy_atten_dist_offline"]

    if cfgs.train_mode in policy_online_method_lists:
        dropout = 0.3
        supports = None
        gcn_bool = True
        addaptadj = True
        aptinit = None
        nhid = 4
        victim_nodes = int(cfgs.train_attack_nodes * len(A_wave))
        policynet = STPolicyAttentionNet(victim_nodes = victim_nodes,device = device, num_nodes=cfgs.num_nodes, dropout=dropout, supports=supports, gcn_bool=gcn_bool,
                      addaptadj=addaptadj, aptinit=aptinit, in_dim=cfgs.num_features, out_dim=cfgs.hidden_embedding_dims,
                      residual_channels=nhid, dilation_channels=nhid, skip_channels=nhid * 8,
                      end_channels=nhid * 16).to(device=args.device)
        policy_optimizer = torch.optim.Adam(policynet.parameters(), lr=1e-3)
    elif cfgs.train_mode in policy_offine_method_lists:
        dropout = 0.3
        supports = None
        gcn_bool = True
        addaptadj = True
        aptinit = None
        nhid = 4
        victim_nodes = int(cfgs.train_attack_nodes * len(A_wave))
        policynet = STPolicyAttentionNet(victim_nodes=victim_nodes, device=device, num_nodes=cfgs.num_nodes,
                                         dropout=dropout, supports=supports, gcn_bool=gcn_bool,
                                         addaptadj=addaptadj, aptinit=aptinit, in_dim=cfgs.num_features,
                                         out_dim=cfgs.hidden_embedding_dims,
                                         residual_channels=nhid, dilation_channels=nhid, skip_channels=nhid * 8,
                                         end_channels=nhid * 16).to(device=args.device)
        policy_optimizer = None
        load_path = cfgs.policynet_path
        logger.info('Loading checkpoint from %s for policynet', load_path)
        policynet.load_state_dict(torch.load(load_path))

    else:

        policynet, policy_optimizer = None, None

    if args.mode == "TRAIN":
        load_path = None
        if cfgs.load_model is not None:
            load_path = cfgs.load_model

        elif cfgs.resume_epoch > 0:
            load_path = os.path.join(cfgs.model_dir, 'epoch{}.pt'.format(cfgs.resume_epoch))
        if load_path is not None:
            assert os.path.exists(load_path), load_path
            logger.info('Loading checkpoint from %s', load_path)
            model.load_state_dict(torch.load(load_path))






        # init loss function, optimizer
        if cfgs.loss_func == 'mae':
            loss_criterion = torch.nn.L1Loss().to(args.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.03, eps=1.0e-8,
                                 weight_decay=0, amsgrad=False)
        elif cfgs.loss_func == 'mse':
            loss_criterion = torch.nn.MSELoss().to(args.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        else:
            raise ValueError




        training_losses = []
        validation_losses = []
        validation_maes = []
        rewarding = []
        start_epoch = cfgs.resume_epoch + 1
        wait = 0
        val_loss_min = np.inf
        total_start = time.time()
        #print('========================start to training===========================')
        for epoch in range(start_epoch, cfgs.epochs + 1):

            if cfgs.dropout_type == 'none_dropout':
                A_wave, edges, edge_weights  = A_wave, edges, edge_weights

            else:
                raise  NameError


            if wait >= cfgs.patience and epoch >= cfgs.minimum_epoch:
                logger.info('early stop at epoch: %04d' % (epoch))
                torch.save(model.state_dict(),
                           os.path.join(cfgs.model_dir, 'epoch_last.pt'))
                torch.save(model.state_dict(),
                           os.path.join(cfgs.model_dir, 'epoch{}.pt'.format(cfgs.minimum_epoch)))
                break
            if  epoch == (cfgs.minimum_epoch -1):
                logger.info('save model at epoch: %04d' % (epoch))
                torch.save(model.state_dict(),
                           os.path.join(cfgs.model_dir, 'epoch{}.pt'.format(cfgs.minimum_epoch-1)))

            train_loss, val_loss, mae_score,train_reward = train(epoch, logger,train_loader,val_loader, model,optimizer, policynet, policy_optimizer,A_wave, A, loss_criterion, max_speed, edges, edge_weights)
            training_losses.append(train_loss)
            validation_losses.append(val_loss)
            validation_maes.append(mae_score)


            if val_loss <= val_loss_min:
                logger.info(
                    'val loss decrease from %.4f to %.4f, saving model to %s ' %
                    (val_loss_min, val_loss, cfgs.model_dir))
                wait = 0
                val_loss_min = val_loss
                torch.save(model.state_dict(),
                           os.path.join(cfgs.model_dir, 'epoch{}.pt'.format(epoch)))
                torch.save(model.state_dict(),
                           os.path.join(cfgs.model_dir, 'best_epoch.pt'))
                logger.info('best model at epoch: %04d' % (epoch))
            else:
                wait += 1

            with open(cfgs.model_dir+"/losses.pk", "wb") as fd:
                pk.dump((training_losses, validation_losses, validation_maes), fd)

        logger_info(logger, distributed, 'Val Clean MAE: {:.4f}'.format(validation_maes[-1]))
        logger_info(logger, distributed, '[Remarks] {} | End of training, saved at {}'.format(cfgs.remark, cfgs.model_dir))
        logger_info(logger, distributed, 'Total training time:{:.3f}'.format(time.time() - total_start))
    elif args.mode == 'TEST':
        if cfgs.loss_func == 'mae':
            loss_criterion = torch.nn.L1Loss().to(args.device)
        elif cfgs.loss_func == 'mse':
            loss_criterion = torch.nn.MSELoss().to(args.device)
        else:
            raise ValueError
        load_path = cfgs.model_path
        logger.info('Loading checkpoint from %s', load_path)
        model.load_state_dict(torch.load(load_path))

        model.eval()
        _, test_clean_predict, test_target, test_adv_predict, index = eval_val_pgd(test_loader, model, A_wave, A, edges,
                                                              edge_weights, loss_criterion, max_speed,
                                                              cfgs)

        test_adv_predict, test_target = test_adv_predict * max_speed, test_target * max_speed
        test_adv_mae_score = mae(test_adv_predict, test_target)

        test_clean_predict = test_clean_predict * max_speed
        test_clean_mae_score = mae(test_clean_predict, test_target)

        logger_info(logger, distributed,
                    'Test Clean MAE: {:.4f}  Test Adv MAE:{:.4f}'.format(test_clean_mae_score, test_adv_mae_score))
        logger_info(logger, distributed,
                    '[Remarks] {} | End of training, saved at {}'.format(cfgs.remark, cfgs.model_dir))
if __name__ == '__main__':
   main()
