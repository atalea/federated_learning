#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from models.test import test_img
from models.Fed import FedAvg
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Update import LocalUpdate
from utils.options import args_parser
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
import torch
from torchvision import datasets, transforms
import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib
import random
import heapq
import os
matplotlib.use('Agg')


# client_power = [73, 61, 64, 81, 64, 81, 68, 76, 62, 68, 64, 79, 79, 71, 63, 72, 66, 76, 66, 68, 71, 59, 63, 74, 71, 80, 77, 79, 67, 78, 71, 62, 65, 67, 63, 68, 74, 61, 78, 76, 66, 70, 70, 74, 65, 72, 73, 79,
#                 75, 80, 67, 63, 64, 79, 69, 70, 73, 64, 78, 80, 68, 74, 71, 77, 72, 67, 74, 78, 78, 71, 73, 78, 73, 60, 76, 76, 77, 79, 71, 63, 64, 70, 75, 62, 65, 61, 63, 73, 66, 67, 78, 63, 70, 67, 71, 68, 74, 60, 62, 72]
accu_power_fedavg = 0
accu_power_ibcs = 0
accu_power_pp = 0
clients_state = []
# successful, unsuccessful = 1, 0
p00 = 0.8509
p10 = 0.0087
p11 = 0.9913
p01 = 0.1491

state_0 = [0.9449, 0.0087, 0.9913]
state_1 = [0.0551, 0.8509, 0.1491]

# state_0 = [0.9449, 0.5, 0.5]
# state_1 = [0.0551, 0.5, 0.5]
# state_0 = [0.9449, 0.8, 0.2]
# state_1 = [0.0551, 0.2, 0.8]

top_k = 20
lam = 0.1
print('States are:', '\n',  state_0, '\n', state_1,
      '\n', 'K= ', top_k, '\n', 'Lambda= ', lam)


def initilization(clients):
    for i in range(clients):
        # print(f'clien stae {i}')
        rand_transision = random.random()
        if rand_transision <= 0.9449:  # state_0[0]:
            # print(f'random here is {rand_transision}')
            clients_state.append(1)
        else:
            # print(f'random here is {rand_transision}')
            clients_state.append(0)


def wireless_channel_transition_probability(clients):
    for i in range((clients)):
        rand_transision = random.random()
        # print(f'random here is {rand_transision}')
        if clients_state[i] == 0:
            if rand_transision <= p01:  # state_0[1]:
                clients_state[i] = 1
            else:
                clients_state[i] = 0
        else:
            if rand_transision <= p10:  # state_0[2]:
                clients_state[i] = 0
            else:
                clients_state[i] = 1


def power(clients):
    clients_power = []
    for i in range(clients):
        rand = random.randint(60, 80)
        clients_power.append(rand)
    return clients_power


def clients_indexing(clients, clients_power):
    # p_11 --> state_1[1]
    # p_10 --> state_1[2]
    # p_01 --> state_0[1]
    # p_00 --> state_0[2]
    user_indices = []
    for i in range((clients)):
        if clients_state[i] == 1:
            v_i_t = -(p11/(clients)) - \
                (lam*((p10*clients_power[i])/100))
            user_indices.append(v_i_t)
            # print(f'client {clients[i]}, is in state {clients_state[i]}')
        elif clients_state[i] == 0:
            v_i_t = -(p01/(clients)) - \
                (lam*((p00*clients_power[i])/100))
            user_indices.append(v_i_t)
    # print('Indices are', user_indices)
    # this prints the top k values
    # top_k_users = heapq.nlargest(top_k, user_indices)
    # This gives the top k user indices
    user_indices = np.argsort(user_indices)
    top_k_users = user_indices[-top_k:]
    return top_k_users


def user_previous_prob(clients):
    # previous state (0, 1) --> 0 un, 1 succ
    succ_probabilities = []
    for i in range(clients):
        if clients_state[i] == 0:
            succ_probabilities.append(p01)
        else:
            succ_probabilities.append(p11)
    # This gives the top k user indices
    succ_probabilities = np.argsort(succ_probabilities)
    top_k_users = succ_probabilities[-top_k:]
    return top_k_users


if __name__ == '__main__':
    # parse args
    args = args_parser()

    args.device = torch.cuda.is_available()
    if args.device == False:
        args.gpu = -1  # 0  # -1 if no GPU is available
        print('No GPU found')
        print('\n')
    else:
        args.gpu = 0  # 0  # -1 if no GPU is available
        print('GPU is on')

    args.device = torch.device('cuda:{}'.format(
        args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    args.dataset = 'mnist'
    args.num_channels = 1
    args.model = 'cnn'

    args.iid = True     # IID or non-IID
    args.epochs = 1000   # communication round
    args.local_bs = 32  # local batch size
    args.local_ep = 10  # local epoch

    if os.path.exists(f"./mnist_results/fedavg_acc_lam_{lam}_k_{top_k}.txt"):

        os.remove(f"./mnist_results/fedavg_acc_lam_{lam}_k_{top_k}.txt")
        os.remove(f"./mnist_results/fedavg_loss_lam_{lam}_k_{top_k}.txt")
        os.remove(f"./mnist_results/fedavg_power_lam_{lam}_k_{top_k}.txt")
        os.remove(f"./mnist_results/ibcs_acc_lam_{lam}_k_{top_k}.txt")
        os.remove(f"./mnist_results/ibcs_loss_lam_{lam}_k_{top_k}.txt")
        os.remove(f"./mnist_results/ibcs_power_lam_{lam}_k_{top_k}.txt")
        os.remove(f"./mnist_results/pp_acc_lam_{lam}_k_{top_k}.txt")
        os.remove(f"./mnist_results/pp_loss_lam_{lam}_k_{top_k}.txt")
        os.remove(f"./mnist_results/pp_power_lam_{lam}_k_{top_k}.txt")

    acc_file_fedavg = open(
        f"./mnist_results/fedavg_acc_lam_{lam}_k_{top_k}.txt", "a")
    loss_file_fedavg = open(
        f"./mnist_results/fedavg_loss_lam_{lam}_k_{top_k}.txt", "a")
    power_file_fedavg = open(
        f"./mnist_results/fedavg_power_lam_{lam}_k_{top_k}.txt", "a")

    acc_file_ibcs = open(
        f"./mnist_results/ibcs_acc_lam_{lam}_k_{top_k}.txt", "a")
    loss_file_ibcs = open(
        f"./mnist_results/ibcs_loss_lam_{lam}_k_{top_k}.txt", "a")
    power_file_ibcs = open(
        f"./mnist_results/ibcs_power_lam_{lam}_k_{top_k}.txt", "a")

    acc_file_pp = open(
        f"./mnist_results/pp_acc_lam_{lam}_k_{top_k}.txt", "a")
    loss_file_pp = open(
        f"./mnist_results/pp_loss_lam_{lam}_k_{top_k}.txt", "a")
    power_file_pp = open(
        f"./mnist_results/pp_power_lam_{lam}_k_{top_k}.txt", "a")

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST(
            '../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST(
            '../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10(
            '../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10(
            '../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob_fedavg = CNNCifar(args=args).to(args.device)
        net_glob_ibcs = CNNCifar(args=args).to(args.device)
        net_glob_pp = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob_fedavg = CNNMnist(args=args).to(args.device)
        net_glob_ibcs = CNNMnist(args=args).to(args.device)
        net_glob_pp = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob_fedavg = MLP(dim_in=len_in, dim_hidden=200,
                              dim_out=args.num_classes).to(args.device)
        net_glob_ibcs = MLP(dim_in=len_in, dim_hidden=200,
                            dim_out=args.num_classes).to(args.device)
        net_glob_pp = MLP(dim_in=len_in, dim_hidden=200,
                          dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    # print(net_glob_fedavg)
    net_glob_fedavg.train()
    net_glob_ibcs.train()
    net_glob_pp.train()

    # copy weights
    w_glob_fedavg = net_glob_fedavg.state_dict()
    w_glob_ibcs = net_glob_ibcs.state_dict()
    w_glob_pp = net_glob_pp.state_dict()

    # training
    # loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    # To save or not
    save_reconstructed = 1
    save_original = 1
    # clients_state.clear()
    client_power = power(args.num_users)
    # print(client_power)
    initilization(args.num_users)
    print(f'Numper of global epochs = {args.epochs}')
    print(f'Number of local epochs = {args.local_ep}')
    print(f'Batch size = {args.local_bs}')

    for iter in range(args.epochs):
        # clients_state.clear()
        w_locals, loss_locals = [], []
        idxs_users_fedavg = np.random.choice(
            range(args.num_users), top_k, replace=False)
        idxs_users_ibcs = clients_indexing(args.num_users, client_power)
        idxs_users_pp = user_previous_prob(args.num_users)
        wireless_channel_transition_probability(args.num_users)
# ************************************************************************************************#
        # FedAvg #

        for idx in idxs_users_fedavg:
            if (clients_state[idx] == 0):
                accu_power_fedavg += client_power[idx]
                continue
            local = LocalUpdate(
                args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(
                net_glob_fedavg).to(args.device))

            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        # update global weights
        w_glob_fedavg = FedAvg(w_locals)

        # copy weight to net_glob_fedavg
        net_glob_fedavg.load_state_dict(w_glob_fedavg)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        # print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        # loss_train.append(loss_avg)

        # Evaluate score
        net_glob_fedavg.eval()
        acc_test, loss_test = test_img(net_glob_fedavg, dataset_test, args)
        print('Round {:3d} fedavg, Accuracy {:.3f}, Loss {:.3f}, Accum Power: {:.3f}'.format(
            iter, acc_test, loss_test, accu_power_fedavg))
        acc_file_fedavg.write("%f \n" % (acc_test))
        loss_file_fedavg.write("%f \n" % (loss_test))
        power_file_fedavg.write("%f \n" % (accu_power_fedavg))

# ************************************************************************************************#
        # ibcs #
        w_locals, loss_locals = [], []
        # idxs_users_ibcs = clients_indexing(args.num_users, client_power)
        for idx in idxs_users_ibcs:
            if (clients_state[idx] == 0):
                accu_power_ibcs += client_power[idx]
                continue
            local = LocalUpdate(
                args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(
                net_glob_ibcs).to(args.device))

            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        # update global weights
        w_glob_ibcs = FedAvg(w_locals)
        # print(len(w_glob_ibcs))

        # copy weight to net_glob_fedavg
        net_glob_ibcs.load_state_dict(w_glob_ibcs)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        # print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        # loss_train.append(loss_avg)

        # Evaluate score
        net_glob_ibcs.eval()
        acc_test, loss_test = test_img(net_glob_ibcs, dataset_test, args)
        print('Round {:3d} ibcs, Accuracy {:.3f}, Loss {:.3f}, Accum Power: {:.3f}'.format(
            iter, acc_test, loss_test, accu_power_ibcs))
        acc_file_ibcs.write("%f \n" % (acc_test))
        loss_file_ibcs.write("%f \n" % (loss_test))
        power_file_ibcs.write("%f \n" % (accu_power_ibcs))
        # wireless_channel_transition_probability(args.num_users)


# ************************************************************************************************#
        # pp #

        w_locals, loss_locals = [], []
        # idxs_users_ibcs = clients_indexing(args.num_users, client_power)
        for idx in idxs_users_pp:
            if (clients_state[idx] == 0):
                accu_power_pp += client_power[idx]
                continue
            local = LocalUpdate(
                args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(
                net_glob_pp).to(args.device))

            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        # update global weights
        w_glob_pp = FedAvg(w_locals)
        # print(len(w_glob_ibcs))

        # copy weight to net_glob_fedavg
        net_glob_pp.load_state_dict(w_glob_pp)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        # print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        # loss_train.append(loss_avg)

        # Evaluate score
        net_glob_pp.eval()
        acc_test, loss_test = test_img(net_glob_pp, dataset_test, args)
        print('Round {:3d} pp, Accuracy {:.3f}, Loss {:.3f}, Accum Power: {:.3f}'.format(
            iter, acc_test, loss_test, accu_power_pp))
        acc_file_pp.write("%f \n" % (acc_test))
        loss_file_pp.write("%f \n" % (loss_test))
        power_file_pp.write("%f \n" % (accu_power_pp))
        # wireless_channel_transition_probability(args.num_users)


#######################################################################################

    # testing
    net_glob_fedavg.eval()
    acc_train, loss_train = test_img(net_glob_fedavg, dataset_train, args)
    acc_test, loss_test = test_img(net_glob_fedavg, dataset_test, args)
    print("FedAvg Training accuracy: {:.2f}".format(acc_train))
    print("FedAvg Training loss: {:.2f}".format(loss_train))
    print("FedAvg Testing accuracy: {:.2f}".format(acc_test))

    # testing
    net_glob_ibcs.eval()
    acc_train, loss_train = test_img(net_glob_ibcs, dataset_train, args)
    acc_test, loss_test = test_img(net_glob_ibcs, dataset_test, args)
    print("IBCS Training accuracy: {:.2f}".format(acc_train))
    print("IBCS Training loss: {:.2f}".format(loss_train))
    print("IBCS Testing accuracy: {:.2f}".format(acc_test))

    # testing
    net_glob_pp.eval()
    acc_train, loss_train = test_img(net_glob_pp, dataset_train, args)
    acc_test, loss_test = test_img(net_glob_pp, dataset_test, args)
    print("IBCS Training accuracy: {:.2f}".format(acc_train))
    print("IBCS Training loss: {:.2f}".format(loss_train))
    print("IBCS Testing accuracy: {:.2f}".format(acc_test))

    acc_file_fedavg.close()
    loss_file_fedavg.close()
    power_file_fedavg.close()
    acc_file_ibcs.close()
    loss_file_ibcs.close()
    power_file_ibcs.close()
    acc_file_pp.close()
    loss_file_pp.close()
    power_file_pp.close()
