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
matplotlib.use('Agg')


client_power = []
accu_power = 0
clients_state = []


state_0 = [0.9449, 0.0087, 0.9913]
state_1 = [0.0551, 0.8509, 0.1491]


def wireless_channel_transition_probability(clients):
    temp = []
    if clients_state == []:
        # print('This is time 0')
        for i in range(clients):
            # print(f'clien stae {i}')
            rand_transision = random.random()
            if rand_transision <= state_0[0]:
                # print(f'random here is {rand_transision}')
                clients_state.append(0)
            else:
                # print(f'random here is {rand_transision}')
                clients_state.append(1)
    else:
        # print('This is Not time 0')
        for i in range((clients)):
            rand_transision = random.random()
            # print(f'random here is {rand_transision}')
            if clients_state[i] == 0:
                if rand_transision <= state_0[1]:
                    clients_state[i] = 1
                else:
                    clients_state[i] = 0
            else:
                if rand_transision <= state_0[2]:
                    clients_state[i] = 0
                else:
                    clients_state[i] = 1


def power(clients):
    clients_power = []
    for i in range(clients):
        rand = random.randint(1, 100)
        clients_power.append(rand)
    return clients_power


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(
        args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    args.gpu = 0  # -1 if no GPU is available

    args.dataset = 'mnist'
    args.num_channels = 1
    args.model = 'cnn'

    args.iid = True     # IID or non-IID
    args.epochs = 1000   # communication round
    args.local_bs = 10  # local batch size
    args.local_ep = 1  # local epoch

    acc_file = open("fedavg_acc.txt", "a")
    loss_file = open("fedavg_loss.txt", "a")
    power_file = open("fedavg_power.txt", "a")

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
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200,
                       dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    # To save or not
    save_reconstructed = 1
    save_original = 1
    client_power = power(args.num_users)
    for iter in range(args.epochs):
        clients_state.clear()
        clients_state = wireless_channel_transition_probability(args.num_users)
        w_locals, loss_locals = [], []
        m = 40
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            if (clients_state[idx] == 1):
                accu_power += client_power[idx]
                continue
            local = LocalUpdate(
                args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))

            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        # update global weights
        w_glob = FedAvg(w_locals)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        # print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)

        # Evaluate score
        net_glob.eval()
        acc_test, loss_test = test_img(net_glob, dataset_test, args)
        print('Round {:3d}, Accuracy {:.3f}, Loss {:.3f}, Accum Power: {:.3f}'.format(
            iter, acc_test, loss_test, accu_power))
        acc_file.write("%f \n" % (acc_test))
        loss_file.write("%f \n" % (loss_test))
        power_file.write("%f \n" % (accu_power))

    # testing
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Training loss: {:.2f}".format(loss_train))
    print("Testing accuracy: {:.2f}".format(acc_test))

    acc_file.close()
    loss_file.close()
    power_file.close()
