import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset

import torchvision.datasets as datasets
import torchvision
import torchvision.transforms as transforms

# general
import pandas as pd 
import numpy as np 
import copy
import pickle
import sys
import time
import os
import random

from helper import *
from utility_func import *

import config




def get_ufunc(dataset, model_type, batch_size, lr, verbose, return_net=False, device='cpu'):
    if dataset in ['MNIST', 'FMNIST']:
        u_func = lambda a, b, c, d: torch_mnist_data_to_acc(model_type, a, b, c, d, batch_size=batch_size, lr=lr, verbose=verbose)
    elif dataset == 'CIFAR10':
        u_func = lambda a, b, c, d: torch_cifar_data_to_acc(model_type, a, b, c, d, batch_size=batch_size, lr=lr, verbose=verbose)
    elif dataset == 'Dog_vs_Cat':
        u_func = lambda a, b, c, d: torch_dogcat_data_to_acc(model_type, a, b, c, d, batch_size=batch_size, lr=lr, verbose=verbose)
    elif dataset == 'Dog_vs_CatFeature':
        u_func = lambda a, b, c, d: torch_dogcatFeature_data_to_acc(model_type, a, b, c, d, batch_size=batch_size, lr=lr, verbose=verbose)
    elif dataset == 'FMNIST':
        sys.exit(1)
    elif dataset in ['covertype']:
        u_func = lambda a, b, c, d: binary_data_to_acc(model_type, a, b, c, d)
    elif dataset in config.OpenML_dataset:
        #u_func = lambda a, b, c, d: torch_binary_data_to_acc(model_type, a, b, c, d, batch_size=batch_size, lr=lr, verbose=verbose)
        def u_func(a, b, c, d):
            return torch_binary_data_to_acc(model_type, a, b, c, d, batch_size=batch_size, lr=lr, verbose=verbose, return_net=return_net, device=device)

    return u_func



def get_weighted_ufunc(dataset, model_type, batch_size, lr, verbose):
    if dataset in ['MNIST', 'FMNIST']:
        u_func = lambda a, b, c, d, w: torch_mnist_data_to_acc(model_type, a, b, c, d, weights=w, batch_size=batch_size, lr=lr, verbose=verbose)
    elif dataset == 'CIFAR10':
        u_func = lambda a, b, c, d, w: torch_cifar_data_to_acc(model_type, a, b, c, d, weights=w, batch_size=batch_size, lr=lr, verbose=verbose)
    elif dataset == 'Dog_vs_Cat':
        u_func = lambda a, b, c, d, w: torch_dogcat_data_to_acc(model_type, a, b, c, d, weights=w, batch_size=batch_size, lr=lr, verbose=verbose)
    elif dataset == 'Dog_vs_CatFeature':
        u_func = lambda a, b, c, d, w: torch_dogcatFeature_data_to_acc(model_type, a, b, c, d, weights=w, batch_size=batch_size, lr=lr, verbose=verbose)
    elif dataset == 'FMNIST':
        sys.exit(1)
    elif dataset in ['covertype']:
        u_func = lambda a, b, c, d, w: binary_data_to_acc(model_type, a, b, c, d, w=w)
    elif dataset in config.OpenML_dataset:
        u_func = lambda a, b, c, d, w: torch_binary_data_to_acc(model_type, a, b, c, d, weights=w, batch_size=batch_size, lr=lr, verbose=verbose)
    return u_func



def make_balance_sample_multiclass(data, target, n_data):

    n_class = len(np.unique(target))

    n_data_per_class = int(n_data / n_class)

    selected_ind = np.array([])

    for i in range(n_class):

        index_class = np.where(target == i)[0]
        # print(len(index_class))
        # print(n_data_per_class)

        ind = np.random.choice(index_class, size=n_data_per_class, replace=False)

        selected_ind = np.concatenate([selected_ind, ind])

    selected_ind = selected_ind.astype(int)

    data, target = data[selected_ind], target[selected_ind]

    assert n_data == len(target)

    idxs=np.random.permutation(n_data)
    data, target=data[idxs], target[idxs]

    return data, target

def make_non_overlapping_sample(x_test, y_test, x_val, y_val, n_val):
    # 将x_val中的元素转换为字符串形式，以便创建一个集合
    val_data = set([str(x) for x in x_val])

    # 找出x_test中不在x_val中的元素的索引
    non_overlapping_indices = [i for i, x in enumerate(x_test) if str(x) not in val_data]

    # 确保y_test中的元素与x_test中的元素对应
    assert len(x_test) == len(y_test)

    # 从这些索引中随机选择n_val个元素
    selected_indices = np.random.choice(non_overlapping_indices, size=n_val, replace=False)

    # 使用选择的索引来创建x_val2和y_val2
    x_val2, y_val2 = x_test[selected_indices], y_test[selected_indices]

    return x_val2, y_val2

def get_processed_data(dataset, n_data, n_val, flip_ratio):
    
    print('-------')
    print('Load Dataset {}'.format(dataset))

    np.random.seed(999)
    if dataset in config.OpenML_dataset:
        X, y, _, _ = get_data(dataset)
        print(y.shape)
        #print(_.shape)

        x_train, y_train = X[:n_data], y[:n_data]
        x_val, y_val = X[n_data:n_data+n_val], y[n_data:n_data+n_val]

        X_mean, X_std= np.mean(x_train, 0), np.std(x_train, 0)
        normalizer_fn = lambda x: (x - X_mean) / np.clip(X_std, 1e-12, None)
        x_train, x_val = normalizer_fn(x_train), normalizer_fn(x_val)

        #y_train[4245] = 1 - y_train[4245] Flip

    else:
        X_train, Y_train, x_test, y_test = get_data(dataset)
        # x_val, y_val = x_test[:n_val], y_test[:n_val]
        # print(x_train.shape)

        if dataset not in ['covertype', 'SST-2','20news']:
            x_train, y_train = make_balance_sample_multiclass(X_train, Y_train, n_data)
            x_val, y_val = make_balance_sample_multiclass(x_test, y_test, n_val)

            #x_train, y_train = x_train[:n_data], y_train[:n_data]
            #x_val, y_val = x_test[1:], y_test[1:]
            # if x_val.shape[0] < n_val:
            #     x_val2, y_val2 = make_non_overlapping_sample(X_train, Y_train, x_train, y_train, n_val)
            # else:
            #     x_val2, y_val2 = make_non_overlapping_sample(x_test, y_test, x_val, y_val, n_val)
            #print(x_train.shape)
        else:
            x_train, y_train = X_train[:n_data], Y_train[:n_data]
            x_val, y_val = x_test[:n_val], y_test[:n_val]
            # if x_val.shape[0] < n_val:
            #     x_val2, y_val2 = X_train[n_data:], Y_train[n_data:]
            # else:
            #     x_val2, y_val2 = x_test[n_val:n_val+n_val], y_test[n_val:n_val+n_val]
            # print(x_val.shape)

    np.random.seed(999)
    n_flip = int(n_data*flip_ratio)

    assert len(y_train.shape)==1
    n_class = len(np.unique(y_train))
    print('# of classes = {}'.format(n_class))
    print('-------')

    if n_class == 2:
        y_train[:n_flip] = 1 - y_train[:n_flip]
    else:
        y_train[:n_flip] = np.array( [ np.random.choice( np.setdiff1d(np.arange(n_class), [y_train[i]]) ) for i in range(n_flip) ] )

    return x_train, y_train, x_val, y_val#, x_val2, y_val2



def get_processed_data_new(dataset, n_data, n_val, flip_ratio):
    
    print('-------')
    print('Load Dataset {}'.format(dataset))

    if dataset in config.OpenML_dataset:
        X, y, _, _ = get_data(dataset)
        x_train, y_train = X[:n_data], y[:n_data]
        x_val, y_val = X[n_data:n_data+n_val], y[n_data:n_data+n_val]
        #x_val, y_val = X[n_data+n_val:n_data+n_val+n_val], y[n_data+n_val:n_data+n_val+n_val]
        #x_val, y_val = X[n_data+n_val+n_val:n_data+n_val+n_val+n_val], y[n_data+n_val+n_val:n_data+n_val+n_val+n_val]
        print(X.shape)
        print(x_train.shape)
        print(x_val.shape)

        X_mean, X_std= np.mean(x_train, 0), np.std(x_train, 0)
        normalizer_fn = lambda x: (x - X_mean) / np.clip(X_std, 1e-12, None)
        x_train, x_val = normalizer_fn(x_train), normalizer_fn(x_val)

    else:
        x_train, y_train, x_val, y_val, x_test, y_test = get_data(dataset)
        #x_val, y_val = x_test, y_test

        if dataset != 'covertype':
            x_train, y_train = make_balance_sample_multiclass(x_train, y_train, n_data)
            x_val, y_val = make_balance_sample_multiclass(x_val, y_val, n_val)
            x_test, y_test = make_balance_sample_multiclass(x_test, y_test, n_val)
            #print(x_train.type())
            #x_train, y_train, x_val, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
            '''
            x_train_val, y_train_val = make_balance_sample_multiclass(x_train, y_train, n_data+n_val)
            x_train, y_train = make_balance_sample_multiclass(x_train_val, y_train_val, n_data)
            mask_x = torch.ones(x_train_val.shape, dtype=torch.bool)
            for val in x_train:
                mask_x &= (x_train_val != val)
            x_val = torch.masked_select(torch.from_numpy(x_train_val), mask_x).numpy()
            mask_y = torch.ones(y_train_val.shape, dtype=torch.bool)
            for val in y_train:
                mask_y &= (y_train_val != val)
            y_val = torch.masked_select(torch.from_numpy(y_train_val), mask_y).numpy()
            #print(x_train.type())'''
            

    np.random.seed(999)
    n_flip = int(n_data*flip_ratio)

    assert len(y_train.shape)==1
    n_class = len(np.unique(y_train))
    print('# of classes = {}'.format(n_class))
    print('-------')

    if n_class == 2:
        y_train[:n_flip] = 1 - y_train[:n_flip]
    else:
        y_train[:n_flip] = np.array( [ np.random.choice( np.setdiff1d(np.arange(n_class), [y_train[i]]) ) for i in range(n_flip) ] )

    return x_train, y_train, x_val, y_val, x_test, y_test




def get_data(dataset):

    if dataset in ['covertype']+config.OpenML_dataset:
        x_train, y_train, x_test, y_test = get_minidata(dataset)
    elif dataset == 'MNIST':
        x_train, y_train, x_test, y_test = get_mnist()
    elif dataset == 'CIFAR10':
        x_train, y_train, x_test, y_test = get_dogcatFeature()
    elif dataset == 'FMNIST':
        x_train, y_train, x_test, y_test = get_fmnist()
    elif dataset == 'AG_NEWS':
        x_train, y_train, x_test, y_test = get_ag_newsFeature()
    elif dataset == 'DBPedia':
        x_train, y_train, x_test, y_test = get_DBPediaFeature()
    elif dataset == 'IMDb':
        x_train, y_train, x_test, y_test = get_IMDbFeature()
    elif dataset == 'SST-2':
        x_train, y_train, x_test, y_test = get_SSTFeature()
    elif dataset == '20news':
        x_train, y_train, x_test, y_test = get_20newsgroups_feature()
    else:
        sys.exit(1)

    return x_train, y_train, x_test, y_test
'''

def get_data(dataset):

    if dataset in ['covertype']+config.OpenML_dataset:
        x_train, y_train, x_test, y_test = get_minidata(dataset)
    elif dataset == 'MNIST':
        x_train, y_train, x_val, y_val, x_test, y_test = get_mnist()
    elif dataset == 'CIFAR10':
        x_train, y_train, x_test, y_test = get_dogcatFeature()
    elif dataset == 'FMNIST':
        x_train, y_train, x_test, y_test = get_fmnist()
    else:
        sys.exit(1)

    #return x_train, y_train, x_test, y_test
    return x_train, y_train, x_val, y_val, x_test, y_test
'''


def make_balance_sample(data, target):
    p = np.mean(target)
    if p < 0.5:
        minor_class=1
    else:
        minor_class=0
    
    index_minor_class = np.where(target == minor_class)[0]
    n_minor_class=len(index_minor_class)
    n_major_class=len(target)-n_minor_class
    new_minor=np.random.choice(index_minor_class, size=n_major_class-n_minor_class, replace=True)

    data=np.concatenate([data, data[new_minor]])
    target=np.concatenate([target, target[new_minor]])
    return data, target



def get_minidata(dataset):

    open_ml_path = 'OpenML_datasets/'

    np.random.seed(999)

    if dataset == 'covertype':
        x_train, y_train, x_test, y_test = pickle.load( open('covertype_200.dataset', 'rb') )

    elif dataset == 'fraud':
        data_dict=pickle.load(open(open_ml_path+'CreditCardFraudDetection_42397.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y'] # data[:,1:], data[:,0]
        target = (target == 1) + 0.0
        target = target.astype(np.int32)
        data, target=make_balance_sample(data, target)

    elif dataset == 'apsfail':
        # ((10000, 28), array([0., 1.], dtype=float32))
        data_dict=pickle.load(open(open_ml_path+'APSFailure_41138.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y'] # data[:,1:], data[:,0]
        target = (target == 1) + 0.0
        target = target.astype(np.int32)
        data, target=make_balance_sample(data, target)

    elif dataset == 'click':
        # ((10000, 28), array([0., 1.], dtype=float32))
        data_dict=pickle.load(open(open_ml_path+'Click_prediction_small_1218.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y'] # data[:,1:], data[:,0]
        target = (target == 1) + 0.0
        target = target.astype(np.int32)
        data, target=make_balance_sample(data, target)

    elif dataset == 'phoneme':
        # ((10000, 28), array([0., 1.], dtype=float32))
        data_dict=pickle.load(open(open_ml_path+'phoneme_1489.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y'] # data[:,1:], data[:,0]
        target = (target == 1) + 0.0
        target = target.astype(np.int32)
        data, target=make_balance_sample(data, target)

    elif dataset == 'wind':
        # ((10000, 28), array([0., 1.], dtype=float32))
        data_dict=pickle.load(open(open_ml_path+'wind_847.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y'] # data[:,1:], data[:,0]
        target = (target == 1) + 0.0
        target = target.astype(np.int32)
        data, target=make_balance_sample(data, target)

    elif dataset == 'pol':
        # ((10000, 28), array([0., 1.], dtype=float32))
        data_dict=pickle.load(open(open_ml_path+'pol_722.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y'] # data[:,1:], data[:,0]
        target = (target == 1) + 0.0
        target = target.astype(np.int32)
        data, target=make_balance_sample(data, target)

    elif dataset == 'creditcard':
        # ((10000, 28), array([0., 1.], dtype=float32))
        data_dict=pickle.load(open(open_ml_path+'default-of-credit-card-clients_42477.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y'] # data[:,1:], data[:,0]
        target = (target == 1) + 0.0
        target = target.astype(np.int32)
        data, target=make_balance_sample(data, target)

    elif dataset == 'cpu':
        # ((10000, 28), array([0., 1.], dtype=float32))
        data_dict=pickle.load(open(open_ml_path+'cpu_act_761.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y'] # data[:,1:], data[:,0]
        target = (target == 1) + 0.0
        target = target.astype(np.int32)
        data, target=make_balance_sample(data, target)

    elif dataset == 'vehicle':
        # ((10000, 28), array([0., 1.], dtype=float32))
        data_dict=pickle.load(open(open_ml_path+'vehicle_sensIT_357.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y'] # data[:,1:], data[:,0]
        target = (target == 1) + 0.0
        target = target.astype(np.int32)
        data, target=make_balance_sample(data, target)

    elif dataset == '2dplanes':
        # ((10000, 28), array([0., 1.], dtype=float32))
        data_dict=pickle.load(open(open_ml_path+'2dplanes_727.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y'] # data[:,1:], data[:,0]
        target = (target == 1) + 0.0
        target = target.astype(np.int32)
        data, target=make_balance_sample(data, target)

    else:
        print('No such dataset!')
        sys.exit(1)


    if dataset not in ['covertype']:
        idxs=np.random.permutation(len(data))
        data, target=data[idxs], target[idxs]
        return data, target, None, None
    else:
        return x_train, y_train, x_test, y_test



def get_mnist():
    transform_train = transforms.Compose([transforms.ToTensor(),])
    transform_test = transforms.Compose([transforms.ToTensor(),])
    trainset = datasets.MNIST(root='.', train=True, download=True,transform=transform_train)
    testset = datasets.MNIST(root='.', train=False, download=True,transform=transform_test)

    (x_train, y_train), (x_test, y_test) = (trainset.data, trainset.targets), (testset.data, testset.targets)

    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    x_train = x_train.numpy()
    y_train = y_train.numpy()
    x_test = x_test.numpy()
    y_test = y_test.numpy()
    # print(x_train.shape)
    # print(x_test.shape)

    return x_train, y_train, x_test, y_test
'''
def get_mnist():
    transform_train = transforms.Compose([transforms.ToTensor(),])
    transform_test = transforms.Compose([transforms.ToTensor(),])
    trainset = datasets.MNIST(root='.', train=True, download=True,transform=transform_train)
    testset = datasets.MNIST(root='.', train=False, download=True,transform=transform_test)

    (x_train, y_train), (x_test, y_test) = (trainset.data, trainset.targets), (testset.data, testset.targets)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, stratify=y_train, random_state=42)

    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
    x_val = x_val.reshape((x_val.shape[0], 28, 28, 1))
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
    x_train = x_train / 255.0
    x_val = x_val / 255.0
    x_test = x_test / 255.0

    x_train = x_train.numpy()
    x_val = x_val.numpy()
    y_train = y_train.numpy()
    y_val = y_val.numpy()
    x_test = x_test.numpy()
    y_test = y_test.numpy()

    return x_train, y_train, x_val, y_val, x_test, y_test
'''

def get_fmnist():
    transform_train = transforms.Compose([transforms.ToTensor(),])
    transform_test = transforms.Compose([transforms.ToTensor(),])
    trainset = datasets.FashionMNIST(root='.', train=True, download=True,transform=transform_train)
    testset = datasets.FashionMNIST(root='.', train=False, download=True,transform=transform_test)

    (x_train, y_train), (x_test, y_test) = (trainset.data, trainset.targets), (testset.data, testset.targets)

    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    x_train = x_train.numpy()
    y_train = y_train.numpy()
    x_test = x_test.numpy()
    y_test = y_test.numpy()

    return x_train, y_train, x_test, y_test




def get_cifar():
    transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
    transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
    trainset = torchvision.datasets.CIFAR10(root='.', train=True, download=True, transform=transform_train) 
    testset = torchvision.datasets.CIFAR10(root='.', train=False, download=True, transform=transform_test)

    (x_train, y_train), (x_test, y_test) = (trainset.data, trainset.targets), (testset.data, testset.targets)

    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    return x_train, y_train, x_test, y_test


def get_dogcat():

    x_train, y_train, x_test, y_test = get_cifar()

    dogcat_ind = np.where(np.logical_or(y_train==3, y_train==5))[0]
    x_train, y_train = x_train[dogcat_ind], y_train[dogcat_ind]
    y_train[y_train==3] = 0
    y_train[y_train==5] = 1

    dogcat_ind = np.where(np.logical_or(y_test==3, y_test==5))[0]
    x_test, y_test = x_test[dogcat_ind], y_test[dogcat_ind]
    y_test[y_test==3] = 0
    y_test[y_test==5] = 1

    return x_train, y_train, x_test, y_test


def get_dogcatFeature():

    # x_train, y_train, x_test, y_test = pickle.load( open('dogvscat_feature.dataset', 'rb') )
    # x_train, y_train, x_test, y_test = pickle.load( open('result/DogCatImageNetPretrain.data', 'rb') )

    import torch
    import torchvision
    import torchvision.transforms as transforms
    from sklearn.neighbors import KNeighborsClassifier
    from torchvision.models import resnet50

    # 加载 CIFAR-10 数据集
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=False)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False)

    # 加载预训练的 ResNet 模型
    model = resnet50(pretrained=True)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))  # 移除最后的全连接层
    model.eval()

    # 提取训练集和测试集的特征
    x_train, y_train = next(iter(trainloader))
    x_train = model(x_train).detach().numpy().reshape(len(trainset), -1)
    #x_train = x_train.numpy().reshape(len(trainset), -1)
    y_train = y_train.numpy()

    x_test, y_test = next(iter(testloader))
    x_test = model(x_test).detach().numpy().reshape(len(testset), -1)
    #x_test = x_test.numpy().reshape(len(testset), -1)
    y_test = y_test.numpy()

    # # 使用 KNN 进行分类
    # knn = KNeighborsClassifier(n_neighbors=10)
    # knn.fit(x_train, y_train)

    # # 预测和评估
    # accuracy = knn.score(x_test, y_test)
    # print("Accuracy:", accuracy)

    # file_path1 = os.path.expanduser('~/桌面/Brandeis/Research/cknnsv-v2/cifar10_feature.npz')
    # np.savez(file_path1, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)


    
    return x_train, y_train, x_test, y_test


def get_ag_newsFeature():
    from torchtext.datasets import AG_NEWS
    from sentence_transformers import SentenceTransformer
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    # 加载 Sentence-BERT 模型
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # 下载 AG News 数据集
    train_iter, test_iter = AG_NEWS(split=('train', 'test'))

    # 函数：提取数据集的特征和标签
    def extract_features(dataset, max_samples=20000):
        labels = []
        embeddings = []
        for i, (label, text) in enumerate(dataset):
            if i >= max_samples:  # 限制样本数量以节省内存和计算资源
                break
            labels.append(label - 1)  # 将标签从 [1, 4] 调整为 [0, 3]
            embedding = model.encode(text, convert_to_tensor=True)
            embeddings.append(embedding.cpu().numpy())
        return embeddings, labels

    # 提取训练和测试数据的特征
    X_train, y_train = extract_features(train_iter, max_samples=20000)
    X_test, y_test = extract_features(test_iter, max_samples=2000)
    
    # 确保特征是二维数组
    X_train = np.vstack(X_train)
    y_train = np.array(y_train)
    X_test = np.vstack(X_test)
    y_test = np.array(y_test)
    # print(X_train.shape)
    # print(y_train.shape)

    # # 使用 KNN 分类器
    # knn_classifier = KNeighborsClassifier(n_neighbors=3)
    # knn_classifier.fit(X_train, y_train)

    # # 预测并计算准确度
    # y_pred = knn_classifier.predict(X_test)
    # accuracy = accuracy_score(y_test, y_pred)
    # print(f"Accuracy: {accuracy * 100:.2f}%")


    return X_train, y_train, X_test, y_test

def get_DBPediaFeature():
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score

    # 加载 Sentence-BERT 模型
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # 加载 DBpedia 数据集
    dataset = load_dataset("dbpedia_14")
    train_dataset = dataset['train']
    test_dataset = dataset['test']

    # 函数：提取数据集的特征和标签
    def extract_features(dataset, max_samples=1000):
        labels = []
        embeddings = []
        for i, example in enumerate(dataset):
            if i >= max_samples:  # 限制样本数量
                break
            labels.append(example['label'])
            embedding = model.encode(example['content'], convert_to_tensor=True)
            embeddings.append(embedding.cpu().numpy())
        return embeddings, labels

    # 提取训练和测试数据的特征
    X_train, y_train = extract_features(train_dataset, max_samples=30000)
    X_test, y_test = extract_features(test_dataset, max_samples=3000)

    # 将特征转换为二维数组
    X_train = np.vstack(X_train)
    y_train = np.array(y_train)
    X_test = np.vstack(X_test)
    y_test = np.array(y_test)

    # # 使用 KNN 分类器
    # knn_classifier = KNeighborsClassifier(n_neighbors=3)
    # knn_classifier.fit(X_train, y_train)

    # # 预测并计算准确度
    # y_pred = knn_classifier.predict(X_test)
    # accuracy = accuracy_score(y_test, y_pred)
    # print(f"Accuracy: {accuracy * 100:.2f}%")

    return X_train, y_train, X_test, y_test


def get_IMDbFeature():
    from torchtext.datasets import IMDB
    from sentence_transformers import SentenceTransformer
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    # 加载 Sentence-BERT 模型
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # 下载 IMDB 数据集
    train_iter, test_iter = IMDB(split=('train', 'test'))

    # 函数：提取数据集的特征和标签
    def extract_features(dataset, max_samples=1000):
        labels = []
        embeddings = []
        for i, (label, text) in enumerate(dataset):
            if i >= max_samples:  # 限制样本数量
                break
            labels.append(0 if label == 'neg' else 1)
            embedding = model.encode(text, convert_to_tensor=True)
            embeddings.append(embedding.cpu().numpy())
        return embeddings, labels

    # 提取训练和测试数据的特征
    X_train, y_train = extract_features(train_iter, max_samples=20000)
    X_test, y_test = extract_features(test_iter, max_samples=2000)

    # 将特征转换为二维数组
    X_train = np.vstack(X_train)
    y_train = np.array(y_train)
    X_test = np.vstack(X_test)
    y_test = np.array(y_test)

    # 使用 KNN 分类器
    knn_classifier = KNeighborsClassifier(n_neighbors=3)
    knn_classifier.fit(X_train, y_train)

    # 预测并计算准确度
    y_pred = knn_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    return X_train, y_train, X_test, y_test

def get_SSTFeature():
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score

    # 加载 Sentence-BERT 模型
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # 加载 SST-2 数据集
    dataset = load_dataset("glue", "sst2")
    train_dataset = dataset['train']
    validation_dataset = dataset['validation']

    # 函数：提取数据集的特征和标签
    def extract_features(dataset, max_samples=1000):
        labels = []
        embeddings = []
        for i, example in enumerate(dataset):
            if i >= max_samples:  # 限制样本数量
                break
            labels.append(example['label'])
            embedding = model.encode(example['sentence'], convert_to_tensor=True)
            embeddings.append(embedding.cpu().numpy())
        return embeddings, labels

    # 提取训练和验证数据的特征
    X_train, y_train = extract_features(train_dataset, max_samples=30000)
    X_val, y_val = extract_features(validation_dataset, max_samples=3000)

    # 将特征转换为二维数组
    X_train = np.vstack(X_train)
    y_train = np.array(y_train)
    X_val = np.vstack(X_val)
    y_val = np.array(y_val)
    # print(X_train.shape)
    # print(y_train.shape)

    # # 使用 KNN 分类器
    # knn_classifier = KNeighborsClassifier(n_neighbors=10)
    # knn_classifier.fit(X_train, y_train)

    # # 预测并计算准确度
    # y_pred = knn_classifier.predict(X_val)
    # accuracy = accuracy_score(y_val, y_pred)
    # print(f"Accuracy: {accuracy * 100:.2f}%")

    return X_train, y_train, X_val, y_val


def get_20newsgroups_feature():
    from sklearn.datasets import fetch_20newsgroups
    from sentence_transformers import SentenceTransformer
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score

    # 加载 Sentence-BERT 模型
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # 加载 20 Newsgroups 数据集
    newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))
    texts_train = newsgroups_train.data
    labels_train = newsgroups_train.target
    texts_test = newsgroups_test.data
    labels_test = newsgroups_test.target

    # 函数：提取数据集的特征和标签
    def extract_features(texts, max_samples=20000):
        embeddings = []
        for i, text in enumerate(texts):
            if i >= max_samples:  # 限制样本数量
                break
            embedding = model.encode(text, convert_to_tensor=True)
            embeddings.append(embedding.cpu().numpy())
        return embeddings

    # 提取训练和测试数据的特征
    X_train = extract_features(texts_train)
    labels_train = np.array(labels_train)
    X_test = extract_features(texts_test)
    labels_test = np.array(labels_test)

    # 将特征转换为二维数组
    X_train = np.vstack(X_train)
    X_test = np.vstack(X_test)

    # # 使用 KNN 分类器
    # knn_classifier = KNeighborsClassifier(n_neighbors=3)
    # knn_classifier.fit(X_train, labels_train)

    # # 预测并计算准确度
    # y_pred = knn_classifier.predict(X_test)
    # accuracy = accuracy_score(labels_test, y_pred)
    # print(f"Accuracy: {accuracy * 100:.2f}%")

    return X_train, labels_train, X_test, labels_test







