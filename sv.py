# python sv.py --dataset cpu --value_type CKNN --n_data 2000 --n_val 200
from ast import arg
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
from prepare_data import *
import config

import pdb

import argparse

import matplotlib.pyplot as plt

#torch.multiprocessing.set_start_method('spawn')

parser = argparse.ArgumentParser('')




parser.add_argument('--dataset', type=str)
parser.add_argument('--value_type', type=str)
parser.add_argument('--model_type', type=str)
parser.add_argument('--n_data', type=int, default=500)
parser.add_argument('--n_val', type=int, default=2000)
parser.add_argument('--n_repeat', type=int, default=1)
parser.add_argument('--n_sample', type=int)
parser.add_argument('--random_state', type=int, default=1)
parser.add_argument('--flip_ratio', type=float, default=0)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--debug', action='store_true')

parser.add_argument('--K', type=int, default=10)
parser.add_argument('--T', type=int, default=1980)
parser.add_argument('--task', type=str, default='remove_detrimental')
args = parser.parse_args()

dataset = args.dataset
value_type = args.value_type
model_type = args.model_type
n_data = args.n_data
n_val = args.n_val
n_repeat = args.n_repeat
n_sample = args.n_sample
random_state = args.random_state
flip_ratio = args.flip_ratio
batch_size = args.batch_size
lr = args.lr

k = args.K
T = args.T
#T = n_data - 2*k


verbose = 0
if args.debug:
    verbose = 1


save_dir = 'result/'
big_dataset = config.big_dataset
OpenML_dataset = config.OpenML_dataset


if dataset in big_dataset+OpenML_dataset:
    save_name = save_dir+'{}_{}_{}_Ndata{}_Nval{}_Nsample{}_BS{}_LR{}_Nrepeat{}_FR{}_Seed{}.data'.format(
        value_type, dataset, model_type, n_data, n_val, n_sample, batch_size, lr, n_repeat, flip_ratio, random_state)
else:
    save_name = save_dir+'{}_{}_{}_Ndata{}_Nval{}_Nsample{}_FR{}.data'.format(
        value_type, dataset, model_type, n_data, n_val, n_sample, flip_ratio)



#print(n_val)
#x_train, y_train, x_val, y_val, x_val2, y_val2 = get_processed_data(dataset, n_data, n_val, flip_ratio)
x_train, y_train, x_val, y_val = get_processed_data(dataset, n_data, n_val, flip_ratio)
print(x_val.shape)
#x_train, y_train, x_val, y_val, x_test, y_test = get_processed_data(dataset, n_data, n_val, flip_ratio)
task = args.task
test_time = False
if test_time:
    #x_train, y_train, x_val, y_val生成同shape的随机数据，数据量要变大
    new_batch_size = 100000  # multiplier is the factor by which you want to increase the batch size
    x_train = np.random.rand(new_batch_size, *x_train.shape[1:])
    y_train = np.random.randint(low=0, high=1, size=(new_batch_size, *y_train.shape[1:]))
    x_val = np.random.rand(new_batch_size//10, *x_train.shape[1:])
    y_val = np.random.randint(low=0, high=1, size=(new_batch_size//10, *y_train.shape[1:]))
    


if task == 'weighted_knn' or task == 'remove_negative' or task == 'remove_detrimental':
    pass
else:
    # utulity function
    utility_func_args = (x_train, y_train, x_val, y_val)
    u_func = get_ufunc(dataset, model_type, batch_size, lr, verbose, device='cuda:0')
    utility_func_mult = lambda a, b, c, d: sample_utility_multiple(a, b, c, d, u_func, n_repeat)
    def utility_func_mult(a, b, c, d):
        return sample_utility_multiple(a, b, c, d, u_func, n_repeat)

n_class = len(np.unique(y_val))
sv_baseline = 1.0/n_class

if(random_state != -1): 
    torch.manual_seed(random_state)
    torch.cuda.manual_seed(random_state)
    np.random.seed(random_state)
    random.seed(random_state)


def process_yfeature(y_feature):
    y_feature = np.array(y_feature)
    if n_repeat==1:
        y_feature = y_feature.reshape(-1)
    return y_feature






if value_type == 'Banzhaf_MC':
    n_sample_per_data = int(n_sample / n_data)
    save_arg = {}
    for target_ind in range(n_data):
        utility_set_tgt = sample_utility_banzhaf_mc(n_sample_per_data, utility_func_mult, utility_func_args, target_ind)
        save_arg[target_ind] = utility_set_tgt

elif value_type == 'Shapley_Perm':
    n_perm = int(n_sample / n_data)
    #X_feature_test, y_feature_test = sample_utility_shapley_perm(n_perm, utility_func_mult, utility_func_args)
    X_feature_test, y_feature_test = sample_utility_shapley_perm_parallel(n_perm, utility_func_mult, utility_func_args)
    y_feature_test = process_yfeature(y_feature_test)
    save_arg = {'X_feature': X_feature_test, 'y_feature': y_feature_test}

elif value_type == 'Banzhaf_GT':
    X_feature_test, y_feature_test = sample_utility_banzhaf_gt(n_sample, utility_func_mult, utility_func_args, dummy=True)
    y_feature_test = process_yfeature(y_feature_test)
    save_arg = {'X_feature': X_feature_test, 'y_feature': y_feature_test}

elif value_type == 'Shapley_GT':
    X_feature_test, y_feature_test = sample_utility_shapley_gt(n_sample, utility_func_mult, utility_func_args)
    y_feature_test = process_yfeature(y_feature_test)
    save_arg = {'X_feature': X_feature_test, 'y_feature': y_feature_test}

elif value_type == 'LOO':
    X_feature_test, y_feature_test, u_total = sample_utility_loo(utility_func_mult, utility_func_args)
    y_feature_test = process_yfeature(y_feature_test)
    u_total = np.array(u_total)
    if n_repeat==1:
        u_total = u_total[0]
    save_arg = {'X_feature': X_feature_test, 'y_feature': y_feature_test, 'u_total': u_total}

elif value_type == 'KNN':
    print("KNN_Shapley")
    k=10

    start_time = time.time()
    sv1, sorted_indices1, sum_sv_per_val, sv_matrix = knn_shapley(x_train, y_train, x_val, y_val, K=k)
    #sv1, sorted_indices1, sum_sv_per_val, sv_matrix = knn_t_shapley(x_train, y_train, x_val, y_val, K=k)
    #sv1 = tnn_shapley(x_train, y_train, x_val, y_val, tau=-0.5, K0=k, dis_metric='cosine')
    end_time = time.time()
    print('sv running time:', end_time-start_time)
    acc1 = knn_classifier(x_train, y_train, x_val, y_val, k=k)
    

    # file_path1 = os.path.expanduser('~/桌面/Brandeis/Research/cknnsv/draw/cknnsv_matrix.npy')
    # np.save(file_path1, sv_matrix)

    print('acc baseline:', acc1)
    # print('sv sum before remove:',np.sum(sv1))
    # print('sv mean before remove:',np.mean(sv1))
    # print('sv var before remove:',np.var(sv1))
    # print('sv negative before remove:', np.sum(sv1 < 0))

    save_arg = {'knn': sv1}

    '''
    from scipy.stats import norm

    # 计算数据的均值和标准差
    mu, sigma = np.mean(sv1 / 1), np.std(sv1 / 1)
    min_value = sv1.min()
    max_value = sv1.max()
    bin_edges = np.linspace(min_value, max_value, num=21) 

    sv1_sorted = np.sort(sv1)

    # 计算每个直方内的元素个数
    elements_per_bin = len(sv1_sorted) // 20  # 2000个元素分成10个直方

    # 创建直方的边界值，确保每个直方内的元素个数相同
    # 对于最后一个边界，直接使用数组的最大值
    bin_edges = [sv1_sorted[i * elements_per_bin] if i < 10 else sv1_sorted[-1] for i in range(11)]
    # 定义10种不同的颜色
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'grey', 'brown']

    # 绘制每个直方，并为每个直方指定不同的颜色
    for i in range(10):
        plt.hist(sv1, bins=[bin_edges[i], bin_edges[i+1]], color=colors[i], alpha=0.6)


    # 假设 sv1_sum, x_train, y_train, x_val, y_val 已经定义
    # 假设 knn_classifier 是一个函数，用于计算KNN分类器的准确度

    # 创建一个图表和两个轴
    fig, ax1 = plt.subplots()

    # 首先对数据进行排序
    sorted_data = np.sort(sv1.flatten())

    # 计算每个柱状条应该包含的数据点数
    points_per_bin = len(sorted_data) // 20
    colors = plt.cm.viridis(np.linspace(0, 1, 10))

    # 计算分界点
    bins = [sorted_data[i] if i < len(sorted_data) else sorted_data[-1] for i in range(0, len(sorted_data), points_per_bin)]

    # 最后一个边界应该是数据的最大值
    bins.append(sorted_data[-1])

    # 在第一个轴（ax1）上绘制直方图
    n, bins, patches = ax1.hist(sv1.flatten(), bins=bins, alpha=0.6, color='g')


    # 初始化存储准确度的列表
    acc_1 = []
    # 初始化存储每个区间内sv1_sum总和的列表
    sv1_totals = []


    # 遍历直方图的每个柱状条
    for i in range(len(bins)-1):
        lower_bound = bins[i]
        upper_bound = bins[i+1]

        # 找到 sv1_sum 中在当前柱状条范围内的索引
        indices = np.where((sv1 >= lower_bound) & (sv1 < upper_bound))[0]

        # 找到sv1_sum中在当前区间内的值
        total = np.sum(sv1[indices])
        sv1_totals.append(total)

        # 删除 x_train 和 y_train 中对应的元素
        x_train_ = np.delete(x_train, indices, axis=0)
        y_train_ = np.delete(y_train, indices, axis=0)

        #sv3, sorted_indices1, sum_sv_per_val, sv_matrix = knn_t_shapley(x_train_, y_train_, x_val, y_val, K=k)
        
        # 计算KNN分类器的准确度
        acc1_ = knn_classifier(x_train_, y_train_, x_val, y_val, k=k)
        acc_1.append(acc1_)
        print('from', lower_bound, 'to', upper_bound, 'acc:', acc1_)

    file_path2 = os.path.expanduser('~/桌面/Brandeis/Research/cknnsv/draw/knnacc.npz')
    np.savez(file_path2, acc_1)

    # 计算柱状条的中心点
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # 计算柱状条的宽度
    widths = np.diff(bins)
    
    # 在第一个轴（ax1）上绘制直方图
    #n, bins, patches = ax1.hist(sv1_sum.flatten(), bins=50, density=False, alpha=0.6, color='g')
    ax1.set_xlabel('sv1的值')
    ax1.set_ylabel('频数', color='g')
    ax1.tick_params(axis='y', labelcolor='g')
    ax1.axvline(x=0, color='b', linestyle='--')
    ax1.bar(bins[:-1], sv1_totals, width=widths, align='edge', alpha=0.6, color='g')

    # 使用twinx创建第二个轴（ax2），它共享相同的x轴
    ax2 = ax1.twinx()
    ax2.plot(bin_centers, acc_1, 'r-', marker='o')
    ax2.set_ylabel('准确度 (acc)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.axhline(y=acc1, color='b', linestyle='--')

    # 显示带有两个纵坐标的图表
    plt.show()
    '''

    '''
    # 创建一个图表和两个轴
    fig, ax1 = plt.subplots()

    # 首先对数据进行排序
    sorted_data = np.sort(sv1.flatten())

    # 计算每个柱状条应该包含的数据点数
    points_per_bin = len(sorted_data) // 10

    # 计算分界点
    bins = [sorted_data[i] if i < len(sorted_data) else sorted_data[-1] for i in range(0, len(sorted_data), points_per_bin)]

    # 最后一个边界应该是数据的最大值
    bins.append(sorted_data[-1])

    # 初始化存储准确度的列表
    acc_1 = []
    # 初始化存储每个区间内sv1_sum总和的列表
    sv1_totals = []

    # 遍历直方图的每个柱状条
    for i in range(len(bins)-1):
        lower_bound = bins[i]
        upper_bound = bins[i+1]

        # 找到 sv1_sum 中在当前柱状条范围内的索引
        indices = np.where((sv1 >= lower_bound) & (sv1 < upper_bound))[0]

        # 计算sv1_sum中当前区间内的总和
        total = np.sum(sv1[indices])
        sv1_totals.append(total)

        # 删除 x_train 和 y_train 中对应的元素
        x_train_ = np.delete(x_train, indices, axis=0)
        y_train_ = np.delete(y_train, indices, axis=0)

        # 计算KNN分类器的准确度
        acc1_ = knn_classifier(x_train_, y_train_, x_val, y_val, k=k)
        acc_1.append(acc1_)
        print('from', lower_bound, 'to', upper_bound, 'acc:', acc1_)

    # 计算柱状条的宽度
    widths = np.diff(bins)

    # 计算柱状条的中心点
    bin_centers = 0.5 * (np.array(bins[:-1]) + np.array(bins[1:]))


    # 在第一个轴（ax1）上绘制直方图，展示sv1_sum的总和
    ax1.bar(bins[:-1], sv1_totals, width=widths, align='edge', alpha=0.6, color='g')
    ax1.set_xlabel('sv1的值')
    ax1.set_ylabel('总和', color='g')
    ax1.tick_params(axis='y', labelcolor='g')
    ax1.axvline(x=0, color='b', linestyle='--')
    # 反转左侧y轴的方向
    ax1.invert_yaxis()

    # 使用twinx创建第二个轴（ax2），它共享相同的x轴
    ax2 = ax1.twinx()
    ax2.plot(bin_centers, acc_1, 'r-', marker='o')
    ax2.set_ylabel('准确度 (acc)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.axhline(y=acc1, color='b', linestyle='--')

    # 显示带有两个纵坐标的图表
    plt.show()
    '''
    
elif value_type == 'TNN':
    print("TKNN_Shapley")
    k=10

    start_time = time.time()
    #sv1, sorted_indices1, sum_sv_per_val, sv_matrix = knn_shapley(x_train, y_train, x_val, y_val, K=k)
    #sv1, sorted_indices1, sum_sv_per_val, sv_matrix = knn_t_shapley(x_train, y_train, x_val, y_val, K=k)
    sv1 = tnn_shapley(x_train.reshape(x_train.shape[0], -1), y_train, x_val.reshape(x_val.shape[0], -1), y_val, tau=-0.5, K0=k, dis_metric='cosine')
    end_time = time.time()
    print('sv running time:', end_time-start_time)

    # file_path1 = os.path.expanduser('~/桌面/Brandeis/Research/cknnsv-v2/tnnsv_flip0.1.npy')
    # np.save(file_path1, sv1)
    # sv1 = np.load(file_path1)

    # acc1 = knn_classifier(x_train, y_train, x_val, y_val, k=k)
    # print('acc baseline:', acc1)
    # print('sv sum before remove:',np.sum(sv1))
    # print('sv mean before remove:',np.mean(sv1))
    # print('sv var before remove:',np.var(sv1))
    print('sv negative before remove:', np.sum(sv1 < 0))
    print('sv zeros:', np.sum(sv1 == 0))

    save_arg = {'knn': sv1}

elif value_type == 'CKNN':
    print("CKNN_Shapley")
    #k=10
    #T = len(y_train) -2*k
    start_time = time.time()
    #sv1, sorted_indices1, sum_sv_per_val, sv_matrix = knn_shapley(x_train, y_train, x_val, y_val, K=k)
    sv1, sorted_indices1, sum_sv_per_val, sv_matrix = knn_t_shapley(x_train, y_train, x_val, y_val, K=k, T=T)
    #sv1 = tnn_shapley(x_train, y_train, x_val, y_val, tau=-0.5, K0=k, dis_metric='cosine')
    end_time = time.time()
    print('sv running time:', end_time-start_time)
    acc1 = knn_classifier(x_train, y_train, x_val, y_val, k=k)
    #acc2 = knn_classifier(x_train, y_train, x_val2, y_val2, k=k)
    

    # file_path1 = os.path.expanduser('~/桌面/cknnsv_flip0.3.npy')
    # np.save(file_path1, sv1)

    print('acc1 baseline:', acc1)
    #print('acc2 baseline:', acc2)

    # print('sv sum before remove:',np.sum(sv1))
    # print('sv mean before remove:',np.mean(sv1))
    # print('sv var before remove:',np.var(sv1))
    # print('sv negative before remove:', np.sum(sv1 < 0))

    save_arg = {'knn': sv1}

elif value_type == 'KNN-JW':
    print("KNN-JW_Shapley")
    k=10
    start_time = time.time()
    sv1 = knn_shapley_JW(x_train, y_train, x_val, y_val, K=k)
    end_time = time.time()
    print('sv running time:', end_time-start_time)
    acc1 = knn_classifier(x_train, y_train, x_val, y_val, k=k)
    print('acc baseline:', acc1)
    # print('sv sum before remove:',np.sum(sv1))
    # print('sv mean before remove:',np.mean(sv1))
    # print('sv var before remove:',np.var(sv1))
    # print('sv negative before remove:', np.sum(sv1 < 0))

    save_arg = {'knn': sv1}
#task = 'remove_negative'

save_arg['sv_baseline'] = sv_baseline
if value_type != 'Uniform':

  if value_type in ['Shapley_Perm', 'Banzhaf_GT', 'BetaShapley'] and dataset in big_dataset:
    args.n_sample *= 10
  value_args = save_arg
  value_args['n_data'] = n_data
else:
  value_args = {}
  value_args['n_data'] = n_data
v_args = copy.deepcopy(value_args)

if value_type in ['Shapley_Perm', 'Banzhaf_GT', 'BetaShapley']:

    if dataset in big_dataset or dataset in OpenML_dataset :
        v_args['y_feature'] = value_args['y_feature']#[:, 0]
    else:
        v_args['y_feature'] = np.clip( value_args['y_feature'] + np.random.normal(scale=args.sigma, size=n_sample) , a_min=0, a_max=1)
    sv1 = compute_value(value_type, v_args)
elif value_type == 'LOO':

    if dataset in big_dataset or dataset in OpenML_dataset :
      v_args['y_feature'] = value_args['y_feature'][:, 0]
      v_args['u_total'] = value_args['u_total'][0]
    else:
      v_args['y_feature'] = np.clip( value_args['y_feature']+np.random.normal(scale=args.sigma, size=len(value_args['y_feature'])), a_min=0, a_max=1)
      v_args['u_total'] = np.clip( value_args['u_total']+np.random.normal(scale=args.sigma), a_min=0, a_max=1)



if task=='weighted_knn':
    print('weighted_knn')
    acc2 = weighted_knn_classifier(x_train, y_train, x_val, y_val, k=k, sample_weights=sv1)
    print('acc after weighted:', acc2)

elif task=='remove_negative':
    print('remove_negative')
    valid_indices = np.where(sv1 > 0)[0]

    # 使用这些索引来筛选 x_train 和 y_train
    x_train_ = x_train[valid_indices]
    y_train_ = y_train[valid_indices]
    sv1 = sv1[valid_indices]
    acc = knn_classifier(x_train_, y_train_, x_val, y_val, k=k)
    
    # from sklearn.neighbors import KNeighborsClassifier
    # knn = KNeighborsClassifier(n_neighbors=10)
    # # X_train 是特征数据，y_train 是对应的标签
    # knn.fit(x_train_, y_train_)

    # # X_test 是你要预测的新数据
    # predictions = knn.predict(x_val)

    # from sklearn.metrics import accuracy_score
    # acc = accuracy_score(y_val, predictions)


    print('acc after remov:', acc)

elif task=='remove_detrimental':
    print('remove_detrimental')
    
    def analyze_bins(sv, x_train, y_train, x_val, y_val, k):
      # Step 1: Sort data and create bins
      sorted_indices = np.argsort(sv)
      bins = np.array_split(sorted_indices, 100)

      # Step 2: Identify detrimental bins
      detrimental_bins = []
      base_acc = knn_classifier(x_train, y_train, x_val, y_val, k)
      for bin in bins:
          # Remove bin from training set
          x_train_reduced = np.delete(x_train, bin, axis=0)
          y_train_reduced = np.delete(y_train, bin, axis=0)

          # Check performance on validation set
          acc = knn_classifier(x_train_reduced, y_train_reduced, x_val, y_val, k)
          if acc > base_acc:
              detrimental_bins.append(bin)

      # Step 3: Analyze detrimental bins
      detrimental_ranges = [(sv[bin].min(), sv[bin].max(), np.mean(sv[bin])) for bin in detrimental_bins]

      # Step 4: Define and calculate metrics
      metrics = {
          'inflation_index': max([sv[bin].max() for bin in detrimental_bins], default=0),
          'inflation_proportion': sum([np.sum(sv[bin] > 0) for bin in detrimental_bins]) / sum([len(bin) for bin in detrimental_bins]) if detrimental_bins else 0
      }

      return detrimental_ranges, metrics
    
    def analyze_bins(sv, x_train, y_train, x_val, y_val, k):
        # Step 1: Sort data and create bins
        sorted_indices = np.argsort(sv)
        bins = np.array_split(sorted_indices, 100)

        # Step 2: Identify detrimental bins and predict bin status
        detrimental_bins = []
        predicted_detrimental_bins = []
        base_acc = knn_classifier(x_train, y_train, x_val, y_val, k)
        false_positives = 0  # Initialize false positives
        false_negatives = 0  # Initialize false negatives
        
        for bin in bins:
            # Remove bin from training set
            x_train_reduced = np.delete(x_train, bin, axis=0)
            y_train_reduced = np.delete(y_train, bin, axis=0)

            # Check performance on validation set
            acc = knn_classifier(x_train_reduced, y_train_reduced, x_val, y_val, k)
            
            # Predict bin as beneficial if the sum of sv > 0, else detrimental
            if sv[bin].sum() > 0:
                predicted_beneficial = True
            else:
                predicted_beneficial = False
            
            if acc > base_acc:
              detrimental_bins.append(bin)
            # Actual beneficial or detrimental based on acc comparison
            actual_beneficial = acc < base_acc
            
            # Compare predictions with actual to compute FP and FN
            if predicted_beneficial and not actual_beneficial:
                false_positives += 1  # Predicted beneficial but was detrimental
                predicted_detrimental_bins.append(bin)  # For analysis
            elif not predicted_beneficial and actual_beneficial:
                false_negatives += 1  # Predicted detrimental but was beneficial

        # Step 3: Analyze detrimental bins
        detrimental_ranges = [(sv[bin].min(), sv[bin].max(), np.mean(sv[bin])) for bin in detrimental_bins]

        # Step 4: Define and calculate metrics
        metrics = {
            'inflation_index': max([sv[bin].max() for bin in detrimental_bins], default=0),
            'inflation_proportion': sum([np.sum(sv[bin] > 0) for bin in detrimental_bins]) / sum([len(bin) for bin in detrimental_bins]) if detrimental_bins else 0,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }

        return detrimental_ranges, metrics

    detrimental_ranges, metrics = analyze_bins(sv1, x_train, y_train, x_val, y_val, k=k)
    #print('Detrimental ranges:', detrimental_ranges)
    print('Metrics:', metrics)

elif task=='mislabel detection':
    print('mislabel detection')
    # acc1, acc2 = kmeans_f1score(sv1, cluster=False), kmeans_f1score(sv1, cluster=True)
    # print('f1 score without cluster:', acc1)
    # print('f2 score with cluster:', acc2)
    # aucroc = aucroc(sv1)
    # print('aucroc:', aucroc)
    from sklearn.metrics import precision_score, recall_score, f1_score
    print('sv negative:', np.sum(sv1 < 0))
    # 数据点总数
    n_data = len(sv1)

    # 创建真实的错误标签数组
    true_labels = np.zeros(n_data)
    true_labels[:int(0.1 * n_data)] = 1  # 假设前10%的标签是错误的

    # 创建基于 Shapley 值的预测标签
    predicted_labels = np.zeros(n_data)
    predicted_labels[sv1 < 0] = 1  # Shapley 值小于0表示预测为错误标签

    # 计算 F1 分数
    f1 = f1_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    print('F1 score:', f1)
    print('precision:', precision)
    print('recall:', recall)

elif task=='online_learning':
    print('online_learning')
    num_batches = 10
    bi = np.linspace(0, len(y_train), num=num_batches+1).astype(int)
    for i in range(num_batches):
        if i ==0:
            batches_x = x_train[bi[i]:bi[i+1]]
            batches_y = y_train[bi[i]:bi[i+1]]
        else:
            batches_x = np.concatenate((batches_x_, x_train[bi[i]:bi[i+1]]), axis=0)
            batches_y = np.concatenate((batches_y_, y_train[bi[i]:bi[i+1]]), axis=0)

        acc1 = knn_classifier(batches_x, batches_y, x_val, y_val, k=k)
        #acc2 = knn_classifier(batches_x, batches_y, x_val2, y_val2, k=k)
        print('batch ', i, 'acc1 before remove:')
        
        sv1, sorted_indices1, sum_sv_per_val, sv_matrix = knn_t_shapley(batches_x, batches_y, x_val, y_val, K=k)
        print('sv negative before remove:', np.sum(sv1 <= 0))

        valid_indices = np.where(sv1 > 0)[0]
        # 使用这些索引来筛选 x_train 和 y_train
        batches_x_ = batches_x[valid_indices]
        batches_y_ = batches_y[valid_indices]

        acc1 = knn_classifier(batches_x_, batches_y_, x_val, y_val, k=k)
        #acc2 = knn_classifier(batches_x_, batches_y_, x_val2, y_val2, k=k)
        print('acc1 after remov:', acc1)

elif task=='remove_highest':
    print('remove_highest')
    n_data = len(sv1)
    indices = np.argsort(sv1)
    ascending_indices_10per = indices[int(0.99 * n_data):]
    ascending_indices_20per = indices[int(0.95 * n_data):]
    ascending_indices_30per = indices[int(0.90 * n_data):]
    # 从训练集中删除这些数据
    x_train_ = np.delete(x_train, ascending_indices_10per, axis=0)
    y_train_ = np.delete(y_train, ascending_indices_10per, axis=0)
    print('acc after remove top 1%:', knn_classifier(x_train_, y_train_, x_val, y_val, k=k))

    x_train_ = np.delete(x_train, ascending_indices_20per, axis=0)
    y_train_ = np.delete(y_train, ascending_indices_20per, axis=0)
    print('acc after remove top 5%:', knn_classifier(x_train_, y_train_, x_val, y_val, k=k))

    x_train_ = np.delete(x_train, ascending_indices_30per, axis=0)
    y_train_ = np.delete(y_train, ascending_indices_30per, axis=0)
    print('acc after remove top 10%:', knn_classifier(x_train_, y_train_, x_val, y_val, k=k))

elif task=='remove_lowest':
    print('remove_lowest')
    n_data = len(sv1)
    indices = np.argsort(sv1)
    ascending_indices_10per = indices[:int(0.1 * n_data)]
    ascending_indices_20per = indices[:int(0.2 * n_data)]
    ascending_indices_30per = indices[:int(0.3 * n_data)]
    # 从训练集中删除这些数据
    x_train_ = np.delete(x_train, ascending_indices_10per, axis=0)
    y_train_ = np.delete(y_train, ascending_indices_10per, axis=0)
    print('acc after remove bottom 10%:', knn_classifier(x_train_, y_train_, x_val, y_val, k=k))

    x_train_ = np.delete(x_train, ascending_indices_20per, axis=0)
    y_train_ = np.delete(y_train, ascending_indices_20per, axis=0)
    print('acc after remove bottom 20%:', knn_classifier(x_train_, y_train_, x_val, y_val, k=k))

    x_train_ = np.delete(x_train, ascending_indices_30per, axis=0)
    y_train_ = np.delete(y_train, ascending_indices_30per, axis=0)
    print('acc after remove bottom 30%:', knn_classifier(x_train_, y_train_, x_val, y_val, k=k))

elif task=='train_deep':
    print('train_deep')
    acc_old_ = 0
    for i in range(10):
        u_func = get_ufunc(dataset, model_type, batch_size, lr, verbose, device='cuda:0')
        acc_old = u_func(x_train, y_train, x_val, y_val)
        acc_old_ += acc_old
    print('deep acc before remove:', acc_old_/10)
    # 从训练集中删除这些数据
    valid_indices = np.where(sv1 > 0)[0]
    x_train_ = x_train[valid_indices]
    y_train_ = y_train[valid_indices]
    # 训练一个深度学习模型
    acc_new_ = 0
    for i in range(10):
        u_func_new = get_ufunc(dataset, model_type, batch_size, lr, verbose, device='cuda:0')
        acc_new = u_func_new(x_train_, y_train_, x_val, y_val)
        acc_new_ += acc_new
    print('deep acc after remove:', acc_new_/10)
    # u_func_new = get_ufunc(dataset, model_type, batch_size, lr, verbose, device='cuda:0')
    # acc_new = u_func_new(x_train_, y_train_, x_val, y_val)
    # print('deep acc after remove:', acc_new)

elif task=='remove+classifier':
    print('remove+classifier')
    acc_svm = 0
    acc_lr = 0
    for i in range(10):
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(random_state=0).fit(x_train, y_train)
        acc_lr += clf.score(x_val, y_val)
        from sklearn import svm
        clf = svm.SVC()
        clf.fit(x_train, y_train)
        acc_svm += clf.score(x_val, y_val)
    print('acc of LR before remov:', acc_lr/10)
    print('acc of SVM before remov:', acc_svm/10)
    valid_indices = np.where(sv1 > 0)[0]
    # 使用这些索引来筛选 x_train 和 y_train
    x_train_ = x_train[valid_indices]
    y_train_ = y_train[valid_indices]
    # Logisitic Regression
    acc_svm = 0
    acc_lr = 0
    for i in range(10):
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(random_state=0).fit(x_train_, y_train_)
        acc_lr += clf.score(x_val, y_val)
        
        #SVM
        from sklearn import svm
        clf = svm.SVC()
        clf.fit(x_train_, y_train_)
        acc_svm += clf.score(x_val, y_val)
    print('acc of LR after remov:', acc_lr/10)
    print('acc of SVM after remov:', acc_svm/10)


        
