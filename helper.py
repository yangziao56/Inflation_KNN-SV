from re import T
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
import math

import scipy
from scipy.special import beta, comb
from random import randint

from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from tqdm import tqdm


from utility_func import *

import config



big_dataset = config.big_dataset
OpenML_dataset = config.OpenML_dataset

save_dir = 'result/'



def kmeans_f1score(value_array, cluster=True):

  n_data = len(value_array)

  if cluster:
    X = value_array.reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    min_cluster = min(kmeans.cluster_centers_.reshape(-1))
    pred = np.zeros(n_data)
    pred[value_array < min_cluster] = 1
  else:
    threshold = np.sort(value_array)[int(0.1*n_data)]
    pred = np.zeros(n_data)
    pred[value_array < threshold] = 1
    
  true = np.zeros(n_data)
  true[:int(0.1*n_data)] = 1
  return f1_score(true, pred)

def aucroc(value_array, cluster=False):

  n_data = len(value_array)

  # if cluster:
  #   X = value_array.reshape(-1, 1)
  #   kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
  #   min_cluster = min(kmeans.cluster_centers_.reshape(-1))
  #   pred = np.zeros(n_data)
  #   pred[value_array < min_cluster] = 1
  # else:
  #   threshold = np.sort(value_array)[int(0.1*n_data)]
  #   pred = np.zeros(n_data)
  #   pred[value_array < threshold] = 1
    
  true = np.zeros(n_data)
  true[:int(0.2*n_data)] = 1
  return roc_auc_score( true, - value_array )

def kmeans_aucroc(value_array, cluster=False):

  n_data = len(value_array)

  if cluster:
    X = value_array.reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    min_cluster = min(kmeans.cluster_centers_.reshape(-1))
    pred = np.zeros(n_data)
    pred[value_array < min_cluster] = 1
  else:
    threshold = np.sort(value_array)[int(0.1*n_data)]
    pred = np.zeros(n_data)
    pred[value_array < threshold] = 1
    
  true = np.zeros(n_data)
  true[:int(0.1*n_data)] = 1
  return roc_auc_score(true, pred)


def kmeans_aupr(value_array, cluster=False):

  n_data = len(value_array)

  if cluster:
    X = value_array.reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    min_cluster = min(kmeans.cluster_centers_.reshape(-1))
    pred = np.zeros(n_data)
    pred[value_array < min_cluster] = 1
  else:
    threshold = np.sort(value_array)[int(0.1*n_data)]
    pred = np.zeros(n_data)
    pred[value_array < threshold] = 1
    
  true = np.zeros(n_data)
  true[:int(0.1*n_data)] = 1
  return average_precision_score(true, pred)




"""
def kmeans_f1score(value_array):

  n_data = len(value_array)

  X = value_array.reshape(-1, 1)
  kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
  min_cluster = min(kmeans.cluster_centers_.reshape(-1))
  pred = np.zeros(n_data)
  pred[value_array < min_cluster] = 1
  true = np.zeros(n_data)
  true[:int(0.1*n_data)] = 1
  return f1_score(true, pred)
"""


"""
def load_value_args(value_type, args):

  if args.dataset == 'Dog_vs_CatFeature':

    if args.value_type == 'LeastCore':
      save_name = save_dir + 'Banzhaf_GT_Dog_vs_CatFeature_MLP_Ndata2000_Nval2000_Nsample100000_BS128_Nrepeat5_FR0.1.data'
      value_arg = pickle.load( open(save_name, 'rb') )

      for i, x in enumerate(value_arg['X_feature']):
        value_arg['X_feature'][i] = x[x<args.n_data]
    else:
      save_name = save_dir + '{}_Dog_vs_CatFeature_MLP_Ndata2000_Nval2000_Nsample100000_BS128_Nrepeat5_FR0.1.data'.format( value_type )
      value_arg = pickle.load( open(save_name, 'rb') )

    value_arg['y_feature'] = np.mean( np.array(value_arg['y_feature']), axis=1 )

  else:

    save_name = save_dir + '{}_{}_Logistic_Ndata200_Nval2000_Nsample2000_FR0.1_Seed{}.data'.format(value_type, args.dataset, args.random_state)
    value_arg = pickle.load( open(save_name, 'rb') )
    if 'y_feature' in value_arg.keys():
      value_arg['y_feature'] = np.array(value_arg['y_feature'])

  return value_arg
"""


def load_value_args(value_type, args):
  if value_type == 'BetaShapley':
    base_value_type = 'Shapley_Perm'
  else:
    base_value_type = value_type

  if args.dataset in big_dataset:
    save_name = save_dir+'{}_{}_{}_Ndata{}_Nval{}_Nsample{}_BS{}_LR{}_Nrepeat{}_FR{}_Seed{}.data'.format(
        base_value_type, args.dataset, args.model_type, args.n_data, args.n_val, args.n_sample, args.batch_size, args.lr, 5, args.flip_ratio, args.random_state)
  elif args.dataset in OpenML_dataset:
    save_name = save_dir+'{}_{}_{}_Ndata{}_Nval{}_Nsample{}_BS{}_LR{}_Nrepeat{}_FR{}_Seed{}.data'.format(
        base_value_type, args.dataset, args.model_type, args.n_data, args.n_val, args.n_sample, args.batch_size, args.lr, 5, args.flip_ratio, args.random_state)
  else:
    save_name = save_dir+'{}_{}_{}_Ndata{}_Nval{}_Nsample{}_FR{}.data'.format(
        base_value_type, args.dataset, args.model_type, args.n_data, args.n_val, args.n_sample, args.flip_ratio)

  #print(args.n_sample)
  value_arg = pickle.load( open(save_name, 'rb') )

  if 'y_feature' in value_arg.keys():
    value_arg['y_feature'] = np.array(value_arg['y_feature'])

  if value_type == 'BetaShapley':
    value_arg['alpha'] = args.alpha
    value_arg['beta'] = args.beta

  return value_arg


# args: a dictionary
def compute_value(value_type, args):
  if value_type == 'Shapley_Perm':
    sv = shapley_permsampling_from_data(args['X_feature'], args['y_feature'], args['n_data'], v0=args['sv_baseline'])
  elif value_type == 'BetaShapley':
    sv = betasv_permsampling_from_data(args['X_feature'], args['y_feature'], args['n_data'], args['alpha'], args['beta'], v0=args['sv_baseline'])
  elif value_type == 'Banzhaf_GT':
    sv = banzhaf_grouptest_bias_from_data(args['X_feature'], args['y_feature'], args['n_data'], dummy=True)
  elif value_type == 'LOO':
    sv = compute_loo(args['y_feature'], args['u_total'])
  elif value_type == 'KNN':
    sv = args['knn']
  elif value_type == 'Uniform':
    sv = np.ones(args['n_data'])
  elif value_type == 'Shapley_GT':
    sv = shapley_grouptest_from_data(args['X_feature'], args['y_feature'], args['n_data'])
  elif value_type == 'LeastCore':
    sv = banzhaf_grouptest_bias_from_data(args['X_feature'], args['y_feature'], args['n_data'], dummy=False)
  return sv



def normalize(val):
  v_max, v_min = np.max(val), np.min(val)
  val = (val-v_min) / (v_max - v_min)
  return val


def shapley_permsampling_from_data(X_feature, y_feature, n_data, v0=0.1):

  n_sample = len(y_feature)
  n_perm = int( n_sample // n_data )

  if n_sample%n_data > 0: 
    print('WARNING: n_sample cannot be divided by n_data')

  sv_vector = np.zeros(n_data)
  
  for i in range(n_perm):
    for j in range(0, n_data):
      target_ind = X_feature[i*n_data+j][-1]
      if j==0:
        without_score = v0
      else:
        without_score = y_feature[i*n_data+j-1]
      with_score = y_feature[i*n_data+j]
      
      sv_vector[target_ind] += (with_score-without_score)
  
  return sv_vector / n_perm


def beta_constant(a, b):
    '''
    the second argument (b; beta) should be integer in this function
    '''
    beta_fct_value=1/a
    for i in range(1,b):
        beta_fct_value=beta_fct_value*(i/(a+i))
    return beta_fct_value


def compute_weight_list(m, alpha=1, beta=1):
    '''
    Given a prior distribution (beta distribution (alpha,beta))
    beta_constant(j+1, m-j) = j! (m-j-1)! / (m-1)! / m # which is exactly the Shapley weights.
    # weight_list[n] is a weight when baseline model uses 'n' samples (w^{(n)}(j)*binom{n-1}{j} in the paper).
    '''
    weight_list=np.zeros(m)
    normalizing_constant=1/beta_constant(alpha, beta)
    for j in np.arange(m):
        # when the cardinality of random sets is j
        weight_list[j]=beta_constant(j+alpha, m-j+beta-1)/beta_constant(j+1, m-j)
        weight_list[j]=normalizing_constant*weight_list[j] # we need this '/m' but omit for stability # normalizing
    return weight_list


def betasv_permsampling_from_data(X_feature, y_feature, n_data, a, b, v0=0.1):

  n_sample = len(y_feature)
  n_perm = int( n_sample // n_data )

  if n_sample%n_data > 0: 
    print('WARNING: n_sample cannot be divided by n_data')

  """
  weight_vector = np.zeros(n_data)
  for j in range(1, n_data+1):
    w = n_data * beta(j+b-1, n_data-j+a) / beta(a, b) * comb(n_data-1, j-1)
    weight_vector[j-1] = w
  """
  weight_vector = compute_weight_list(n_data, alpha=a, beta=b)
  #print(weight_vector[:1000])

  sv_vector = np.zeros(n_data)
  
  for i in range(n_perm):
    for j in range(0, n_data):
      target_ind = X_feature[i*n_data+j][-1]
      if j==0:
        without_score = v0
      else:
        without_score = y_feature[i*n_data+j-1]
      with_score = y_feature[i*n_data+j]
      
      sv_vector[target_ind] += weight_vector[j]*(with_score-without_score)

  return sv_vector / n_perm



def banzhaf_grouptest_bias_from_data(X_feature, y_feature, n_data, dummy=True):

  n_sample = len(y_feature)
  if dummy:
    N = n_data+1
  else:
    N = n_data

  A = np.zeros((n_sample, N))
  B = y_feature

  for t in range(n_sample):
    A[t][X_feature[t]] = 1

  sv_approx = np.zeros(n_data)

  for i in range(n_data):
    if np.sum(A[:, i]) == n_sample:
      sv_approx[i] = np.dot( A[:, i], B ) / n_sample
    elif np.sum(A[:, i]) == 0:
      sv_approx[i] = - np.dot( (1-A[:, i]), B ) / n_sample
    else:
      sv_approx[i] = np.dot(A[:, i], B)/np.sum(A[:, i]) - np.dot(1-A[:, i], B)/np.sum(1-A[:, i])

  return sv_approx


def sample_utility_multiple(x_train, y_train, x_test, y_test, utility_func, n_repeat):

  acc_lst = []

  for _ in range(n_repeat):
    acc = utility_func(x_train, y_train, x_test, y_test)
    acc_lst.append(acc)

  return acc_lst


def sample_utility_shapley_perm(n_perm, utility_func, utility_func_args):

  x_train, y_train, x_val, y_val = utility_func_args
  n_data = len(y_train)

  X_feature_test = []
  y_feature_test = []
  
  for k in range(n_perm):

    print('Permutation {} / {}'.format(k, n_perm))
    perm = np.random.permutation(range(n_data))

    for i in range(1, n_data+1):
      subset_index = perm[:i]
      X_feature_test.append(subset_index)
      y_feature_test.append(utility_func(x_train[subset_index], y_train[subset_index], x_val, y_val))

  return X_feature_test, y_feature_test





def compute_utility(args):
    utility_func, x_train, y_train, x_val, y_val, perm = args
    #print("Processing permutation:", perm)
    X_feature_test_local = []
    y_feature_test_local = []
    n_data = len(y_train)
    for i in range(1, n_data+1):
        subset_index = perm[:i]
        X_feature_test_local.append(subset_index)
        y_feature_test_local.append(utility_func(x_train[subset_index], y_train[subset_index], x_val, y_val))
    return X_feature_test_local, y_feature_test_local

import dill as pickle

#multiprocessing.set_start_method('spawn')

    

def sample_utility_shapley_perm_parallel(n_perm, utility_func, utility_func_args):
    
    x_train, y_train, x_val, y_val = utility_func_args
    n_data = len(y_train)
    all_perms = [np.random.permutation(range(n_data)) for _ in range(n_perm)]
    import multiprocessing
    # multiprocessing.set_start_method('spawn', force=True)
    # import torch.multiprocessing as mp
    # mp.set_start_method('spawn', force=True)
    def spawn_method():
      return 'spawn'
    
    multiprocessing.get_start_method = spawn_method()
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    results = pool.map(compute_utility, [(utility_func, x_train, y_train, x_val, y_val, perm) for perm in all_perms])
    pool.close()
    pool.join()
    
    X_feature_test = [x for result in results for x in result[0]]
    y_feature_test = [y for result in results for y in result[1]]

    return X_feature_test, y_feature_test

'''batch
def sample_utility_shapley_perm_parallel(n_perm, utility_func, utility_func_args):
    x_train, y_train, x_val, y_val = utility_func_args
    n_data = len(y_train)

    X_feature_test = []
    y_feature_test = []

    for k in range(n_perm):
        print('Permutation {} / {}'.format(k, n_perm))
        perm = np.random.permutation(range(n_data))
        
        # 对于当前排列，生成所有可能的子集
        subsets_for_perm = [perm[:i] for i in range(1, n_data + 1)]
        max_length = n_data
        subsets_for_perm = [np.pad(perm[:i], (0, max_length - i), mode='constant', constant_values=0) for i in range(1, n_data + 1)]


        # 对于当前排列，生成所有可能的x_train和y_train子集
        x_train_subsets = np.stack([x_train[index] for index in subsets_for_perm])
        y_train_subsets = np.stack([y_train[index] for index in subsets_for_perm])
        
        # 对当前排列的所有子集执行效用函数
        utilities_for_perm = utility_func(x_train_subsets, y_train_subsets, x_val, y_val)
        
        # 将结果添加到X_feature_test和y_feature_test
        X_feature_test += [subset for subset in subsets_for_perm]
        y_feature_test.extend(utilities_for_perm)

    return X_feature_test, y_feature_test
'''

def sample_L_utility_shapley_perm(n_perm, du_model, n_data):

  X_feature_test = []
  y_feature_test = []
  
  for k in range(n_perm):

    print('Permutation {} / {}'.format(k, n_perm))
    perm = np.random.permutation(range(n_data))

    for i in range(1, n_data+1):
      subset_index = perm[:i]
      X_feature_test.append(subset_index)

      subset_bin = np.zeros((1, 200))
      subset_bin[0, subset_index] = 1

      y = du_model(torch.tensor(subset_bin).float().cuda()).cpu().detach().numpy().reshape(-1)
      y_feature_test.append(y[0])

  return X_feature_test, np.array(y_feature_test)




def uniformly_subset_sample(dataset):

  sampled_set = []

  for data in dataset:
    if randint(0, 1) == 1:
      sampled_set.append(data)

  return sampled_set


def sample_utility_banzhaf_mc(n_sample, utility_func, utility_func_args, target_ind):

  x_train, y_train, x_val, y_val = utility_func_args
  n_data = len(y_train)

  n_sample_per_data = int( n_sample / 2 )

  # utility set will store tuples (with/without target index)
  utility_set = []

  dataset = np.arange(n_data)
  leave_one_out_set = np.delete(dataset, target_ind)

  for _ in range(n_sample_per_data):
    sampled_idx_without = np.array(uniformly_subset_sample(leave_one_out_set))
    utility_without = utility_func(x_train[sampled_idx_without], y_train[sampled_idx_without], x_val, y_val)
    sampled_idx_with = np.array( list(sampled_idx_without) + [target_ind] )
    utility_with = utility_func(x_train[sampled_idx_with], y_train[sampled_idx_with], x_val, y_val)

    to_be_store = { 'ind': sampled_idx_without, 'u_without': utility_without, 'u_with': utility_with }

    utility_set.append(to_be_store)

  return utility_set


# Implement Dummy Data Point Idea
def sample_utility_banzhaf_gt(n_sample, utility_func, utility_func_args, dummy=False):

  x_train, y_train, x_val, y_val = utility_func_args
  n_data = len(y_train)

  if dummy:
    N = n_data + 1
  else:
    N = n_data

  X_feature_test = []
  y_feature_test = []

  for t in range(n_sample):

    # Uniformly sample data points from N data points
    subset_ind = np.array(uniformly_subset_sample( np.arange(N) )).astype(int)

    X_feature_test.append(subset_ind)

    if dummy:
      subset_ind = subset_ind[subset_ind < n_data]

    y_feature_test.append( utility_func(x_train[subset_ind], y_train[subset_ind], x_val, y_val) )
  
  return X_feature_test, y_feature_test




# Leave-one-out
def sample_utility_loo(utility_func, utility_func_args):

  x_train, y_train, x_val, y_val = utility_func_args
  n_data = len(y_train)

  N = n_data

  X_feature_test = []
  y_feature_test = []

  u_total = utility_func(x_train, y_train, x_val, y_val)

  for i in range(N):

    loo_index = np.ones(N)
    loo_index[i] = 0
    loo_index = loo_index.nonzero()[0]

    X_feature_test.append( loo_index )
    y_feature_test.append( utility_func(x_train[loo_index], y_train[loo_index], x_val, y_val) )

  return X_feature_test, y_feature_test, u_total


# y_feature is 1-dim array, u_total is scalar
def compute_loo(y_feature, u_total):
  score = np.zeros(len(y_feature))
  for i in range(len(y_feature)):
    score[i] = u_total - y_feature[i]
  return score

def rank_neighbor_train_test(x_test, x_train):
  distance = np.array([np.linalg.norm(x - x_train) for x in x_test])
  return np.argsort(distance)

def rank_neighbor(x_test, x_train):
  distance = np.array([np.linalg.norm(x - x_test) for x in x_train])
  return np.argsort(distance)


# x_test, y_test are single data point
def knn_shapley_single(x_train_few, y_train_few, x_test, y_test, K):
  N = len(y_train_few)
  sv = np.zeros(N)
  rank = rank_neighbor(x_test, x_train_few)
  sv[int(rank[-1])] += int(y_test==y_train_few[int(rank[-1])]) / N

  for j in range(2, N+1):
    i = N+1-j
    sv[int(rank[-j])] = sv[int(rank[-(j-1)])] + ( (int(y_test==y_train_few[int(rank[-j])]) - int(y_test==y_train_few[int(rank[-(j-1)])])) / K ) * min(K, i) / i

  att_point =sv[0]#[58]#[1074]
  #print(att_point)

  import matplotlib.pyplot as plt

  # 假设 sv 和 rank 已经被计算
  # sv = [计算得到的 Shapley 值]
  # rank = rank_neighbor(x_test, x_train_few)

  # 根据 rank 重排 sv
  sv_sorted_by_rank = sv[rank] #10%的数据为正，其余为负

  # positive_count = np.sum(sv_sorted_by_rank > 0)
  # negative_count = np.sum(sv_sorted_by_rank < 0)

  # print(f"正数的数量: {positive_count}")
  # print(f"负数的数量: {negative_count}")
  
  sum_sv = np.sum(sv)
  sum_sv_k = np.sum(sv_sorted_by_rank[:K])
  sum_sv_behind_k = np.sum(sv_sorted_by_rank[K:])
  
  # # 绘制折线图
  # plt.figure(figsize=(10, 6))
  # plt.plot(sv_sorted_by_rank, marker='o') # 使用圆点标记每个点
  # plt.title('Shapley Values Sorted by Distance Rank')
  # plt.xlabel('Rank (from closest to farthest)')
  # plt.ylabel('Shapley Value')
  # plt.grid(True)
  # # 设置坐标轴范围
  # plt.xlim((0,2000))
  # plt.ylim((-0.075,0.075))
  # plt.text(x=0, y=max(sv_sorted_by_rank), s=f"Sum of SV: {sum_sv:.2f}", fontsize=12, color='blue')
  # plt.text(x=100, y=max(sv_sorted_by_rank), s=f"sum_sv_k = {sum_sv_k}", fontsize=12, color='blue')
  # plt.text(x=100, y=max(sv_sorted_by_rank)-0.01, s=f"sum_sv_behind_k = {sum_sv_behind_k}", fontsize=12, color='blue')
  # plt.text(x=100, y=max(sv_sorted_by_rank)-0.02, s=f"sum_sv = {sum_sv}", fontsize=12, color='blue')
  # plt.show()
  

  return sv, att_point, sum_sv


def knn_shapley(x_train_few, y_train_few, x_val_few, y_val_few, K):
  #sprint(x_train_few.shape)
  
  N = len(y_train_few)
  sv = np.zeros(N)
  att = []
  sum_sv = []

  n_test = len(y_val_few)
  sv_matrix = []
  for i in range(n_test):
    x_test, y_test = x_val_few[i], y_val_few[i]
    
    shapley_value, att_point, sum_shapley_value = knn_shapley_single(x_train_few, y_train_few, x_test, y_test, K)
    att.append(att_point)
    sum_sv.append(sum_shapley_value)
    sv += shapley_value
    sv_matrix.append(shapley_value)
  sv_matrix = np.array(sv_matrix)
  #print(sv_matrix.shape) # (200, 800)

  indexed_shapley_values = sorted(enumerate(att), key=lambda x: x[1], reverse=True)
  sorted_indices = [index for index, value in indexed_shapley_values]
  sorted_values = [value for index, value in indexed_shapley_values]
  #print(sorted_indices)
  #print(sorted_values)
  #print(n_test)


  
  # import matplotlib.pyplot as plt
  # import matplotlib.colors as mcolors

  # # 画图
  # indices = np.arange(len(sum_sv))
  # plt.figure(figsize=(12, 6))
  # plt.scatter(indices, sum_sv)
  # plt.title("Line Chart of sum_sv")
  # plt.xlabel("Index")
  # plt.ylabel("Value")
  # plt.grid(True)
  # plt.show()

  # # 将 sum_sv 中的值四舍五入到最近的 0.1
  # rounded_values = np.round(sum_sv, 1)
  # # 计算每个四舍五入后的值的出现次数
  # values, counts = np.unique(rounded_values, return_counts=True)

  # # 绘制柱状图
  # plt.figure(figsize=(12, 6))
  # plt.bar(values, counts, width=0.05)  # 设置柱子的宽度为 0.05 以避免相互重叠
  # plt.title("Frequency of Rounded Values in sum_sv")
  # plt.xlabel("Rounded Value")
  # plt.ylabel("Frequency")
  # plt.xticks(np.arange(0, 1.1, 0.1))  # 设置 x 轴刻度
  # plt.show()

  
  # for j in range(len(y_train_few)):
  #   draw_sv = sv_matrix[:,j]
  #   rank = rank_neighbor_train_test(x_val_few, x_train_few[j])
  #   draw_sv = draw_sv[rank]
  #   plt.figure(figsize=(10, 6))
  #   plt.plot(draw_sv, marker='o') # 使用圆点标记每个点
  #   plt.title('Shapley Values Sorted by Distance Rank')
  #   plt.xlabel('Rank (from closest to farthest)')
  #   plt.ylabel('Shapley Value')
  #   plt.grid(True)
  #   plt.text(x=100, y=max(draw_sv), s=f"sum_sv = {np.sum(draw_sv)}", fontsize=12, color='blue')
  #   plt.show()






  #画出sv热力图
  '''
  # 创建一个从白色到灰色再到黑色的渐变
  # colors = ["white", "grey", "black"]
  # cmap_name = 'my_list'
  # cm = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors, N=256) 

  # 定义一个函数，将小于阈值的部分设置为白色
  def adjust_colors(values, cmap, threshold):
      colors = cmap(np.arange(cmap.N))
      # 设置小于阈值的颜色为白色
      colors[values < threshold, :] = [1, 1, 1, 1]
      new_cmap = mcolors.ListedColormap(colors)
      return new_cmap

  # 创建一个viridis颜色映射
  viridis_cmap = plt.cm.get_cmap('viridis', 256)

  # 调整颜色映射，使得小的值显示为白色
  new_cmap = adjust_colors(np.linspace(0, 1, 256), viridis_cmap, 0.6)
  plt.figure(figsize=(80, 20))
  plt.imshow(sv_matrix, cmap=new_cmap, aspect='auto')
  plt.colorbar(label='Shapley Value')
  plt.xlabel('Training Set Index')
  plt.ylabel('Test Set Index')
  plt.title('Shapley Values Heatmap')
  plt.show()

  # Counting the number of non-white (non-zero) points in each row and column
  non_white_counts_rows = np.count_nonzero(np.array(sv_matrix) >= 0.03, axis=1)
  non_white_counts_columns = np.count_nonzero(np.array(sv_matrix) >= 0.03, axis=0)

  # Plotting the counts for rows and columns
  fig, axes = plt.subplots(2, 1, figsize=(10, 8))

  # Row counts
  axes[0].plot(non_white_counts_rows, color='blue', marker='o')
  axes[0].set_title('Number of Non-White Points in Each Row')
  axes[0].set_xlabel('Row Index')
  axes[0].set_ylabel('Count')

  # Column counts
  axes[1].plot(non_white_counts_columns, color='green', marker='o')
  axes[1].set_title('Number of Non-White Points in Each Column')
  axes[1].set_xlabel('Column Index')
  axes[1].set_ylabel('Count')

  plt.tight_layout()
  plt.show()







  # 绘制直方图
  # 将矩阵转换为一维数组
  sv_matrix = np.array(sv_matrix)
  shapley_values = sv_matrix.flatten()
  plt.figure(figsize=(20, 10))
  plt.hist(shapley_values, bins=30, color='blue', edgecolor='black')  # bins参数决定了区间的数量
  plt.yscale('log')  # 设置纵轴为对数刻度
  plt.title('Distribution of Shapley Values')
  plt.xlabel('Shapley Value')
  plt.ylabel('Frequency')
  plt.show()

  #正态分布
  from scipy.stats import norm

  # 假设 data 是一个形状为 (200, 800) 的 NumPy 数组

  # 计算数据的均值和标准差
  mu, sigma = np.mean(shapley_values), np.std(shapley_values)

  # 生成直方图
  #n, bins, patches = 
  plt.hist(shapley_values, bins=30, density=True, alpha=0.6, color='g')
  #plt.hist(shapley_values, bins=30, color='blue', edgecolor='black') 

  # 添加正态分布曲线
  xmin, xmax = plt.xlim()
  x = np.linspace(xmin, xmax, 100)
  p = norm.pdf(x, mu, sigma)
  plt.plot(x, p, 'k', linewidth=2)

  title = "Fit results: mu = %.2f,  std = %.2f" % (mu, sigma)
  plt.title(title)

  plt.show()

  print(mu, sigma)

  # #画二部图
  # import networkx as nx
  # sorted_indices = np.argsort(shapley_values)

  # # 标记前20%最大的值
  # top_20_percent = sorted_indices[-int(len(shapley_values) * 0.2):]

  # # 创建一个同样大小的零矩阵
  # binary_matrix = np.zeros_like(shapley_values)

  # # 将前20%最大的值的位置设为1
  # binary_matrix[top_20_percent] = 1

  # # 将一维数组重塑回原来的矩阵形状
  # binary_matrix = binary_matrix.reshape(sv_matrix.shape)
  # print(sv_matrix.shape)

  # # 创建双部图
  # B = nx.Graph()

  # # 添加顶点
  # rows = range(sv_matrix.shape[0])  # 行顶点
  # cols = range(sv_matrix.shape[1])  # 列顶点
  # B.add_nodes_from(rows, bipartite=0)
  # B.add_nodes_from(cols, bipartite=1)

  # # 添加边
  # for i in rows:
  #     for j in cols:
  #         if binary_matrix[i, j] == 1:
  #             B.add_edge(i, j)

  # # 绘制双部图
  # # 为了可视化，我们可能只展示部分图
  # pos = {node: [0, i] for i, node in enumerate(rows)}  # 行顶点位置在左侧
  # pos.update({node: [1, i] for i, node in enumerate(cols)})  # 列顶点位置在右侧
  # plt.figure(figsize=(12, 8))
  # color_map = ['green' if node in rows else 'blue' for node in cols]
  # nx.draw(B, pos, with_labels=True, node_size=50, alpha=0.6, edge_color="r", node_color=color_map)
  # plt.title("Bipartite Graph from Binary Matrix")
  # plt.show()
  '''
  

  return sv/n_test, sorted_indices, np.array(sum_sv), sv_matrix/n_test


def knn_t_shapley_single(x_train_few, y_train_few, x_test, y_test, K, T):
  N = len(y_train_few)
  sv = np.zeros(N)
  rank = rank_neighbor(x_test, x_train_few)
  #T = N-2*K  # 您可以根据需要更改这个值
  #T = 1800
  T = T

  # 将 sv 中排名最后 T 个元素的值设置为0
  for i in range(-T, 0):
      sv[rank[i]] = 0

  sv[int(rank[N-T])] = int(y_test==y_train_few[int(rank[N-T])]) / float(N-T)
  # print(int(y_test==y_train_few[int(rank[N-T])]))
  # print(N-T)
  # print(sv[int(rank[N-T])])

  for j in range(T+2, N+1):
    i = N+1-j
    sv[int(rank[-j])] = sv[int(rank[-(j-1)])] +  (int(y_test==y_train_few[int(rank[-j])]) - int(y_test==y_train_few[int(rank[-(j-1)])])) / max(K, i) 
    #sv[int(rank[-j])] = sv[int(rank[-(j-1)])] + ( (int(y_test==y_train_few[int(rank[-j])]) - int(y_test==y_train_few[int(rank[-(j-1)])])) / K ) * i / i
  att_point =sv[0]#[58]#[1074]
  #print(att_point)

  import matplotlib.pyplot as plt

  # 假设 sv 和 rank 已经被计算
  # sv = [计算得到的 Shapley 值]
  # rank = rank_neighbor(x_test, x_train_few)

  # 根据 rank 重排 sv
  sv_sorted_by_rank = sv[rank] #10%的数据为正，其余为负

  # positive_count = np.sum(sv_sorted_by_rank > 0)
  # negative_count = np.sum(sv_sorted_by_rank < 0)

  # print(f"正数的数量: {positive_count}")
  # print(f"负数的数量: {negative_count}")
  
  sum_sv = np.sum(sv)
  sum_sv_k = np.sum(sv_sorted_by_rank[:K])
  sum_sv_behind_k = np.sum(sv_sorted_by_rank[K:])
  
  # # 绘制折线图
  # plt.figure(figsize=(10, 6))
  # plt.plot(sv_sorted_by_rank, marker='o') # 使用圆点标记每个点
  # plt.title('Shapley Values Sorted by Distance Rank')
  # plt.xlabel('Rank (from closest to farthest)')
  # plt.ylabel('Shapley Value')
  # plt.grid(True)
  # # 设置坐标轴范围
  # # plt.xlim((0,2000))
  # # plt.ylim((-0.075,0.075))
  # plt.text(x=0, y=max(sv_sorted_by_rank), s=f"Sum of SV: {sum_sv:.2f}", fontsize=12, color='blue')
  # plt.text(x=100, y=max(sv_sorted_by_rank), s=f"sum_sv_k = {sum_sv_k}", fontsize=12, color='blue')
  # plt.text(x=100, y=max(sv_sorted_by_rank)-0.01, s=f"sum_sv_behind_k = {sum_sv_behind_k}", fontsize=12, color='blue')
  # plt.text(x=100, y=max(sv_sorted_by_rank)-0.02, s=f"sum_sv = {sum_sv}", fontsize=12, color='blue')
  # plt.show()
  

  return sv, att_point, sum_sv


def knn_t_shapley(x_train_few, y_train_few, x_val_few, y_val_few, K, T=0):
  #sprint(x_train_few.shape)
  
  N = len(y_train_few)
  sv = np.zeros(N)
  att = []
  sum_sv = []

  n_test = len(y_val_few)
  sv_matrix = []
  for i in range(n_test):
    x_test, y_test = x_val_few[i], y_val_few[i]
    
    shapley_value, att_point, sum_shapley_value = knn_t_shapley_single(x_train_few, y_train_few, x_test, y_test, K, T)
    att.append(att_point)
    sum_sv.append(sum_shapley_value)
    sv += shapley_value
    sv_matrix.append(shapley_value)
  sv_matrix = np.array(sv_matrix)
  #print(sv_matrix.shape) # (200, 800)

  indexed_shapley_values = sorted(enumerate(att), key=lambda x: x[1], reverse=True)
  sorted_indices = [index for index, value in indexed_shapley_values]
  sorted_values = [value for index, value in indexed_shapley_values]
  #print(sorted_indices)
  #print(sorted_values)
  #print(n_test)


  
  # import matplotlib.pyplot as plt
  # import matplotlib.colors as mcolors

  # # 画图
  # indices = np.arange(len(sum_sv))
  # plt.figure(figsize=(12, 6))
  # plt.scatter(indices, sum_sv)
  # plt.title("Line Chart of sum_sv")
  # plt.xlabel("Index")
  # plt.ylabel("Value")
  # plt.grid(True)
  # plt.show()

  # # 将 sum_sv 中的值四舍五入到最近的 0.1
  # rounded_values = np.round(sum_sv, 1)
  # # 计算每个四舍五入后的值的出现次数
  # values, counts = np.unique(rounded_values, return_counts=True)

  # # 绘制柱状图
  # plt.figure(figsize=(12, 6))
  # plt.bar(values, counts, width=0.05)  # 设置柱子的宽度为 0.05 以避免相互重叠
  # plt.title("Frequency of Rounded Values in sum_sv")
  # plt.xlabel("Rounded Value")
  # plt.ylabel("Frequency")
  # plt.xticks(np.arange(0, 1.1, 0.1))  # 设置 x 轴刻度
  # plt.show()

  
  # for j in range(len(y_train_few)):
  #   draw_sv = sv_matrix[:,j]
  #   rank = rank_neighbor_train_test(x_val_few, x_train_few[j])
  #   draw_sv = draw_sv[rank]
  #   plt.figure(figsize=(10, 6))
  #   plt.plot(draw_sv, marker='o') # 使用圆点标记每个点
  #   plt.title('Shapley Values Sorted by Distance Rank')
  #   plt.xlabel('Rank (from closest to farthest)')
  #   plt.ylabel('Shapley Value')
  #   plt.grid(True)
  #   plt.text(x=100, y=max(draw_sv), s=f"sum_sv = {np.sum(draw_sv)}", fontsize=12, color='blue')
  #   plt.show()






  #画出sv热力图
  '''
  # 创建一个从白色到灰色再到黑色的渐变
  # colors = ["white", "grey", "black"]
  # cmap_name = 'my_list'
  # cm = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors, N=256) 

  # 定义一个函数，将小于阈值的部分设置为白色
  def adjust_colors(values, cmap, threshold):
      colors = cmap(np.arange(cmap.N))
      # 设置小于阈值的颜色为白色
      colors[values < threshold, :] = [1, 1, 1, 1]
      new_cmap = mcolors.ListedColormap(colors)
      return new_cmap

  # 创建一个viridis颜色映射
  viridis_cmap = plt.cm.get_cmap('viridis', 256)

  # 调整颜色映射，使得小的值显示为白色
  new_cmap = adjust_colors(np.linspace(0, 1, 256), viridis_cmap, 0.6)
  plt.figure(figsize=(80, 20))
  plt.imshow(sv_matrix, cmap=new_cmap, aspect='auto')
  plt.colorbar(label='Shapley Value')
  plt.xlabel('Training Set Index')
  plt.ylabel('Test Set Index')
  plt.title('Shapley Values Heatmap')
  plt.show()

  # Counting the number of non-white (non-zero) points in each row and column
  non_white_counts_rows = np.count_nonzero(np.array(sv_matrix) >= 0.03, axis=1)
  non_white_counts_columns = np.count_nonzero(np.array(sv_matrix) >= 0.03, axis=0)

  # Plotting the counts for rows and columns
  fig, axes = plt.subplots(2, 1, figsize=(10, 8))

  # Row counts
  axes[0].plot(non_white_counts_rows, color='blue', marker='o')
  axes[0].set_title('Number of Non-White Points in Each Row')
  axes[0].set_xlabel('Row Index')
  axes[0].set_ylabel('Count')

  # Column counts
  axes[1].plot(non_white_counts_columns, color='green', marker='o')
  axes[1].set_title('Number of Non-White Points in Each Column')
  axes[1].set_xlabel('Column Index')
  axes[1].set_ylabel('Count')

  plt.tight_layout()
  plt.show()







  # 绘制直方图
  # 将矩阵转换为一维数组
  sv_matrix = np.array(sv_matrix)
  shapley_values = sv_matrix.flatten()
  plt.figure(figsize=(20, 10))
  plt.hist(shapley_values, bins=30, color='blue', edgecolor='black')  # bins参数决定了区间的数量
  plt.yscale('log')  # 设置纵轴为对数刻度
  plt.title('Distribution of Shapley Values')
  plt.xlabel('Shapley Value')
  plt.ylabel('Frequency')
  plt.show()

  #正态分布
  from scipy.stats import norm

  # 假设 data 是一个形状为 (200, 800) 的 NumPy 数组

  # 计算数据的均值和标准差
  mu, sigma = np.mean(shapley_values), np.std(shapley_values)

  # 生成直方图
  #n, bins, patches = 
  plt.hist(shapley_values, bins=30, density=True, alpha=0.6, color='g')
  #plt.hist(shapley_values, bins=30, color='blue', edgecolor='black') 

  # 添加正态分布曲线
  xmin, xmax = plt.xlim()
  x = np.linspace(xmin, xmax, 100)
  p = norm.pdf(x, mu, sigma)
  plt.plot(x, p, 'k', linewidth=2)

  title = "Fit results: mu = %.2f,  std = %.2f" % (mu, sigma)
  plt.title(title)

  plt.show()

  print(mu, sigma)

  # #画二部图
  # import networkx as nx
  # sorted_indices = np.argsort(shapley_values)

  # # 标记前20%最大的值
  # top_20_percent = sorted_indices[-int(len(shapley_values) * 0.2):]

  # # 创建一个同样大小的零矩阵
  # binary_matrix = np.zeros_like(shapley_values)

  # # 将前20%最大的值的位置设为1
  # binary_matrix[top_20_percent] = 1

  # # 将一维数组重塑回原来的矩阵形状
  # binary_matrix = binary_matrix.reshape(sv_matrix.shape)
  # print(sv_matrix.shape)

  # # 创建双部图
  # B = nx.Graph()

  # # 添加顶点
  # rows = range(sv_matrix.shape[0])  # 行顶点
  # cols = range(sv_matrix.shape[1])  # 列顶点
  # B.add_nodes_from(rows, bipartite=0)
  # B.add_nodes_from(cols, bipartite=1)

  # # 添加边
  # for i in rows:
  #     for j in cols:
  #         if binary_matrix[i, j] == 1:
  #             B.add_edge(i, j)

  # # 绘制双部图
  # # 为了可视化，我们可能只展示部分图
  # pos = {node: [0, i] for i, node in enumerate(rows)}  # 行顶点位置在左侧
  # pos.update({node: [1, i] for i, node in enumerate(cols)})  # 列顶点位置在右侧
  # plt.figure(figsize=(12, 8))
  # color_map = ['green' if node in rows else 'blue' for node in cols]
  # nx.draw(B, pos, with_labels=True, node_size=50, alpha=0.6, edge_color="r", node_color=color_map)
  # plt.title("Bipartite Graph from Binary Matrix")
  # plt.show()
  '''
  

  return sv/n_test, sorted_indices, np.array(sum_sv), sv_matrix/n_test


#KL散度
from scipy.stats import entropy
def flatten_images(x):
    """ 将图像数据从二维转换为一维 """
    return x.reshape(x.shape[0], -1)

def knn_distances(xTrain, xVal, k):
    """ 计算KNN距离 """
    dists = -2 * np.dot(xVal, xTrain.T) + np.sum(xTrain**2, axis=1) + np.sum(xVal**2, axis=1)[:, np.newaxis]
    nearest = np.argsort(dists, axis=1)[:, :k]
    return nearest

def class_probabilities(nearest, yTrain, num_classes, smoothing=1e-10):
    counts = np.apply_along_axis(lambda x: np.bincount(x, minlength=num_classes), axis=1, arr=yTrain[nearest])
    # 添加平滑处理
    smoothed_counts = counts + smoothing
    probabilities = smoothed_counts / smoothed_counts.sum(axis=1, keepdims=True)
    return probabilities


def kl_divergence(p, q):
    """ 计算KL散度 """
    return entropy(p, q)


# def knn_shapley_single_att(x_train_few, y_train_few, x_test, y_test, K):
#   N = 1
#   sv = np.zeros(N)
#   rank = rank_neighbor(x_test, x_train_few)
#   #sv[int(rank[-1])] += int(y_test==y_train_few[int(rank[-1])]) / N

#   for j in range(2, N+1):
#     i = N+1-j
#     sv[int(rank[-j])] = sv[int(rank[-(j-1)])] + ( (int(y_test==y_train_few[int(rank[-j])]) - int(y_test==y_train_few[int(rank[-(j-1)])])) / K ) * min(K, i) / i

#   return sv


# def knn_shapley_att(x_train_few, y_train_few, x_val_few, y_val_few, K):
  
#   N = len(y_train_few)
#   svs = []

#   n_test = len(y_val_few)
#   for i in range(n_test):
#     x_test, y_test = x_val_few[i], y_val_few[i]
#     sv = knn_shapley_single_att(x_train_few[1074], y_train_few[1074], x_test, y_test, K)
#     svs.append(sv)

#   indexed_shapley_values = sorted(enumerate(svs), key=lambda x: x[1], reverse=True)

#   # 分别提取排序后的值和索引
#   sorted_indices = [index for index, value in indexed_shapley_values]
#   sorted_values = [value for index, value in indexed_shapley_values]
#   return sorted_indices, sorted_values

'''
def knn_classifier(x_train, y_train, x_val, y_val, k=10):
    """
    A simple KNN classifier using NumPy.
    :param x_train: Training data, numpy array of shape (n_samples, height, width, channels)
    :param y_train: Training labels, numpy array of shape (n_samples,)
    :param x_val: Validation data, numpy array of shape (n_samples, height, width, channels)
    :param y_val: Validation labels, numpy array of shape (n_samples,)
    :param k: Number of neighbors to use for k-nearest neighbors
    :return: Accuracy of the classifier on the validation set
    """
    predictions = []

    for i in range(len(x_val)):
        # Calculate distances from the validation sample to all training samples
        distances = np.sqrt(np.sum((x_train - x_val[i])**2, axis=(1, 2, 3)))

        # Find the indices of the k nearest neighbors
        k_indices = np.argsort(distances)[:k]

        # Get the labels of the nearest neighbors
        k_labels = y_train[k_indices]

        # Vote for the most common label
        votes = np.bincount(k_labels)
        predicted_label = np.argmax(votes)
        predictions.append(predicted_label)

    # Calculate accuracy
    accuracy = np.mean(np.array(predictions) == y_val)
    return accuracy
'''

def knn_classifier(x_train, y_train, x_val, y_val, k=10):
    """
    An adaptable KNN classifier using NumPy, works for both multi-dimensional image data
    and flattened feature data.
    :param x_train: Training data, numpy array of shape (n_samples, height, width, channels) or (n_samples, features)
    :param y_train: Training labels, numpy array of shape (n_samples,)
    :param x_val: Validation data, numpy array of shape (n_samples, height, width, channels) or (n_samples, features)
    :param y_val: Validation labels, numpy array of shape (n_samples,)
    :param k: Number of neighbors to use for k-nearest neighbors
    :return: Accuracy of the classifier on the validation set
    """
    predictions = []

    # Check the number of dimensions in the training data
    if x_train.ndim == 4:
        # Multi-dimensional image data
        axis = (1, 2, 3)
    elif x_train.ndim == 2:
        # Flattened feature data
        axis = 1
    else:
        raise ValueError("Unsupported shape of training data")

    for i in range(len(x_val)):
        # Calculate distances from the validation sample to all training samples
        distances = np.sqrt(np.sum((x_train - x_val[i])**2, axis=axis))

        # Find the indices of the k nearest neighbors
        k_indices = np.argsort(distances)[:k]

        # Get the labels of the nearest neighbors
        k_labels = y_train[k_indices]

        # Vote for the most common label
        votes = np.bincount(k_labels)
        # if votes.size == 0:
        #   # Handle the empty votes case (e.g., skip or assign a default label)
        #   continue
        # else:
        #   predicted_label = np.argmax(votes)
        #   predictions.append(predicted_label)

        predicted_label = np.argmax(votes)
        predictions.append(predicted_label)

    # Calculate accuracy
    accuracy = np.mean(np.array(predictions) == y_val)
    return accuracy


def weighted_knn_classifier(x_train, y_train, x_val, y_val, k=10, sample_weights=None):
    """
    An adaptable KNN classifier using NumPy, works for both multi-dimensional image data
    and flattened feature data. It uses fixed weights for each training sample.
    :param x_train: Training data, numpy array of shape (n_samples, height, width, channels) or (n_samples, features)
    :param y_train: Training labels, numpy array of shape (n_samples,)
    :param x_val: Validation data, numpy array of shape (n_samples, height, width, channels) or (n_samples, features)
    :param y_val: Validation labels, numpy array of shape (n_samples,)
    :param k: Number of neighbors to use for k-nearest neighbors
    :param sample_weights: Weights for each training sample, numpy array of shape (n_samples,)
    :return: Accuracy of the classifier on the validation set
    """
    predictions = []

    # Check the number of dimensions in the training data
    if x_train.ndim == 4:
        # Multi-dimensional image data
        axis = (1, 2, 3)
    elif x_train.ndim == 2:
        # Flattened feature data
        axis = 1
    else:
        raise ValueError("Unsupported shape of training data")

    # If no sample weights are given, use uniform weights
    if sample_weights is None:
        sample_weights = np.ones(len(x_train))

    for i in range(len(x_val)):
        # Calculate distances from the validation sample to all training samples
        distances = np.sqrt(np.sum((x_train - x_val[i])**2, axis=axis))

        # Find the indices of the k nearest neighbors
        k_indices = np.argsort(distances)[:k]

        # Get the labels and weights of the nearest neighbors
        k_labels = y_train[k_indices]
        k_weights = sample_weights[k_indices]

        # Weighted vote
        unique_labels, counts = np.unique(k_labels, return_counts=True)
        label_weights = np.zeros_like(unique_labels, dtype=float)

        for idx, label in enumerate(unique_labels):
            label_indices = np.where(k_labels == label)
            label_weights[idx] = np.sum(k_weights[label_indices])

        predicted_label = unique_labels[np.argmax(label_weights)]
        predictions.append(predicted_label)

    # Calculate accuracy
    accuracy = np.mean(np.array(predictions) == y_val)
    return accuracy

def tnn_shapley_single(x_train_few, y_train_few, x_test, y_test, tau=0, K0=10, dis_metric='cosine'):

  N = len(y_train_few)
  sv = np.zeros(N)

  C = max(y_train_few)+1
  if dis_metric == 'cosine':
    distance = -np.dot(x_train_few, x_test) / np.linalg.norm( x_train_few, axis=1 )
  else:
    distance = np.array([np.linalg.norm(x - x_test) for x in x_train_few])
  Itau = (distance < tau).nonzero()[0]

  Ct = len(Itau)
  Ca = np.sum( y_train_few[Itau] == y_test )

  reusable_sum = 0
  stable_ratio = 1
  for j in range(N):
    stable_ratio *= (N-j-Ct) / (N-j)
    reusable_sum += (1/(j+1)) * (1 - stable_ratio)
    # reusable_sum += (1/(j+1)) * (1 - comb(N-1-j, Ct) / comb(N, Ct))

  for i in Itau:
    sv[i] = ( int(y_test==y_train_few[i]) - 1/C ) / Ct
    if Ct >= 2:
      ca = Ca - int(y_test==y_train_few[i])
      sv[i] += ( int(y_test==y_train_few[i])/Ct - ca/(Ct*(Ct-1)) ) * ( reusable_sum - 1 )

  return sv

def tnn_shapley(x_train_few, y_train_few, x_val_few, y_val_few, tau=0, K0=10, dis_metric='cosine'):
  
  N = len(y_train_few)
  sv = np.zeros(N)
  n_test = len(y_val_few)
  print('tau in tnn shapley', tau)
  for i in tqdm(range(n_test)):
    x_test, y_test = x_val_few[i], y_val_few[i]
    sv += tnn_shapley_single(x_train_few, y_train_few, x_test, y_test, tau, K0, dis_metric=dis_metric)

  return sv

# x_test, y_test are single data point
def knn_shapley_JW_single(x_train_few, y_train_few, x_test, y_test, K):
  N = len(y_train_few)
  sv = np.zeros(N)
  rank = rank_neighbor(x_test, x_train_few).astype(int)
  C = max(y_train_few)+1

  c_A = np.sum( y_test==y_train_few[rank[:N-1]] )

  const = np.sum([ 1/j for j in range(1, min(K, N)+1) ])

  sv[rank[-1]] = (int(y_test==y_train_few[rank[-1]]) - c_A/(N-1)) / N * ( np.sum([ 1/(j+1) for j in range(1, min(K, N)) ]) ) + (int(y_test==y_train_few[rank[-1]]) - 1/C) / N

  for j in range(2, N+1):
    i = N+1-j
    coef = (int(y_test==y_train_few[int(rank[-j])]) - int(y_test==y_train_few[int(rank[-(j-1)])])) / (N-1)

    sum_K3 = K

    sv[int(rank[-j])] = sv[int(rank[-(j-1)])] + coef * ( const + int( N >= K ) / K * ( min(i, K)*(N-1)/i - sum_K3 ) )

  return sv


# Soft-label KNN-Shapley proposed in https://arxiv.org/abs/2304.04258 
def knn_shapley_JW(x_train_few, y_train_few, x_val_few, y_val_few, K):
  
  N = len(y_train_few)
  sv = np.zeros(N)

  n_test = len(y_val_few)
  for i in range(n_test):
    x_test, y_test = x_val_few[i], y_val_few[i]
    sv += knn_shapley_JW_single(x_train_few, y_train_few, x_test, y_test, K)

  return sv/n_test



def uniformly_subset_givensize(dataset, size):

  sampled_set = np.random.permutation(dataset)

  return sampled_set[:int(size)]


def sample_utility_givensize(n_sample_lst, utility_func, utility_func_args):

  x_train, y_train, x_val, y_val = utility_func_args
  n_data = len(y_train)

  X_feature_test = []
  y_feature_test = []

  for size in n_sample_lst:

    subset_ind = np.array(uniformly_subset_givensize( np.arange(n_data), size )).astype(int)

    X_feature_test.append(subset_ind)

    y_feature_test.append( utility_func(x_train[subset_ind], y_train[subset_ind], x_val, y_val) )
  
  return X_feature_test, y_feature_test


# Implement Dummy Data Point Idea
def sample_utility_shapley_gt(n_sample, utility_func, utility_func_args):

  x_train, y_train, x_val, y_val = utility_func_args
  n_data = len(y_train)

  N = n_data + 1
  Z = np.sum([1/k+1/(N-k) for k in range(1, N)])
  q = [1/Z * (1/k+1/(N-k)) for k in range(1, N)]

  X_feature_test = []
  y_feature_test = []

  for t in range(n_sample):
    # Randomly sample size from 1,...,N-1
    size = np.random.choice(np.arange(1, N), p=q)

    # Uniformly sample k data points from N data points
    subset_ind = np.random.choice(np.arange(N), size, replace=False)

    X_feature_test.append(subset_ind)

    subset_ind = subset_ind[subset_ind < n_data]

    if size == 0:
      y_feature_test.append( [0.1] )
    else:
      y_feature_test.append( utility_func(x_train[subset_ind], y_train[subset_ind], x_val, y_val) )
  
  return X_feature_test, y_feature_test


# Implement Dummy Data Point Idea
def sample_L_utility_shapley_gt(n_sample, du_model, n_data):

  N = n_data + 1
  Z = np.sum([1/k+1/(N-k) for k in range(1, N)])
  q = [1/Z * (1/k+1/(N-k)) for k in range(1, N)]

  X_feature_test = []
  y_feature_test = []

  for t in range(n_sample):
    # Randomly sample size from 1,...,N-1
    size = np.random.choice(np.arange(1, N), p=q)

    # Uniformly sample k data points from N data points
    subset_ind = np.random.choice(np.arange(N), size, replace=False)

    X_feature_test.append(subset_ind)

    subset_ind = subset_ind[subset_ind < n_data]
    subset_bin = np.zeros((1, n_data))
    subset_bin[0, subset_ind] = 1

    y = du_model(torch.tensor(subset_bin).float().cuda()).cpu().detach().numpy().reshape(-1)
    y_feature_test.append(y[0])

  return X_feature_test, np.array(y_feature_test)


# Implement Dummy Data Point Idea
def sample_L_utility_banzhaf_gt(n_sample, du_model, n_data, dummy=False):

  if dummy:
    N = n_data + 1
  else:
    N = n_data

  X_feature_test = []
  y_feature_test = []

  for t in range(n_sample):

    # Uniformly sample data points from N data points
    subset_ind = np.array(uniformly_subset_sample( np.arange(N) )).astype(int)

    X_feature_test.append(subset_ind)

    if dummy:
      subset_ind = subset_ind[subset_ind < n_data]

    subset_bin = np.zeros((1, n_data))
    subset_bin[0, subset_ind] = 1

    y = du_model(torch.tensor(subset_bin).float().cuda()).cpu().detach().numpy().reshape(-1)
    y_feature_test.append(y[0])

  return X_feature_test, np.array(y_feature_test)



def shapley_grouptest_from_data(X_feature, y_feature, n_data):

  n_sample = len(y_feature)
  N = n_data+1
  Z = np.sum([1/k+1/(N-k) for k in range(1, N)])

  A = np.zeros((n_sample, N))
  B = y_feature

  for t in range(n_sample):
    A[t][X_feature[t]] = 1

  C = {}
  for i in range(N):
    for j in [n_data]:
      C[(i,j)] = Z*(B.dot(A[:,i] - A[:,j]))/n_sample

  sv_last = 0
  sv_approx = np.zeros(n_data)

  for i in range(n_data): 
    sv_approx[i] = C[(i, N-1)] + sv_last
  
  return sv_approx



def poison_attack(x_train, y_train, x_val, y_val, model, u_func, attack_idx=114, t=0.01, epsilon=1e-6, random_seed=42):
    """
    A simplified gradient-based poisoning attack on a sample in x_train.

    Args:
    - x_train, y_train: Training data and labels.
    - x_val, y_val: Validation data and labels.
    - model: The neural network model being attacked.
    - u_func: The utility function that measures the performance of the model.
    - attack_idx: Index of the data point in x_train to be poisoned.
    - t: Step size for gradient ascent.
    - epsilon: Convergence threshold.

    Returns:
    - x_train: The training data with the attacked point modified.
    """
    # import multiprocessing
    # multiprocessing.set_start_method('spawn', force=True)
    # import torch.multiprocessing as mp
    # mp.set_start_method('spawn', force=True)

    device = 'cuda:1' #if torch.cuda.is_available() else 'cpu'
    
    criterion = torch.nn.CrossEntropyLoss()
    
    # Make sure gradients can be computed on x_train
    x_train = torch.tensor(x_train, device=device, requires_grad=True, dtype=torch.float32)
    y_train = torch.tensor(y_train, device=device, dtype=torch.long)
    x_val = torch.tensor(x_val, device=device, dtype=torch.float32)
    y_val = torch.tensor(y_val, device=device, dtype=torch.long)
    
    # Initial utility
    #u_prev, _ = u_func(x_train, y_train, x_val, y_val)
    u_prev = []
    for j in range(10):
        u_prev_, _ = u_func(x_train, y_train, x_val, y_val)
        u_prev.append( u_prev_ )
    u_prev = np.mean(u_prev_)
    
    print(u_prev)
    model_old = model
    for i in range(20):


    #while True:  
      model.zero_grad()
      # Set random seed for determinism in u_func
      torch.manual_seed(random_seed)
      np.random.seed(random_seed)

      # Ensure gradients can be computed
      #x_train = x_train.clone().detach().requires_grad_(True)
      #x_train.retain_grad()
      #attacked_data = x_train[attack_idx].clone().detach().requires_grad_(True)

      model.train()
      # Zero-out all existing gradients in the model
      
      # Forward pass
      outputs = model(x_train)
      loss = criterion(outputs, y_train)
      # print(loss)

      # lambda_reg = 1e-5
      # loss += lambda_reg * torch.norm(x_train[attack_idx])

      
      
      # Compute gradients
      loss.backward()
      
      # Extract gradients for the attacked data point
      #gradient = x_train[attack_idx].grad
      gradient = x_train.grad[attack_idx].sign()

      # Check if gradient is not None
      if gradient is None:
          raise ValueError("Gradient for the attacked data point is None. Check the backward pass.")
      
      #Normalize the gradient to unit length
      # norm = torch.norm(gradient)
      # if norm != 0:
      #     unit_gradient = gradient / norm
      # else:
      #     unit_gradient = gradient

      # 使用非原地操作来更新攻击点

      with torch.no_grad():
          x_train[attack_idx] += t * gradient



      # Compute new utility
      #model.eval()
      u_new, _ = u_func(x_train, y_train, x_val, y_val)
      u_new = []
      for j in range(10):
          u_new_, _ = u_func(x_train, y_train, x_val, y_val)
          u_new.append( u_new_ )
      u_new = np.mean(u_new)
      

      # Convergence check
      # if abs(u_new - u_prev) < epsilon:
      #   #print(u_new-u_prev)
      #   break
      # else:
      #   #x_train = x_train_
      #   #print(u_new-u_prev)
      #   print(u_new)
      print(u_new)
      #if math.sqrt(sum((a - b)**2 for a, b in zip(u_new, u_prev))):
      #    break

      u_prev = u_new



    model.zero_grad()

    return x_train.detach().cpu().numpy()