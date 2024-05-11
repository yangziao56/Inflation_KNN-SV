#!/bin/bash

# value_type CKNN, TNN, KNN
#python sv.py --dataset MNIST --value_type CKNN --n_data 50000 --n_val 5000 --K 10 --T 49980
#python sv.py --dataset FMNIST --value_type CKNN --n_data 50000 --n_val 5000 --K 10 --T 49980
python sv.py --dataset CIFAR10 --value_type CKNN --n_data 50000 --n_val 5000 --K 10 --T 49980

python sv.py --dataset pol --value_type CKNN --n_data 2000 --n_val 200 --K 10 --T 1980
python sv.py --dataset wind --value_type CKNN --n_data 2000 --n_val 200 --K 10 --T 1980
python sv.py --dataset cpu --value_type CKNN --n_data 2000 --n_val 200 --K 10 --T 1980

python sv.py --dataset AG_NEWS --value_type CKNN --n_data 10000 --n_val 1000 --K 10 --T 9980
python sv.py --dataset SST-2 --value_type CKNN --n_data 10000 --n_val 1000 --K 10 --T 9980
python sv.py --dataset 20news --value_type CKNN --n_data 10000 --n_val 1000 --K 10 --T 9980
