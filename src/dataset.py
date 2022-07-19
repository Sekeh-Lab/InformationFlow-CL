"""Functions for loading and preparing the training and test data. Adapted from https://github.com/arunmallya/packnet/blob/master/src/dataset.py"""

import collections
import glob
import os

import numpy as np
from PIL import Image

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import DataGenerator as DG


def train_loader(dataset, path, batch_size, num_workers=4, pin_memory=False, normalize=None):
    if dataset == 'CIFAR10':
        t = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    else:
        t = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
        
    x_train = np.load(os.path.join(path, "X.npy"))
    y_train = np.load(os.path.join(path, "y.npy"))

    # map label to 0-19
    max_label = np.max(y_train)
    if max_label > 19:
        y_train = y_train - (max_label-19)

    x_train = torch.from_numpy(x_train).float()
    y_train = torch.from_numpy(y_train)

    x_train = t(x_train)

    ### Makes a custom dataset for CIFAR through torch
    training_set = DG.CifarDataGenerator(x_train, y_train)

    ### Loads the custom data into the dataloader
    return data.DataLoader(training_set, batch_size = batch_size, shuffle = True, num_workers = num_workers, pin_memory=pin_memory)


def test_loader(dataset, path, batch_size, num_workers=4, pin_memory=False, normalize=None):
    if dataset == 'CIFAR10':
        t = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    else:
        t = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
        
    x_test = np.load(os.path.join(path, "X.npy"))
    y_test = np.load(os.path.join(path, "y.npy"))

    # map label to 0-19
    max_label = np.max(y_test)
    if max_label > 19:
        y_test = y_test - (max_label-19)

    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test)
    
    x_test = t(x_test)

    test_set = DG.CifarDataGenerator(x_test, y_test)
    return data.DataLoader(test_set, batch_size = batch_size, shuffle = False, num_workers = num_workers, pin_memory=pin_memory)
