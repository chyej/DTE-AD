import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import collections
import numbers
import math
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle


class MSLSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        #self.scaler = StandardScaler()
        data = np.load(data_path + "/MSL_train.npy")
        #self.scaler.fit(data)
        #data = self.scaler.transform(data)
        self.test = np.load(data_path + "/MSL_test.npy")
        #self.test = self.scaler.transform(test_data)
        data_len = len(data)
        self.train = data[:(int)(data_len * 0.8)]
        self.val = data[(int)(data_len * 0.8):]
        test_labels = np.load(data_path + "/MSL_test_label.npy")
        self.test_labels = np.repeat(test_labels[:, np.newaxis], 55, axis=1)


        print(f"test: {self.test.shape}")
        print(f"train: {self.train.shape}")
        print(f"labels: {self.test_labels.shape}")

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMAPSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        #self.scaler = StandardScaler()
        data = np.load(data_path + "/SMAP_train.npy")
        #self.scaler.fit(data)
        #data = self.scaler.transform(data)
        self.test = np.load(data_path + "/SMAP_test.npy")
        #self.test = self.scaler.transform(test_data)
        data_len = len(data)
        self.train = data[:(int)(data_len * 0.8)]
        self.val = data[(int)(data_len * 0.8):]
        test_labels = np.load(data_path + "/SMAP_test_label.npy")
        self.test_labels = np.repeat(test_labels[:, np.newaxis], 25, axis=1)

        print(f"test: {self.test.shape}")
        print(f"train: {self.train.shape}")
        print(f"labels: {self.test_labels.shape}")

    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SWaTSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/SWaT_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SWaT_test.npy")
        self.test = self.scaler.transform(test_data)
        #self.val = self.test
        test_labels = np.load(data_path + "/SWaT_labels.npy")
        data_len = len(data)
        self.train = data[:(int)(data_len * 0.8)]
        self.val = data[(int)(data_len * 0.8):]
        self.test_labels = np.repeat(test_labels[:, np.newaxis], 51, axis=1)
        np.save('swatlabels.npy', self.test_labels)

        print(f"test: {self.test.shape}")
        print(f"train: {self.train.shape}")
        print(f"labels: {self.test_labels.shape}")

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


def get_loader_segment(data_path, batch_size, win_size, mode='train', dataset='KDD'):
    if (mode == 'test'): step = win_size
    else: step = 1

    if (dataset == 'MSL'):
        dataset = MSLSegLoader(data_path, win_size, step, mode)
        plot_train = dataset.train
        plot_test = dataset.test
    elif (dataset == 'SMAP'):
        dataset = SMAPSegLoader(data_path, win_size, step, mode)
        plot_train = dataset.train
        plot_test = dataset.test
    elif (dataset == 'SWaT'):
        dataset = SWaTSegLoader(data_path, win_size, step, mode)
        plot_train = dataset.train
        plot_test = dataset.test

    shuffle = False
    if mode == 'train':
        shuffle = True

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=0,
                             drop_last=False)
                             

    return data_loader, plot_train, plot_test
