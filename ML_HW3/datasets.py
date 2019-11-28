import cv2
import h5py
import pickle
import numpy as np
import progressbar
import matplotlib.pyplot as plt

import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

class dataset:
    def loadCIFAR(self):
        trainTransform  = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                            ])
        train_dataset = datasets.ImageFolder(
            root=self.datapath,
            transform=trainTransform
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batchsize,
            num_workers=4,
            shuffle=True
        )
        return train_loader, train_dataset

    def loadMNIST(self):
        data = {}
        with h5py.File(self.datapath, 'r') as f:
            for key in f.keys():
                data[key] = np.array(f[key])
        return data
    
    def __init__(self, datapath, batchsize=1):
        self.datapath = datapath
        self.batchsize = 1
