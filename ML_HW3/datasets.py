import h5py
import pickle
import numpy as np
import torch
import torch.utils.data as utils

class dataset:
    def loadCIFAR(self):
        with open(self.datapath, 'rb') as handle:
            data = pickle.load(handle)
        x = data["X"]
        y = data["Y"]
        tensor_x = torch.stack([torch.Tensor(i) for i in x]) # transform to torch tensors
        y_stack = [torch.Tensor([i]) for i in y]
        tensor_y = torch.stack(y_stack)
        dataset = utils.TensorDataset(tensor_x, tensor_y) # create your datset
        dataloader = utils.DataLoader(dataset)
        return dataloader

    def loadMNIST(self):
        data = {}
        with h5py.File(self.datapath, 'r') as f:
            for key in f.keys():
                data[key] = np.array(f[key])
        return data
    
    def __init__(self, datapath):
        self.datapath = datapath
