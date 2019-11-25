import h5py
import numpy as np
import pickle

class dataset:
    def loadCIFAR(self):
        with open(self.datapath, 'rb') as handle:
            b = pickle.load(handle)
        return b
    
    def loadMNIST(self):
        data = {}
        with h5py.File(self.datapath, 'r') as f:
            for key in f.keys():
                data[key] = np.array(f[key])
        return data
    
    def __init__(self, datapath):
        self.datapath = datapath
