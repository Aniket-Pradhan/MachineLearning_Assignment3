import os
import sys
import time
import pickle
import argparse
import progressbar
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import models

from ML_HW3 import datasets

def savePickle(filename, data):
	with open(filename, 'wb') as file:
		pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

def loadPickle(filename):
	with open(filename, 'rb') as handle:
		b = pickle.load(handle)
	return b

def storemodel(model, name):
	root = str(Path(__file__).parent.parent)
	modeldir = root + "/models"
	checkandcreatedir(modeldir)
	filepath = modeldir + "/" + name
	savePickle(filepath, model)
	print("saved", filepath)

def loadmodel(filename):
	try:
		model = loadPickle(filename)
		return model
	except:
		raise Exception("Model not found: " + filename )

def checkandcreatedir(path):
	if not os.path.isdir(path):
		os.makedirs(path)

test = "test"
storemodel(test, "test_model")

ap = argparse.ArgumentParser()
ap.add_argument("-t1", "--train", required=False, default="", help="Path to the train dataset")
ap.add_argument("-t2", "--test", required=False, default="", help="Path to the test dataset")
# ap.add_argument("-o", "--out", required=True, help="Output path to the features")
args = ap.parse_args()

train_dataset_path = args.train if args.train != "" else False
test_dataset_path = args.test if args.test != "" else False

if train_dataset_path:
    train_data = datasets.dataset(train_dataset_path)
    train_loader, train_dataset = train_data.loadCIFAR()
if test_dataset_path:
    test_data = datasets.dataset(test_dataset_path)    
    test_loader, test_dataset = test_data.loadCIFAR()

alex = models.alexnet(pretrained=True)

if torch.cuda.is_available():
	device = torch.device("cuda:0")
	cudnn.benchmark = True
else:
	device = torch.device("cpu")

model = nn.Sequential(*list(alex.classifier.children()))
alex.classifier = model

if train_dataset_path:
    train_features_x = []
    train_y = []
    start = time.time()
    for batch_num, (data, target) in progressbar.progressbar(enumerate(train_loader)):
        data, target = data.to(device), target.to(device)
        output = alex.forward(data)
        train_features_x.append(output.detach().numpy())
        train_y.append(target.detach().numpy()[0])

    train_features_x = np.array(train_features_x)
    train_y = np.array(train_y)
    traindata = {"X": train_features_x, "Y": train_y}
    storemodel(traindata, "cifar_features_train")
    print('It took', time.time()-start, 'seconds.')

if test_dataset_path:
    test_features_x = []
    test_y = []
    start = time.time()
    for batch_num, (data, target) in progressbar.progressbar(enumerate(test_loader)):
        data, target = data.to(device), target.to(device)
        output = alex.forward(data)
        test_features_x.append(output.detach().numpy())
        test_y.append(target.detach().numpy()[0])
    test_features_x = np.array(test_features_x)
    test_y = np.array(test_y)
    testdata = {"X": test_features_x, "Y": test_y}
    storemodel(testdata, "cifar_features_test")
    print('It took', time.time()-start, 'seconds.')
