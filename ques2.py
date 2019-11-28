import os
import pickle
import argparse
import numpy as np
from pathlib import Path

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

ap = argparse.ArgumentParser()
ap.add_argument("-t1", "--train", required=True, help="Path to the train dataset")
ap.add_argument("-t2", "--test", required=True, help="Path to the test dataset")
# ap.add_argument("-m1", "--model1", required=False, default="", help="Path to first model (self NN)")
# ap.add_argument("-m2", "--model2", required=False, default="", help="Path to second model (sklearn MLP)")
args = ap.parse_args()

train_dataset_path = args.train
test_dataset_path = args.test

train_dataset = datasets.dataset(train_dataset_path)
test_dataset = datasets.dataset(test_dataset_path)
train_loader = train_dataset.loadCIFAR()
test_loader = train_dataset.loadCIFAR()

