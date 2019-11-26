import os
import sys
import pickle
import numpy as np
np.seterr(over="ignore")
from pathlib import Path
from random import randrange
from sklearn.model_selection import train_test_split

from ML_HW3 import datasets
from ML_HW3 import neuralnetwork

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
	root = str(Path(__file__).parent.parent)
	modeldir = root + "/models"
	filename = modeldir + "/" + filename
	try:
		model = loadPickle(filename)
		return model
	except:
		raise Exception("Model not found: " + filename )

def checkandcreatedir(path):
	if not os.path.isdir(path):
		os.makedirs(path)

dataset = datasets.dataset(sys.argv[1])
data = dataset.loadMNIST()
x_train, x_test, y_train, y_test = train_test_split(data["X"], data["Y"], test_size=0.2)
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2)

model_dir = -1
if len(sys.argv) == 3:
	model_dir = sys.argv[2]

if model_dir == -1:
	nn = neuralnetwork.NN(x_train, y_train, num_layers = 3, num_neurons = [100, 50, 50])
	nn.train()
	storemodel(nn, "nn_model")
else:
	nn = loadmodel("nn_model")

correct_pred = 0
for x, y in zip(x_test, y_test):
	pred = nn.predict(x)
	if pred == y:
		correct_pred += 1
# print accuracy
print("Accuracy:", correct_pred/x_test.shape[0])
