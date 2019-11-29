import os
import sys
import pickle
import argparse
import numpy as np
np.seterr(over="ignore")
from pathlib import Path
from random import randrange
import matplotlib.pyplot as plt
# from keras.utils import np_utils
# from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
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
	try:
		model = loadPickle(filename)
		return model
	except:
		raise Exception("Model not found: " + filename )

def checkandcreatedir(path):
	if not os.path.isdir(path):
		os.makedirs(path)

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="Path to the dataset")
ap.add_argument("-m1", "--model1", required=False, default="", help="Path to first model (self NN)")
ap.add_argument("-m2", "--model2", required=False, default="", help="Path to second model (sklearn MLP)")
ap.add_argument("-e", "--epochs", required=False, default=1, help="Path to second model (sklearn MLP)")
args = vars(ap.parse_args())

dataset_path = args["dataset"]
model1 = args["model1"]
model2 = args["model2"]
num_epochs = int(args["epochs"])

dataset = datasets.dataset(dataset_path)
data = dataset.loadMNIST()
x_train, x_test, y_train, y_test = train_test_split(data["X"], data["Y"], test_size=0.2, random_state = 0)
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state = 0)

if model1 == "":
	nn = neuralnetwork.NN(x_train, y_train, x_valid, y_valid, x_test, y_test, num_layers = 3, num_neurons = [100, 50, 50], epochs=num_epochs)
	nn.train()
	storemodel(nn, "nn_model")
else:
	nn = loadmodel(model1)

print(nn.train_losses, nn.valid_losses, nn.test_losses)

plt.figure()
plt.plot(nn.train_losses, label="Training Loss")
plt.plot(nn.valid_losses, label="Validation Loss")
plt.plot(nn.test_losses, label="Test Loss")
plt.ylabel("Cross-Entropy loss")
plt.xlabel("Epoch(s)")
plt.legend()
plt.title("Loss v/s Epochs")

plt.figure()
plt.plot(nn.train_acc, label="Training Accuracy")
plt.plot(nn.valid_acc, label="Validation Accuracy")
plt.plot(nn.test_acc, label="Test Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch(s)")
plt.legend()
plt.title("Accuracy v/s Epochs")

plt.figure()
plt.plot(nn.losses)
plt.xlabel("Training step")
plt.ylabel("Cross Entropy loss")
plt.title("Training loss v/s training step")

plt.show()

## Part c
# tsne = TSNE(n_components=2, verbose = 1)
# tsne_results = tsne.fit_transform(nn.W[-1])
# print(tsne_results.T)

# y_test = [0 if i == 7 else 1 for i in y_test]
# y_test = np.array(y_test)

# y_test_cat = np_utils.to_categorical(y_test, num_classes = 2)
# color_map = np.argmax(y_test_cat, axis=1)
# plt.figure(figsize=(10,10))
# for cl in range(2):
# 	indices = np.where(color_map==cl)
# 	indices = indices[0]
# 	print(indices, tsne_results.shape)
# 	plt.scatter(tsne_results[indices,0], tsne_results[indices, 1], label=cl)
# plt.legend()
# plt.show()

## Part d
x_train = np.reshape(x_train, (len(x_train), 28*28))
x_test = np.reshape(x_test, (len(x_test), 28*28))

if model2 == "":
	clf = MLPClassifier(solver='sgd', activation="logistic", hidden_layer_sizes=(100, 50, 50), learning_rate_init=0.1, max_iter=1)
	clf.fit(x_train, y_train)
	storemodel(clf, "sklean_mlp")
else:
	clf = loadmodel(model2)

correct_pred = 0
pred = clf.predict(x_test)
accuracy = accuracy_score(y_test, pred)
print("Accuracy:", accuracy)
