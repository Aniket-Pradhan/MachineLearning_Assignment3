import os
import pickle
import argparse
import numpy as np
import progressbar
from pathlib import Path
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

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
args = ap.parse_args()

train_dataset_path = args.train
test_dataset_path = args.test

train_data = datasets.dataset(train_dataset_path)
test_data = datasets.dataset(test_dataset_path)
train_loader, train_dataset = train_data.loadCIFAR()
test_loader, test_dataset = train_data.loadCIFAR()


"""
Visualize the dataset
# plt.imshow(image[0].numpy().astype('uint8'))
# plt.show()
# exit()
"""

if torch.cuda.is_available():
	device = torch.device("cuda:0")
	cudnn.benchmark = True
else:
	device = torch.device("cpu")

print(device)

model = torch.hub.load('pytorch/vision:v0.4.2', 'alexnet', pretrained=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times
	print("Epoch:", epoch+1)
	print("\ntrain:")
	model.train()
	train_loss = 0
	train_correct = 0
	total = 0
	for batch_num, (data, target) in progressbar.progressbar(enumerate(train_loader)):
		data, target = data.to(device), target.to(device)
		optimizer.zero_grad()
		output = model(data)
		loss = criterion(output, target)
		loss.backward()
		optimizer.step()
		train_loss += loss.item()
		prediction = torch.max(output, 1)  # second param "1" represents the dimension to be reduced
		total += target.size(0)

		# train_correct incremented by one if predicted right
		train_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())
		if batch_num == 1:
			storemodel(model, "test_model")

	print(train_loss, train_correct / total)
	storemodel(model, str(epoch) + "_model")

	## test
	model.eval()
	test_loss = 0
	test_correct = 0
	total = 0

	with torch.no_grad():
		for batch_num, (data, target) in enumerate(test_loader):
			data, target = data.to(device), target.to(device)
			output = model(data)
			loss = criterion(output, target)
			test_loss += loss.item()
			prediction = torch.max(output, 1)
			total += target.size(0)
			test_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())
	print(test_loss, test_correct / total)
