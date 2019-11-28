import os
import sys
import cv2
import pickle
import progressbar
import numpy as np
import matplotlib.pyplot as plt

datapath = sys.argv[1]
train_or_test = "train_" if "train" in datapath else "test_"

with open(datapath, 'rb') as handle:
	data = pickle.load(handle)
	X = data["X"]
	Y = data["Y"]

out_datapath = sys.argv[2]
if out_datapath[-1] != "/":
	out_datapath + "/"
out_datapath = out_datapath + train_or_test + "cifar/"

if not os.path.isdir(out_datapath):
	os.mkdir(out_datapath)

size = (256, 256)

for i, (x, y) in progressbar.progressbar(enumerate(zip(X, Y))):
	out_dir = out_datapath + str(y) + "/"
	if not os.path.isdir(out_dir):
		os.mkdir(out_dir)
	im = np.array(x, np.int32)
	im_r = im[0:1024].reshape(32, 32).astype('float32')
	im_g = im[1024:2048].reshape(32, 32).astype('float32')
	im_b = im[2048:].reshape(32, 32).astype('float32')
	img = np.dstack((im_r, im_g, im_b))
	img = cv2.resize(img, size)
	img = np.array(img, np.int32)
	im_name = out_dir + str(i+1) + ".png"
	cv2.imwrite(im_name, img)
