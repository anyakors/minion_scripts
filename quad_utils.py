# import the necessary packages
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import os
import h5py
import time

def stretch_interp(data, dur):

	longest = dur*3012
	data_new = []
	for i in np.arange(0, len(data)):
		array_old = data[i]
		indices_old = np.arange(0,len(array_old))
		indices_new = np.linspace(0,len(array_old)-1,longest)
		spl = UnivariateSpline(indices_old, array_old, k=3, s=0)
		array_new = spl(indices_new)
		data_new.append(array_new)

	return data_new


def data_import(args):

	# initialize the data and labels
	print("[INFO] loading datasets...")
	start_t = time.clock()
	data = []
	labels = []

	(root, dirs, files) = os.walk(args["dataset"])

	dataPaths = [os.path.join(root[0], subfolder) for subfolder in root[1]] 
	files = [os.listdir(dataPath) for dataPath in dataPaths]
	filePaths = [ [dataPaths[i]]*len(files[i]) for i in range(len(files)) ]
	filePathsFull = []

	for sublist in zip(filePaths, files):
		for i in range(len(sublist[0])):
			filePathsFull.append(os.path.join(sublist[0][i], sublist[1][i]))

	for filePath in filePathsFull:

		if filePath.endswith('.fast5'):
			f = h5py.File(filePath, 'r')
			read_no = list(f['Raw']['Reads'].keys())
			signal = f['Raw']['Reads'][read_no[0]]['Signal'][()]
			if int(len(signal)/3012)>dur:
				dur = int(len(signal)/3012)
			data.append(signal)
			label = filePath.split(os.path.sep)[-2]
			labels.append(label)
		if "Channel" in filePath:
			try:
				signal = np.loadtxt(filePath, encoding='latin1', delimiter="\n")
				if 1000<len(signal)<20000:
					data.append(signal)
					label = filePath.split(os.path.sep)[-2]
					labels.append(label)
			except ValueError:
				print("ValueError happened, moving on")
				continue
			dur = 2 # 7 is int(20000/3012)

	elapsed = time.clock() - start_t
	print("[INFO] done with data import...", elapsed, "elapsed s")
	labels = np.array(labels)
	print("Data array shape:", np.shape(data), ", labels array shape:", np.shape(labels))
	dataUni = stretch_interp(data, dur)
	dataUni = np.array(dataUni, dtype="float")
	print("DataUni array shape:", np.shape(dataUni))
	return (dataUni, labels)