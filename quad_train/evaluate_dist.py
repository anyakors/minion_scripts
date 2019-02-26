# import the necessary packages
import tensorflow as tf
from tensorflow.keras.models import load_model
from scipy.interpolate import UnivariateSpline
import numpy as np
import argparse
import random
import pickle
import os
import h5py
from tensorflow.keras.utils import plot_model

def stretch_interp(data, dur):

	longest = dur*3012
	indices_old = np.arange(0,len(data))
	indices_new = np.linspace(0,len(data)-1,longest)
	spl = UnivariateSpline(indices_old, data, k=3, s=0)
	data_new = spl(indices_new)

	return data_new

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--file", required=True,
	help="path to input file we are going to classify")
ap.add_argument("-m", "--model", required=True,
	help="path to trained Keras model")
ap.add_argument("-l", "--label-bin", required=True,
	help="path to label binarizer")
args = vars(ap.parse_args())

# load the model and label binarizer
print("[INFO] loading network and label binarizer...")
model = load_model(args["model"])
lb = pickle.loads(open(args["label_bin"], "rb").read())

hist = np.loadtxt(args["file"], encoding='latin1', delimiter="\n")
print(np.shape(hist))
hist = hist.reshape(1, 100)

# make a prediction on the image
preds = model.predict(hist)

# find the class label index with the largest corresponding
# probability
i = preds.argmax(axis=1)[0]

label = lb.classes_[i]

print("prediction:", i, "label", label)