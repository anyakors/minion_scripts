import matplotlib
matplotlib.use("Agg")
 
# import the necessary packages
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Conv2D, SimpleRNN, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import numpy as np
import argparse
import random
import pickle
import os
import h5py

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

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset of images")
ap.add_argument("-l", "--label-bin", required=True,
	help="path to output label binarizer")
ap.add_argument("-m", "--model", required=True,
	help="path to output trained model")
args = vars(ap.parse_args())

# initialize the data and labels
print("[INFO] loading datasets...")
data = []
labels = []
dur = 0

#dataPaths = os.listdir(args["dataset"])
(root, dirs, files) = os.walk(args["dataset"])

#dataPaths = list(filter(os.path.isdir(args["dataset"]), os.listdir(args["dataset"])))

#dataPathsFull = [os.path.join(args["dataset"], dataPath) for dataPath in dataPaths] 
# if something is not a directory

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

		#signal = np.loadtxt(filePath, delimiter=',')
		try:
			signal = np.loadtxt(filePath, encoding='latin1', delimiter="\n")

			if 1000<len(signal)<20000:
				data.append(signal)
				label = filePath.split(os.path.sep)[-2]
				labels.append(label)

		except ValueError:
			print("ValueError happened, moving on")
			continue
		#signal = f.read()

		dur = 7 # 7 is int(20000/3012)

print("[INFO] done with data import...")

labels = np.array(labels)

print("Data array shape:", np.shape(data), ", labels array shape:", np.shape(labels))

dataUni = stretch_interp(data, dur)
dataUni = np.array(dataUni, dtype="float")

print("DataUni array shape:", np.shape(dataUni))

(train_x, test_x, train_y, test_y) = train_test_split(dataUni,
	labels, test_size=0.1, random_state=14)

lb = LabelBinarizer()
train_y = lb.fit_transform(train_y)
test_y = lb.transform(test_y)

#batch_size = np.shape(train_x)[0]
batch_size = 32

train_x = train_x.reshape(train_x.shape[0], 753, dur*4, 1)
test_x = test_x.reshape(test_x.shape[0], 753, dur*4, 1)

model = Sequential()
#model.add(Conv1D(3012, input_shape=(53, dur*3012), kernel_size=(12), activation='relu'))

model.add(Conv2D(32, (4,4), activation='relu', data_format="channels_last", input_shape=(753, dur*4, 1)))
print(model.input_shape, model.output_shape)
model.add(Flatten())

model.add(Dense(3012, activation="relu"))
model.add(Dense(1506, activation="sigmoid"))
model.add(Dropout(0.5))

model.add(Dense(753, activation="sigmoid"))
model.add(Dropout(0.25))
model.add(Dense(75, activation="sigmoid"))
#model.add(SimpleRNN(len(lb.classes_), input_shape=(253,)))
model.add(Dense(len(lb.classes_), activation="softmax"))

INIT_LR = 0.01
EPOCHS = 75

print("[INFO] training network...")
opt = SGD(lr=INIT_LR)
#model.compile(loss="binary_crossentropy", optimizer=opt,
#	metrics=["accuracy"])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

H = model.fit(train_x, train_y, validation_data=(test_x, test_y), 
	epochs=EPOCHS, batch_size=batch_size)

print("[INFO] evaluating network...")
predictions = model.predict(test_x, batch_size=batch_size)
print(classification_report(test_y.argmax(axis=1),
	predictions.argmax(axis=1), target_names=lb.classes_))


print("[INFO] serializing network and label binarizer...")
model.save(args["model"])
f = open(args["label_bin"], "wb")
f.write(pickle.dumps(lb))
f.close()