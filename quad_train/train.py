# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import os
import h5py
import time

import quad_utils
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

#os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from keras.models import Sequential
from keras.layers import Dense, Conv1D, Conv2D, SimpleRNN, Dropout, Flatten, MaxPooling2D, BatchNormalization
from keras.optimizers import SGD
from keras.utils import to_categorical

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset of images")
ap.add_argument("-l", "--label-bin", required=True,
	help="path to output label binarizer")
ap.add_argument("-m", "--model", required=True,
	help="path to output trained model")
args = vars(ap.parse_args())
 
(dataUni, labels) = quad_utils.data_import(args)

(train_x, test_x, train_y, test_y) = train_test_split(dataUni,
	labels, test_size=0.1, random_state=14)

lb = LabelBinarizer()
train_y = lb.fit_transform(train_y)
test_y = lb.transform(test_y)

train_y = to_categorical(train_y, num_classes=2)
test_y = to_categorical(test_y, num_classes=2)

dur = 2

batch_size = 64

train_x = train_x.reshape(train_x.shape[0], 753, dur*4, 1)
test_x = test_x.reshape(test_x.shape[0], 753, dur*4, 1)

model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', data_format="channels_last", input_shape=(753, dur*4, 1)))
#model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024, activation="sigmoid"))
model.add(BatchNormalization())
model.add(Dense(1024, activation="sigmoid"))
model.add(Dropout(0.5))
model.add(Dense(512, activation="sigmoid"))
model.add(Dropout(0.5))
model.add(Dense(128, activation="sigmoid"))
#model.add(SimpleRNN(len(lb.classes_)))
model.add(Dense(len(lb.classes_), activation="softmax"))

INIT_LR = 0.0001
EPOCHS = 25

print("[INFO] training network...")
opt = SGD(lr=INIT_LR)

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

H = model.fit(train_x, train_y, validation_data=(test_x, test_y), 
	epochs=EPOCHS, batch_size=batch_size)

#H = model.train_on_batch(test_x, test_y)

elapsed = time.clock() - elapsed
print("[INFO] evaluating network...", elapsed, "elapsed s")
predictions = model.predict(test_x, batch_size=batch_size)
print(classification_report(test_y.argmax(axis=1),
	predictions.argmax(axis=1), target_names=lb.classes_))


print("[INFO] serializing network and label binarizer...")
model.save(args["model"])
f = open(args["label_bin"], "wb")
f.write(pickle.dumps(lb))
f.close()

plt.plot(H.history['loss'])
plt.show()