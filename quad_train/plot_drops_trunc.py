import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
from zscore_utils import thresholding_algo, peak_frequency, drop_occurence


def continuous_drop(array):
    k = np.argmax(array) #last change
    drop_number = 0
    current = 0
    for i in range(0,len(array)):
        if array[i]==1:
            current += 1 #length of constant value
            if current==1:
                drop_number += 1
        else:
            current = 0
    return drop_number


def masked_y(array):
	y_ = np.ma.array(y)
	y_masked = np.ma.masked_where(y_==1, y_)
	return y_masked


ap = argparse.ArgumentParser()
ap.add_argument("-i1", "--input1", required=False,
    help="path to input dataset of arrays")
ap.add_argument("-i2", "--input2", required=False,
    help="path to input dataset of arrays")
args = vars(ap.parse_args())

#inp = "~/workspace/minion/selected/adapter/"

lag = 5
threshold = 5
influence = 0

files1 = os.listdir(args["input1"])
files2 = os.listdir(args["input2"])
#files = os.listdir(inp)
total = 0
arr1 = []
arr2 = []

freq_threshold = 1000

plt.subplot(211)
N1, N2 = 0, 0
i = 0

for file in files1:
    if 'ch' in file and i<200:
        y = np.loadtxt(os.path.join(args["input1"], file), delimiter=',')
        result = thresholding_algo(y, lag=lag, threshold=threshold, influence=influence)
        drops = drop_occurence(result["signals"], freq_threshold)
        drop_number = continuous_drop(drops)
        N1 += drop_number
        i += 1
        drops = [x+0.5*i for x in drops]
        y = [x+10.0*i for x in y]
        #plt.plot(np.arange(len(y)), y)
        plt.plot(np.arange(len(drops)), drops)
        plt.xlim(0, 180000)


i = 0
plt.subplot(212)
for file in files2:
    if 'ch' in file and i<200:
        y = np.loadtxt(os.path.join(args["input2"], file), delimiter=',')
        result = thresholding_algo(y, lag=lag, threshold=threshold, influence=influence)
        drops = drop_occurence(result["signals"], freq_threshold)
        drop_number = continuous_drop(drops)
        N2 += drop_number
        i += 1
        drops = [x+0.5*i for x in drops]
        y = [x+10.0*i for x in y]
        #plt.plot(np.arange(len(y)), y)
        plt.plot(np.arange(len(drops)), drops)
        plt.xlim(0, 180000)

plt.show()
print("Drop number for dataset 1: {}, drop number for dataset 2: {}".format(N1, N2))