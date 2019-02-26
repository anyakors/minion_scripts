import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
from threshold_alg import thresholding_algo, peak_frequency, drop_occurence

ap = argparse.ArgumentParser()
ap.add_argument("-i1", "--input1", required=False,
    help="path to input dataset of arrays")
ap.add_argument("-i2", "--input2", required=False,
    help="path to input dataset of arrays")
args = vars(ap.parse_args())

#inp = "~/workspace/minion/selected/adapter/"

files1 = os.listdir(args["input1"])
files2 = os.listdir(args["input2"])
#files = os.listdir(inp)
total = 0
arr1 = []
arr2 = []

plt.subplot(121)
for file in files1:
    if 'ch' in file:
        arr = np.loadtxt(os.path.join(args["input1"], file), delimiter=',')
        plt.plot(np.arange(len(arr)), arr)

plt.subplot(122)
for file in files2:
    if 'ch' in file:
        arr = np.loadtxt(os.path.join(args["input2"], file), delimiter=',')
        plt.plot(np.arange(len(arr)), arr)

plt.show()