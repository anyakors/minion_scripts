import os
import h5py
import numpy as np
import argparse
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=False,
    help="path to input dataset of arrays")
args = vars(ap.parse_args())

#inp = "~/workspace/minion/selected/adapter/"

files = os.listdir(args["input"])
#files = os.listdir(inp)
total = 0
dist = []

for file in files:

    if 'Channel' in file:
        total += 1
        arr = np.loadtxt(os.path.join(args["input"], file), delimiter=',')
        dist.append(len(arr))

print("total: ", total)
print(dist)

sns_plt = sns.distplot(dist)
sns_plt.figure.savefig("distrib_strands_len.png")