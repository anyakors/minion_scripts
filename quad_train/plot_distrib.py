import os
import h5py
import numpy as np
import argparse
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()

ap = argparse.ArgumentParser()
ap.add_argument('file', type=argparse.FileType('r'), nargs='+')
args = ap.parse_args()

dist = []

for f in args.file:
	print(f)
	data = np.loadtxt(f, delimiter=',')
	#dist.append(data)
	dist.append(np.extract(data<50000, data))

print(len(dist))

sns_plt = sns.violinplot(data=dist, inner='quartile', cut=0)
#sns_plt.set_ylim(0,50000)
sns_plt.set(xlabel='molecule', ylabel='read length, samples')
sns_plt.figure.savefig("violin_dist.png")
plt.show()