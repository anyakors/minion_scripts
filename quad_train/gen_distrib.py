import os
import numpy as np
import argparse
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument('file', type=argparse.FileType('r'), nargs='+')
args = ap.parse_args()

for f in args.file:
	print(f)
	print(str(f))
	f_new = (str(f)).replace("<_io.TextIOWrapper name='./", "").replace("' mode='r' encoding='UTF-8'>", "")
	print(f_new)
	data = np.loadtxt(f, delimiter=',')
	hist = np.histogram(data, bins=100, range=(0,100000))[0]
	np.savetxt('hist_'+f_new, hist, delimiter=',')
	plt.hist(data, bins=100, range=(0,100000))

plt.show()