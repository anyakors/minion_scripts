import os
import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

fast5s = os.listdir('/home/mookse/workspace/DeepSimulator/fast5')
#fast5 = h5py.File('/home/mookse/workspace/DeepSimulator/fast5/signal_1_8ee7e214-9f29-4e09-8f49-e7656565710d.fast5')
i = 1

for fast5 in fast5s:

    print(fast5)
    fast5 = '/home/mookse/workspace/DeepSimulator/fast5/' + fast5
    f = h5py.File(fast5)
    read_no = list(f['Raw']['Reads'].keys()) 
    signal= f['Raw']['Reads'][read_no[0]]['Signal'].value
    x = np.arange(0,len(signal))
    ax = plt.subplot(len(fast5s), 1, i)
    i+=1
    plt.plot(x, signal)
    plt.setp(ax.get_xticklabels(), visible=False)


plt.show()