import os
import h5py
import pandas as pd
import numpy as np
import plotly.offline as pyo
import plotly.graph_objs as go
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.interpolate import UnivariateSpline


sns.set(color_codes=True)

model_state_X = 'AAGAC'
model_state_Y = 'UGAGA'

def extract_dynamic(events, signal):

    start = np.array(events['start'].astype(np.float))
    model_state = events['model_state']
    move = np.array(events['move'].astype(np.int))
    g_region = []
    
    ind_start = 0
    ind_end = 0

    for i in np.arange(0, len(model_state)):
        if (((model_state[i]).decode('UTF-8')).replace('T', 'U'))==model_state_X and ind_start==0:
            ind_start = i
            start_point = start[i]
        if (((model_state[i]).decode('UTF-8')).replace('T', 'U'))==model_state_X and i<ind_start:
            ind_start = i
            start_point = start[i]
        if (((model_state[i]).decode('UTF-8')).replace('T', 'U'))==model_state_Y and i>ind_end:
            ind_end = i
            end_point = start[i]

    if ind_end>ind_start>0:
        g_region = signal[int(start_point):int(end_point)]
        return g_region

    else:
        return 0


fast5s = os.listdir('/home/mookse/Desktop/Analysis/RNA3_short/alb_bc/workspace/pass/test')
data = []
total, hit = 0, 0

for fast5 in fast5s:
    if fast5.endswith('.fast5'):
        total+=1
        f = h5py.File('/home/mookse/Desktop/Analysis/RNA3_short/alb_bc/workspace/pass/test/{}'.format(fast5), 'r')
        fastq = f['Analyses']['Basecall_1D_000']['BaseCalled_template']['Fastq'][()].decode('ascii')

        #if ('CAGAA' in fastq.split("\n")[1]) and ('AGAGU' in fastq.split("\n")[1]):
        if (b'AAGAC' in f['Analyses']['Basecall_1D_000']['BaseCalled_template']['Events']['model_state']) and (b'TGAGA' in f['Analyses']['Basecall_1D_000']['BaseCalled_template']['Events']['model_state']):
            hit+=1
            #print('got one', f, fastq)
            events = f['Analyses']['Basecall_1D_000']['BaseCalled_template']['Events'][()]
            dset = [f['Raw']['Reads'][key]['Signal'] for key in f['Raw']['Reads'].keys()]
            signal = dset[0][:]

            g_region = extract_dynamic(events, signal)

            if not isinstance(g_region, int):
                data.append(g_region)


data = np.array(data)
longest = 0

for i in np.arange(0, len(data)):
    if len(data[i])>longest:
        longest = len(data[i])

print("longest array =", longest)
time = np.arange(0,longest)

fig=plt.figure()

for i in np.arange(0, len(data)):
    array_old = data[i]
    print("Length of", i, "th array is", len(data[i]))
    indices_old = np.arange(0,len(array_old))
    indices_new = np.linspace(0,len(array_old)-1,longest)
    spl = UnivariateSpline(indices_old, array_old, k=3, s=0)
    array_new = spl(indices_new)
    if i==0:
        ax1 = plt.subplot(len(data),1,len(data)-i)
        plt.plot(time, array_new)
        print("yes axis created")
    else:
        ax = plt.subplot(len(data),1,len(data)-i, sharex = ax1)
        plt.plot(time, array_new)
        plt.setp(ax.get_xticklabels(), visible=False)

print("total =", total, ", hit =", hit)

plt.show()