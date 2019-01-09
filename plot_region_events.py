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
    mean = np.array(events['mean'].astype(np.float))
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
        g_region_signal = signal[int(start_point):int(end_point)]
        g_region_events = mean[ind_start:ind_end]
        return g_region_events

    else:
        return 0


def stretch_interp(data):
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
    return


def stretch_repeat(data):
    for i in np.arange(0, len(data)):
        array_old = data[i]
        array_new = np.repeat(array_old, 5, axis=0)
        ax = plt.subplot(len(data),1,len(data)-i)
        ax.plot(np.arange(0,len(array_new)), array_new)
        plt.setp(ax.get_xticklabels(), visible=False)
    return


fast5s = os.listdir('/home/mookse/Desktop/Analysis/RNA9_short/alb_bc/workspace/pass/0')
data = []
total, hit = 0, 0

for fast5 in fast5s:
    if fast5.endswith('.fast5') and total<1000:
        total+=1
        f = h5py.File('/home/mookse/Desktop/Analysis/RNA9_short/alb_bc/workspace/pass/0/{}'.format(fast5), 'r')
        fastq = f['Analyses']['Basecall_1D_000']['BaseCalled_template']['Fastq'][()].decode('ascii')

        #if ('CAGAA' in fastq.split("\n")[1]) and ('AGAGU' in fastq.split("\n")[1]):
        if (b'AAGAC' in f['Analyses']['Basecall_1D_000']['BaseCalled_template']['Events']['model_state']) \
        and (b'TGAGA' in f['Analyses']['Basecall_1D_000']['BaseCalled_template']['Events']['model_state']): #\ 
        #and not (b'GGGTG' in f['Analyses']['Basecall_1D_000']['BaseCalled_template']['Events']['model_state']):
            #print('got one', f, fastq)
            events = f['Analyses']['Basecall_1D_000']['BaseCalled_template']['Events'][()]
            dset = [f['Raw']['Reads'][key]['Signal'] for key in f['Raw']['Reads'].keys()]
            signal = dset[0][:]

            g_region = extract_dynamic(events, signal)

            if not isinstance(g_region, int):
                hit+=1
                data.append(g_region)


data = np.array(data)

#stretch_interp(data)
stretch_repeat(data)

print("total =", total, ", hit =", hit)

plt.show()