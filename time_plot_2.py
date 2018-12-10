import os
import h5py
import pandas as pd
import numpy as np
import plotly.offline as pyo
import plotly.graph_objs as go
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

sns.set(color_codes=True)

model_state_X = 'AATGA'
model_state_Y = 'ATAGG'

def extract_dynamic(arr):

    start = np.array(arr['start'].astype(np.float))
    model_state = arr['model_state']
    move = np.array(arr['move'].astype(np.int))
    dynamic = []

    #np.savetxt('model_state_1', model_state, delimiter=' ', fmt="%s")
    model_state_ = []
    block_dist = []
    
    ind_start = 0
    ind_end = 0

    for i in np.arange(0, len(model_state)):
        model_state_.append(model_state[i].decode("utf-8"))
        if model_state_[-1:][0]==model_state_X:
            ind_start = i
        if model_state_[-1:][0]==model_state_Y:
            ind_end = i

    if ind_start>0 and ind_end>0:

        start_trunc = start[ind_start:ind_end]
        move_trunc = move[ind_start:ind_end]
        model_state_trunc = model_state_[ind_start:ind_end]
        print('total move = ', np.sum(move_trunc))

        return (np.sum(move_trunc), 1)

    else:

        return (0, 0)


fast5s = os.listdir('./alb_rna5/workspace/pass')
data = []
total = 0
hit = 0
total_time = 0
block_dist_total = []

for fast5 in fast5s:
    if fast5.endswith('.fast5'):
        total += 1
        f = h5py.File('./alb_rna5/workspace/pass/{}'.format(fast5), 'r')
        fastq = f['Analyses']['Basecall_1D_000']['BaseCalled_template']['Fastq'][()].decode('ascii')
        if ('GGAUA' in fastq.split("\n")[1]) and ('AGUAA' in fastq.split("\n")[1]):
            events = f['Analyses']['Basecall_1D_000']['BaseCalled_template']['Events'][()]

            block_dist, k = extract_dynamic(events)
            hit += k
            
            #block_dist_total.append(block_dist)
            total_time += block_dist*15

            print("total time of blockage = ", total_time, "average time of blockage = ", total_time/hit)

print("hit = ", hit, "total = ", total)

#plt.hist(block_dist, bins='auto')
#plt.show()
#data.append(trace_state)
#layout = go.Layout(title='Squiggle')

#fig = go.Figure(data=data,layout=layout)
#pyo.plot(fig,filename='read_0.html')

#print('{} pass, {} total'.format(hit, total))