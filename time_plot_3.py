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

model_state_X = 'AGGAC'
model_state_X_1 = 'GGACC'
model_state_Y = 'ATAGG'
model_state_Y_1 = 'AATAG'

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
        k = 0
        for i in np.arange(1, len(start_trunc)):
            if move_trunc[i]==0:
                k += 1
            if move_trunc[i]>0:
                block_dist.append([k])
                k = 0

        return (block_dist, 1)

    else:

        return (0, 0)


fast5s = os.listdir('./alb_bc/workspace/pass/0')
data = []
total = 0
hit = 0
total_time = 0
block_dist_total = []
average = 4660.
std = 0

for fast5 in fast5s:
    if fast5.endswith('.fast5'):
        total += 1
        f = h5py.File('./alb_bc/workspace/pass/0/{}'.format(fast5), 'r')
        fastq = f['Analyses']['Basecall_1D_000']['BaseCalled_template']['Fastq'][()].decode('ascii')
        if ('AGGAC' in fastq.split("\n")[1]) and ('GGAUA' in fastq.split("\n")[1]):
            events = f['Analyses']['Basecall_1D_000']['BaseCalled_template']['Events'][()]

            block_dist, k = extract_dynamic(events)
            hit += k

            #block_dist_total.append(block_dist)
            blk_time = np.sum(block_dist)*15
            print("blk_time = ", blk_time)

            if blk_time>0:
                std += np.square(blk_time - average)
                print("std = ", std)

            total_time += blk_time
            print("total time pf blockage = ", total_time, "average time of blockage = ", total_time/hit)

print("hit = ", hit, "total = ", total)
print("std = ", np.sqrt(std/hit))

#plt.hist(block_dist, bins='auto')
#plt.show()
#data.append(trace_state)
#layout = go.Layout(title='Squiggle')

#fig = go.Figure(data=data,layout=layout)
#pyo.plot(fig,filename='read_0.html')

#print('{} pass, {} total'.format(hit, total))