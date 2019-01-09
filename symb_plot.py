import os
import h5py
import pandas as pd
import numpy as np
import plotly.offline as pyo
import plotly.graph_objs as go

def extract_dynamic(arr):

    start = np.array(arr['start'].astype(np.float))
    model_state = arr['model_state']
    move = np.array(arr['move'].astype(np.int))

    dynamic = []

    for i in np.arange(0, len(start)):
        if move[i]>0:
            dynamic.append([start[i].astype(np.float), ((model_state[i][-move[i]:]).decode('UTF-8')).replace('T', 'U')])

    return dynamic


files = os.listdir('/home/mookse/Desktop/Analysis/RNA3_short/alb_bc/workspace/pass/0/')

for file in files:
    if file.endswith('read_148_ch_440_strand.fast5'):
        fast5 = h5py.File('/home/mookse/Desktop/Analysis/RNA3_short/alb_bc/workspace/pass/0/' + file)
        print(file)

data = []
total = 0
hit = 0

fastq = fast5['Analyses']['Basecall_1D_000']['BaseCalled_template']['Fastq'][()].decode('ascii')
print(fastq.split("\n")[1])

events = fast5['Analyses']['Basecall_1D_000']['BaseCalled_template']['Events'][()]
dynamic = extract_dynamic(events)
dynamic = np.array(dynamic)
print(dynamic[:,0])
print(dynamic[:,1])

dset = [fast5['Raw']['Reads'][key]['Signal'] for key in fast5['Raw']['Reads'].keys()]
raw = pd.Series(dset[0][:])
x = np.arange(0,len(raw))

trace = go.Scatter(x=x, y=raw,
                mode='lines',
                name='raw_{}'.format(total))
data.append(trace)

trace_state = go.Scatter(
    x=dynamic[:,0],
    y=np.ones(len(dynamic)),
    mode='markers+text',
    name='Model state',
    text=dynamic[:,1],
    textfont=dict(
        family='sans serif',
        size=10,
        color='#000000'
    ),
    textposition='top center'
)

data.append(trace_state)
layout = go.Layout(title='Squiggle')

fig = go.Figure(data=data,layout=layout)
pyo.plot(fig,filename='RNA3_read_148_ch_440.html')

print('{} pass, {} total'.format(hit, total))