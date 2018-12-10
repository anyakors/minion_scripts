import os
import h5py
import pandas as pd
import numpy as np
import plotly.offline as pyo
import plotly.graph_objs as go


model_state_X = 'AGGAC'
model_state_Y = 'ATAGG'

def extract_dynamic(arr):

    start = np.array(arr['start'].astype(np.float))
    model_state = arr['model_state']
    move = np.array(arr['move'].astype(np.int))
    dynamic = []
    model_state_ = []
    ind_start, ind_end = 0, 0
    check = 0

    for i in np.arange(0, len(model_state)):
        model_state_.append(model_state[i].decode("utf-8"))
        if model_state_[-1:][0]==model_state_X:
            ind_start = i
        if model_state_[-1:][0]==model_state_Y:
            ind_end = i

    if ind_start>0 and ind_end>0:
        check = 1   
        start_trunc = start[ind_start:ind_end]
        move_trunc = move[ind_start:ind_end]
        model_state_trunc = model_state_[ind_start:ind_end]
    else:
        return (dynamic, check)

    for i in np.arange(0, len(start_trunc)):
        if move_trunc[i]>0:
            dynamic.append([start_trunc[i], (model_state_trunc[i][-move_trunc[i]:]).replace('T', 'U')])

    return (dynamic, check)


fast5s = os.listdir('./alb_bc/workspace/pass/test')
total = 0
hit = 0

for fast5 in fast5s:

    if fast5.endswith('.fast5'):
        data = []
        total += 1
        f = h5py.File('./alb_bc/workspace/pass/test/{}'.format(fast5), 'r')
        fastq = f['Analyses']['Basecall_1D_000']['BaseCalled_template']['Fastq'][()].decode('ascii')
        # if 700 <= len(fastq.split("\n")[1]) <= 750:
        if ('AGGAC' in fastq.split("\n")[1]) and ('GGAUA' in fastq.split("\n")[1]):
            events = f['Analyses']['Basecall_1D_000']['BaseCalled_template']['Events'][()]
            dynamic, check = extract_dynamic(events)
            dynamic = np.array(dynamic)

            #print(dynamic[0])

            if check:
                dset = [f['Raw']['Reads'][key]['Signal'] for key in f['Raw']['Reads'].keys()]
                hit += 1
                raw = pd.Series(dset[0][:])
                x = np.arange(0,len(raw))
                trace = go.Scatter(x=x, y=raw,
                                   mode='lines',
                                   name='raw_{}'.format(hit))
                data.append(trace)

                trace_state = go.Scatter(
                    x=dynamic[:,0].astype(np.float),
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
                pyo.plot(fig,filename='_read_{}.html'.format(hit))

print('{} pass, {} total'.format(hit, total))