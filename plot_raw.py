import os
import h5py
import pandas as pd
import numpy as np
import plotly.offline as pyo
import plotly.graph_objs as go


#fast5 = h5py.File('/media/mookse/DATA1/minion/bulk/mookse_Veriton_X4650G_20180618_FAH54070_MN21778_sequencing_run_RNA5_long_G4_49077.fast5')

fast5_read = h5py.File('/home/mookse/workspace/MinKNOW/data/reads/20180618_0526_RNA5_long_G4/fast5/0/mookse_Veriton_X4650G_20180618_FAH54070_MN21778_sequencing_run_RNA5_long_G4_49077_read_6_ch_100_strand.fast5')

#dataset = fast5['Raw']['Channel_100']['Signal']

dataset_read = fast5_read['Raw']['Reads']['Read_6']['Signal']

print(dataset_read)

#raw = pd.Series(dataset)
raw_read = pd.Series(dataset_read)

print(raw_read)

#x = np.arange(0,len(raw))
x_read = np.arange(0,len(raw_read))

data = []
trace = go.Scatter(x=x_read, y=raw_read,
                    mode='lines',
                    name='raw_100')
data.append(trace)

layout = go.Layout(title='Squiggle')

fig = go.Figure(data=data,layout=layout)
pyo.plot(fig, filename='ch_100.html')