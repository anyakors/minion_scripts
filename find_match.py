import os
import h5py
import pandas as pd
import numpy as np
import plotly.offline as pyo
import plotly.graph_objs as go

def rolling_window(a, size):
    shape = a.shape[:-1] + (a.shape[-1] - size + 1, size)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


fast5 = h5py.File('/media/mookse/DATA1/minion/bulk/mookse_Veriton_X4650G_20180618_FAH54070_MN21778_sequencing_run_RNA5_long_G4_49077.fast5')

fast5_read = h5py.File('/home/mookse/workspace/MinKNOW/data/reads/20180618_0526_RNA5_long_G4/fast5/0/mookse_Veriton_X4650G_20180618_FAH54070_MN21778_sequencing_run_RNA5_long_G4_49077_read_6_ch_100_strand.fast5')

dataset = fast5['Raw']['Channel_100']['Signal'].value
dataset_read = fast5_read['Raw']['Reads']['Read_6']['Signal'].value

print(list(fast5_read['Raw']['Reads'].keys()))

x = np.arange(0,len(dataset))
x_read = np.arange(0,len(dataset_read))

print(dataset[32269:32269+10])
print(dataset_read[0:10])

bool_indices = np.all(rolling_window(dataset, 10) == dataset_read[0:10], axis=1)

print(bool_indices)

if bool_indices.all == False:
    print('ALL FALSE')

print(np.flatnonzero(bool_indices))