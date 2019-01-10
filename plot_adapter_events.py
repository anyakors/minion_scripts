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

from tombo import tombo_helper, tombo_stats, resquiggle
import mappy



def extract_adapter_events(events, states):

    event_start = np.array(events['start'].astype(np.int))
    lengths = np.array(events['length'].astype(np.int))
    mean = np.array(events['mean'].astype(np.float))

    state = np.array(states['summary_state'].astype(np.int))
    state_start = np.array(states['acquisition_raw_index'].astype(np.int))

    start_point, end_point = [], []
    ind_start_event, ind_end_event = [], []

    for i in np.arange(0, len(state)):
        if state[i]==5 and state[i+1]==3:
            start_point.append(state_start[i])
            end_point.append(state_start[i+1])

    print(start_point, end_point)
    
    for m in np.arange(0, len(start_point)):
        for i in np.arange(0, len(event_start)):
            if start_point[m]>=event_start[i] and start_point[m]<event_start[i+1]:
                ind_start_event.append(i)
                break

    for m in np.arange(0, len(end_point)):
        for i in np.arange(0, len(event_start)):
            if end_point[m]>=event_start[i] and end_point[m]<event_start[i+1]:
                ind_end_event.append(i)
                break

    print('ind_start:', len(ind_start_event), len(ind_end_event))

    adapter_region_events = []
    buf_region = []

    for i in np.arange(0, len(ind_start_event)):
        region_mean = mean[ind_start_event[i]:ind_end_event[i]]
        region_lens = lengths[ind_start_event[i]:ind_end_event[i]]
        for l in np.arange(0, len(region_mean)):
            buf_region.extend(np.repeat(region_mean[l], region_lens[l]))
        #adapter_region_events.append(region_mean)                       #just mean
        adapter_region_events.append(buf_region)                         #mean*length
        buf_region = []

    print('region len:', len(adapter_region_events))

    return adapter_region_events


def stretch_repeat(data): 
    for i in np.arange(0, len(data)):
        array_old = data[i]
        array_new = np.repeat(array_old, 5, axis=0)
        ax = plt.subplot(len(data),1,len(data)-i)
        ax.plot(np.arange(0,len(array_new)), array_new)
        plt.setp(ax.get_xticklabels(), visible=False)
    return


fast5 = h5py.File('/media/mookse/DATA1/minion_data/bulk/mookse_Veriton_X4650G_20180613_FAH54029_MN21778_sequencing_run_RNA3_G4_false_79563.fast5')

events = fast5['IntermediateData']['Channel_10']['Events'][()]
states = fast5['StateData']['Channel_10']['States'][()]

data = extract_adapter_events(events, states)

print(np.shape(data[0]))




#tombo_model = '/home/mookse/anaconda3/pkgs/ont-tombo-1.4-py36r341h24bf2e0_0/lib/python3.6/site-packages/tombo/tombo_models/tombo.DNA.model'
tombo_model = '/home/mookse/anaconda3/pkgs/ont-tombo-1.5-py36r341h24bf2e0_0/lib/python3.6/site-packages/tombo/tombo_models/tombo.DNA.model'
reference_fn = 'GGCTTCTTCTTGCTCTTAGGTAGTAGGTTC'

instance = tombo_stats.TomboModel(tombo_model)
print([func for func in dir(tombo_stats.TomboModel) if callable(getattr(tombo_stats.TomboModel, func))])

std_model = instance.get_exp_levels_from_seq(reference_fn, rev_strand=False)

print(len(std_model[0]))
print(len(reference_fn))

model_new = np.repeat(std_model[0], 48, axis=0)
model_new = np.concatenate((np.zeros(2000), model_new, np.zeros(2000)), axis=None)

ax = plt.subplot(np.rint((len(data)+1)/2), 2, 1)
plt.plot(np.arange(0, len(model_new)), model_new, 'r')



#data = np.array(data)
#array_old = data
i = 1

for instance in data:
    #instance_new = np.repeat(np.array(instance), 10, axis=0)
    ax = plt.subplot(np.rint((len(data)+1)/2), 2, i+1)
    i+=1
    #plt.plot(np.arange(0, len(instance_new)), instance_new)
    plt.plot(np.arange(0, len(instance)), instance)
    plt.setp(ax.get_xticklabels(), visible=False)

plt.show()