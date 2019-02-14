#extracts random strand events from a given bulk file obeying normal distrib

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.interpolate import UnivariateSpline
import argparse
import statistics
import seaborn as sns

sns.set()

def extract_strand_events(events, states):

    event_start = np.array(events['start'].astype(np.int))
    lengths = np.array(events['length'].astype(np.int))
    mean = np.array(events['mean'].astype(np.float))

    state = np.array(states['summary_state'].astype(np.int))
    state_start = np.array(states['acquisition_raw_index'].astype(np.int))
    #state_start = np.array(states['analysis_raw_index'].astype(np.int))

    start_point, end_point = [], []
    ind_start_event, ind_end_event = [], []

    for i in np.arange(0, len(state)):
        # state 5 -- adapter, state 3 -- strand, so we only take those adapter labels which are followed by strand
        if state[i]==3: 
            start_point.append(state_start[i])
            end_point.append(state_start[i+1])
    
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

    strand_events = []
    buf_region = []

    for i in np.arange(0, len(ind_start_event)):
        region_mean = mean[ind_start_event[i]:ind_end_event[i]]
        region_lens = lengths[ind_start_event[i]:ind_end_event[i]]
        for l in np.arange(0, len(region_mean)):
            buf_region.extend(np.repeat(region_mean[l], region_lens[l]))
        #adapter_region_events.append(region_mean)                       #just mean
        strand_events.append(buf_region)                         #mean*length
        buf_region = []

    return strand_events


def stretch_repeat(data): 
    for i in np.arange(0, len(data)):
        array_old = data[i]
        array_new = np.repeat(array_old, 5, axis=0)
        ax = plt.subplot(len(data),1,len(data)-i)
        ax.plot(np.arange(0,len(array_new)), array_new)
        plt.setp(ax.get_xticklabels(), visible=False)
    return


ap = argparse.ArgumentParser()
ap.add_argument("-f", "--file", required=True,
    help="path to bulk fast5 file")
#ap.add_argument("-c", "--channel", required=True,
#    help="channel number")
ap.add_argument("-o", "--output", required=True,
    help="path to savedir")
ap.add_argument("-p", "--plot", required=True,
    help="path to plot file savedir")
args = vars(ap.parse_args())

fast5 = h5py.File(args["file"])
i = 0
len_arr = []
overlay = []

for key in fast5['IntermediateData'].keys():

    print(key)
    events = fast5['IntermediateData'][key]['Events'][()]
    states = fast5['StateData'][key]['States'][()]
    try:
        data = extract_strand_events(events, states)
    except IndexError:
        print("IndexError happened, moving on")
        continue
    i += 1
    print(len(data), "strand matches")
    for instance in data:
        if instance:
            len_arr.append(len(instance))
    if len_arr and len(len_arr)>10:
        print("After", len(len_arr), "strands, mean =", statistics.mean(len_arr), "std =", statistics.stdev(len_arr))
    if i>512:
        break
    if len_arr and len(len_arr)//1000==10:
        np.savetxt(args["output"], len_arr, delimiter=',')
        break

    # the part for plotting an overlay
    #if 10000<len(instance)<20000:
    #    overlay.append(instance)
    #if overlay and len(overlay)>500:
    #    break

#print("After", len(len_arr), "strands, mean =", statistics.mean(len_arr), "std =", statistics.stdev(len_arr))
#np.savetxt(args["output"], len_arr, delimiter=',')

np.savetxt(args["output"], len_arr, delimiter=',')

sns_plt = sns.distplot(len_arr)
#sns_plt.set_xlim(0, 200000)
sns_plt.figure.savefig(args["plot"])

# the part for plotting an overlay
#for instance in overlay:
#    plt.plot(np.arange(0,len(instance)), instance, alpha=0.5)

plt.show()