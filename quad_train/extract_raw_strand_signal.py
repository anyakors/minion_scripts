#extracts random strand events from a given bulk file obeying normal distrib

import os
import h5py
import numpy as np
from scipy import stats
from scipy.interpolate import UnivariateSpline
import argparse
import statistics

def extract_strand_events(raw, states):

    signal = np.array(raw.astype(np.float))

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
    
    strand_signal = []

    for m in np.arange(0, len(start_point)):
        strand_signal.append(signal[start_point[m]:end_point[m]])

    return strand_signal


ap = argparse.ArgumentParser()
ap.add_argument("-f", "--file", required=True,
    help="path to bulk fast5 file")
ap.add_argument("-o", "--output", required=True,
    help="path to dist savedir")
args = vars(ap.parse_args())

fast5 = h5py.File(args["file"])
i = 0
len_arr = []
overlay = []

for key in fast5['IntermediateData'].keys():

    print(key)
    events = fast5['IntermediateData'][key]['Events'][()]
    states = fast5['StateData'][key]['States'][()]
    raw = fast5['Raw'][key]['Signal'][()]
    try:
        data = extract_strand_events(raw, states)
    except IndexError:
        print("IndexError happened, moving on")
        continue
    i += 1
    k = 1
    print(len(data), "strand matches")
    for instance in data:
        if len(instance)>1608:
            filename = 'ch_{}_{}'.format(i, k)
            np.savetxt(os.path.join(args["output"], filename), instance[1607:], delimiter=',')
            k += 1
    if k>1000:
        break