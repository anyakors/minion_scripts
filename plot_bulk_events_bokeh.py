#plot the events from a bulk file, from one of the channels, with no labels or markings

import os
import h5py
import numpy as np
from scipy import stats
from scipy.interpolate import UnivariateSpline

from bokeh.plotting import figure, output_file, show

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

#fast5 = h5py.File('/media/mookse/DATA1/minion_data/bulk/mookse_Veriton_X4650G_20181213_FAH54070_MN21778_sequencing_run_RNA11_27121.fast5')
fast5 = h5py.File('/media/mookse/DATA1/minion_data/bulk/mookse_Veriton_X4650G_20180612_FAH54029_MN21778_sequencing_run_RNA3_G4_89485.fast5')
data = []
events = fast5['IntermediateData']['Channel_167']['Events'][()]
start = np.array(events['start'].astype(np.float))
mean = np.array(events['mean'].astype(np.float))
length = np.array(events['length'].astype(np.int))
signal = fast5['Raw']['Channel_167']['Signal'].value

array_new = np.array(signal)
#array_new = []

#for i in np.arange(0, len(mean)):
#    array_new.extend([mean[i]]*length[i])

output_file("lines.html")
# create a new plot with a title and axis labels
p = figure(title="simple line example", x_axis_label='t, 1/3012s', y_axis_label='I, pA', sizing_mode='stretch_both')
# add a line renderer with legend and line thickness
p.line(np.arange(0,len(array_new)), array_new, legend="bulk", line_width=1)
# show the results
show(p)