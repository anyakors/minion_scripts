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


channel_no = 'Channel_60'
channel_filename = 'ch_60_strand.fast5'

#raw_fast5 = h5py.File('/media/mookse/DATA1/minion/bulk/mookse_Veriton_X4650G_20180618_FAH54070_MN21778_sequencing_run_RNA5_long_G4_49077.fast5')

raw_fast5 = h5py.File('/media/mookse/DATA1/minion_data/bulk/mookse_Veriton_X4650G_20180613_FAH54029_MN21778_sequencing_run_RNA3_G4_false_79563.fast5')

signal = raw_fast5['Raw'][channel_no]['Signal'].value

#===================PASS=====================

match_list_p = []
match_text_p = []

#fast5s = os.listdir('/home/mookse/workspace/MinKNOW/data/reads/20180618_0526_RNA5_long_G4/alb_rna5/workspace/pass')
fast5s = os.listdir('/home/mookse/Desktop/Analysis/RNA3_short/alb_bc/workspace/pass/0')

for fast5 in fast5s:

	if fast5.endswith(channel_filename):

		f = h5py.File('/home/mookse/Desktop/Analysis/RNA3_short/alb_bc/workspace/pass/0/{}'.format(fast5), 'r')
		read_no = list(f['Raw']['Reads'].keys())
		print(read_no)
		signal_read = f['Raw']['Reads'][read_no[0]]['Signal'].value

		bool_indices = np.all(rolling_window(signal, 10) == signal_read[0:10], axis=1)

		match = np.flatnonzero(bool_indices)
		print(match)
		print(f['Analyses']['Basecall_1D_000']['BaseCalled_template']['Fastq'].value)

		match_list_p.append(match[0])
		match_text_p.append(read_no[0])


#===================FAIL=====================

match_list_f = []
match_text_f = []

#fast5s = os.listdir('/home/mookse/workspace/MinKNOW/data/reads/20180618_0526_RNA5_long_G4/alb_rna5/workspace/fail')
fast5s = os.listdir('/home/mookse/Desktop/Analysis/RNA3_short/alb_bc/workspace/fail/0')

for fast5 in fast5s:

	if fast5.endswith(channel_filename):

		f = h5py.File('/home/mookse/Desktop/Analysis/RNA3_short/alb_bc/workspace/fail/0/{}'.format(fast5), 'r')
		read_no = list(f['Raw']['Reads'].keys())
		print(read_no)
		signal_read = f['Raw']['Reads'][read_no[0]]['Signal'].value

		bool_indices = np.all(rolling_window(signal, 10) == signal_read[0:10], axis=1)

		match = np.flatnonzero(bool_indices)
		print(match)
		#print(f['Analyses']['Basecall_1D_000']['BaseCalled_template']['Fastq'].value)

		match_list_f.append(match[0])
		match_text_f.append(read_no[0])



x = np.arange(0,len(signal))

data = []
trace = go.Scatter(x=x[1*500000:3*500000], y=signal[1*500000:3*500000],
                    mode='lines',
                    name=channel_no)
data.append(trace)

trace_state_p = go.Scatter(
				x=match_list_p,
				y=np.ones(len(match_list_p)),
				mode='markers+text',
				name='Passed read start',
				text=match_text_p,
				textfont=dict(
						family='sans serif',
						size=10,
						color='#000000' 
					),
				marker = dict(
						color = 'rgb(55, 255, 55)',
						size = 6,
					),
				textposition='top center'
			)

trace_state_f = go.Scatter(
				x=match_list_f,
				y=np.ones(len(match_list_f)),
				mode='markers+text',
				name='Failed read start',
				text=match_text_f,
				textfont=dict(
						family='sans serif',
						size=10,
						color='#000000'
					),
				marker = dict(
						color = 'rgb(255, 55, 55)',
						size = 6,
					),
				textposition='top center'
			)

data.append(trace_state_p)
data.append(trace_state_f)

layout = go.Layout(title='Squiggle')

fig = go.Figure(data=data,layout=layout)
pyo.plot(fig, filename='rna_3_ch_60.html')   