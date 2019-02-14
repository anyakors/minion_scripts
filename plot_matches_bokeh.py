#plots all matches in a given bulk file + reads from one folder corresponding to this bulk file, marks them as pass/fail

import os
import h5py
import numpy as np

from bokeh.plotting import figure, output_file, show

def rolling_window(a, size):
    shape = a.shape[:-1] + (a.shape[-1] - size + 1, size)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def fscatter(p, x, y, marker):
    p.scatter(x, y, marker=marker, size=10,
              line_color="navy", fill_color="red", alpha=0.75)

def pscatter(p, x, y, marker):
    p.scatter(x, y, marker=marker, size=10,
              line_color="navy", fill_color="green", alpha=0.75)

def mtext(p, x, y, text):
    p.text(x, y, text=[text],
           text_color="grey", text_align="center", text_font_size="10pt")


#raw_fast5 = h5py.File('/media/mookse/DATA1/minion/bulk/mookse_Veriton_X4650G_20180618_FAH54070_MN21778_sequencing_run_RNA5_long_G4_49077.fast5')
raw_fast5 = h5py.File('/media/mookse/DATA1/minion_data/bulk/mookse_Veriton_X4650G_20180614_FAH54029_MN21778_sequencing_run_RNA9_mut_55737.fast5')

signal = raw_fast5['Raw']['Channel_10']['Signal'].value

#===================PASS=====================

match_list_p = []
match_list_p_end = []
match_text_p = []
#fast5s = os.listdir('/home/mookse/workspace/MinKNOW/data/reads/20180618_0526_RNA5_long_G4/alb_rna5/workspace/pass')
fast5s = os.listdir('/home/mookse/Desktop/Analysis/RNA9_short/alb_bc/workspace/pass/0')

for fast5 in fast5s:
	if fast5.endswith('ch_10_strand.fast5'):
		f = h5py.File('/home/mookse/Desktop/Analysis/RNA9_short/alb_bc/workspace/pass/0/{}'.format(fast5), 'r')
		read_no = list(f['Raw']['Reads'].keys())
		print(read_no)
		signal_read = f['Raw']['Reads'][read_no[0]]['Signal'].value
		bool_indices = np.all(rolling_window(signal, 10) == signal_read[0:10], axis=1)
		match = np.flatnonzero(bool_indices)
		match_list_p.append(match[0])
		match_text_p.append(read_no[0])

		bool_indices = np.all(rolling_window(signal, 10) == signal_read[-10:], axis=1) #match end
		match = np.flatnonzero(bool_indices)
		match_list_p_end.append(match[0])

#===================FAIL=====================

match_list_f = []
match_list_f_end = []
match_text_f = []
#fast5s = os.listdir('/home/mookse/workspace/MinKNOW/data/reads/20180618_0526_RNA5_long_G4/alb_rna5/workspace/fail')
fast5s = os.listdir('/home/mookse/Desktop/Analysis/RNA9_short/alb_bc/workspace/fail/0')

for fast5 in fast5s:
	if fast5.endswith('ch_10_strand.fast5'):
		f = h5py.File('/home/mookse/Desktop/Analysis/RNA9_short/alb_bc/workspace/fail/0/{}'.format(fast5), 'r')
		read_no = list(f['Raw']['Reads'].keys())
		print(read_no)
		signal_read = f['Raw']['Reads'][read_no[0]]['Signal'].value
		bool_indices = np.all(rolling_window(signal, 10) == signal_read[0:10], axis=1) #match start
		match = np.flatnonzero(bool_indices)
		match_list_f.append(match[0])
		match_text_f.append(read_no[0])

		bool_indices = np.all(rolling_window(signal, 10) == signal_read[-10:], axis=1) #match end
		match = np.flatnonzero(bool_indices)
		match_list_f_end.append(match[0])

signal_trim = signal[0:1000000]
x = np.arange(0,len(signal_trim))

output_file("lines.html")
# create a new plot with a title and axis labels
p = figure(title="bulk_matches", x_axis_label='t, 1/3012s', y_axis_label='I, pA', sizing_mode='stretch_both')
# add a line renderer with legend and line thickness
p.line(x, signal_trim, legend="bulk", line_width=1)
# show the results

pscatter(p, match_list_p, np.ones(len(match_list_p)), 'circle')
pscatter(p, match_list_p_end, np.ones(len(match_list_p_end)), 'circle_x')
mtext(p, match_list_p, np.ones(len(match_list_p))+3, match_text_p)

fscatter(p, match_list_f, np.ones(len(match_list_f)), 'circle')
fscatter(p, match_list_f_end, np.ones(len(match_list_f_end)), 'circle_x')
mtext(p, match_list_f, np.ones(len(match_list_f))+3, match_text_f)

show(p)