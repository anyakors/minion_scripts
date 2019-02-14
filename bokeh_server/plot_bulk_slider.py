#plots all matches in a given bulk file + reads from one folder corresponding to this bulk file, marks them as pass/fail

import os
import h5py
import numpy as np

from bokeh.plotting import figure, output_file, show
from bokeh.layouts import column, row, widgetbox
from bokeh.models import CustomJS, ColumnDataSource, Slider

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
raw_fast5 = h5py.File('/media/mookse/DATA1/minion_data/bulk/mookse_Veriton_X4650G_20180613_FAH54029_MN21778_sequencing_run_RNA3_G4_false_7_59576.fast5')

signal = raw_fast5['Raw']['Channel_10']['Signal'].value

signal_trim = signal[0:500000]
x = np.arange(0,len(signal_trim))

source = ColumnDataSource(data=dict(x=x, y=signal_trim))

output_file("lines.html")
# create a new plot with a title and axis labels
# sizing_mode='stretch_both'
p = figure(title="bulk_matches", x_axis_label='t, 1/3012s', y_axis_label='I, pA', plot_width=800, plot_height=600)
# add a line renderer with legend and line thickness

p.line('x', 'y', source=source, legend="bulk", line_width=1)
# show the results

def callback(source=source, window=None):
    data = source.data
    f = cb_obj.value                         # NOQA
    x, y = data['x'], data['y']
    y = signal[f*500000:(f+1)*500000]
    x = np.arange(f*500000,(f+1)*500000)
    source.change.emit();

frames = int(len(signal)/500000) + 1

slider = Slider(start=1, end=frames+1, value=1, step=1, title="frame",
                callback=CustomJS.from_py_func(callback))

phase_slider = Slider(start=0, end=6.4, value=0, step=.1,
                      title="Phase", callback=CustomJS.from_py_func(callback))

layout = column(
    widgetbox(slider, phase_slider),
    p,
    )

show(p)