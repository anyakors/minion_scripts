from tombo import tombo_helper, tombo_stats, resquiggle
import h5py, mappy
import matplotlib.pyplot as plt
import numpy as np


#tombo_model = '/home/mookse/anaconda3/pkgs/ont-tombo-1.4-py36r341h24bf2e0_0/lib/python3.6/site-packages/tombo/tombo_models/tombo.DNA.model'
tombo_model = '/home/mookse/anaconda3/pkgs/ont-tombo-1.5-py36r341h24bf2e0_0/lib/python3.6/site-packages/tombo/tombo_models/tombo.DNA.model'
reference_fn = 'GGCTTCTTCTTGCTCTTAGGTAGTAGGTTC'

instance = tombo_stats.TomboModel(tombo_model)
print('Class methods available:', [func for func in dir(tombo_stats.TomboModel) if callable(getattr(tombo_stats.TomboModel, func))])

std_model = instance.get_exp_levels_from_seq(reference_fn, rev_strand=False)

print(len(std_model[0]))
print(len(reference_fn))

model_new = np.repeat(std_model[0], 10, axis=0)

plt.plot(np.arange(0, len(model_new)), model_new)
plt.show()