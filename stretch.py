import numpy as np
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt

x = np.arange(0, 6, 0.01)
a = np.sin(x**2/5)

old_indices = np.arange(0,len(a))
new_length = 15
new_indices = np.linspace(0,len(a)-1,new_length)
spl = UnivariateSpline(old_indices, a, k=3, s=0)
new_array = spl(new_indices)

print("a =", a, "interp =", new_array)

print(new_indices)

plt.plot(new_indices, new_array)
plt.show()