import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn')

plt.rcdefaults()
fig, ax = plt.subplots()

# Example data
molecule = ('rna3', 'rna3*', 'rna9', 'rna5', 'rna6', 'rna11')
y_pos = np.arange(len(molecule))
performance = [26660.2, 44132.65, 10169.2, 45476, 38291, 7956]
error = [90666.5, 135540, 43349.1, 175108, 147106, 7933]

ax.barh(y_pos, performance, xerr=error, align='center',
        color='lightsteelblue', ecolor='0.25')
ax.set_yticks(y_pos)
ax.set_xlim(0, None)
ax.set_yticklabels(molecule)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('average raw read length')

plt.show()