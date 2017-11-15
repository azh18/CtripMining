import numpy as np
from scipy.interpolate import UnivariateSpline
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt

import pylab as plt
# import numpy as np

# Sample data
side = np.linspace(-2,2,15)
X,Y = np.meshgrid(side,side)
Z = np.exp(-((X-1)**2+Y**2))

# Plot the density map using nearest-neighbor interpolation
plt.pcolormesh(X,Y,Z)
plt.colorbar()
# plt.show()
plt.savefig('heatmap_test.pdf')






