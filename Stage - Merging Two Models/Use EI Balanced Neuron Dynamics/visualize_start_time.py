import matplotlib.pyplot as plt
from numpy import genfromtxt
import numpy as np
import pdb

eicann_time = (genfromtxt('eicann_time.csv', delimiter=',') - 1900) * 0.01
scann_time = (genfromtxt('scann_time.csv', delimiter=',') - 1900) * 0.01

plt.hist(eicann_time, bins=50, density=True, alpha = 0.5)
plt.hist(scann_time, bins=10, density=True, alpha = 0.5)
plt.xlabel(r't [ms]')
plt.show()
