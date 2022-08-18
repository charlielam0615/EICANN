import matplotlib.pyplot as plt
from numpy import genfromtxt
import numpy as np
import pdb

eicann_time = (genfromtxt('eicann_time.csv', delimiter=',') - 1990) * 0.01
scann_time = (genfromtxt('scann_time.csv', delimiter=',') - 1990) * 0.01

plt.hist(eicann_time, bins=10, density=True, alpha = 0.5, label='EI')
plt.hist(scann_time, bins=10, density=True, alpha = 0.5, label='cann')
plt.legend()
plt.xlabel(r't [ms]')
plt.show()
