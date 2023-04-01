import numpy as np
import matplotlib.pyplot as plt
import pdb

x = np.linspace(-100, 100, 10000)
a = np.pi/12
y = 1/(np.sqrt(2*np.pi)*a) * np.exp(-0.5 * (x/a)**2)
pdb.set_trace()
