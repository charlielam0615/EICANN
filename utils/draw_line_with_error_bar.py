import numpy as np
import matplotlib.pyplot as plt

x = np.array([0.2, 0.4, 0.6])
# Fast CANN + EI balance
y_1 = np.array([0.001469717, 0.00548109725, 0.0273983789])
yerr_1 = np.array([0.000397065617, 0.001624004546, 0.0094596945])
# Slow CANN + EI balance
y_2 = np.array([0.00093133859, 0.00297619819, 0.0192002511])
yerr_2 = np.array([0.00028096710, 0.00086133618, 0.00868763492378367])
# Fast CANN
y_3 = np.array([0.010457518, 0.0467914, 0.208805307])
yerr_3 = np.array([0.0012138791, 0.00857175, 0.082133788])
# Slow CANN
y_4 = np.array([0.00513407412, 0.0309202357, 0.174637828])
yerr_4 = np.array([0.000644729614, 0.006673775, 0.0520009])

plt.style.use('ggplot')
fig = plt.figure(figsize=(6, 2.5))
ax = plt.subplot()
# ax.errorbar(x, y_1, yerr=yerr_1, label="Fast CANN + EI balance", alpha=0.5)
ax.errorbar(x, y_2, yerr=yerr_2, label="Slow CANN + EI balance", alpha=0.5)
# ax.errorbar(x, y_3, yerr=yerr_3, label="Fast CANN", alpha=0.5)
ax.errorbar(x, y_4, yerr=yerr_4, label="Slow CANN", alpha=0.5)
plt.legend(loc='upper left')
plt.ylabel('Decoding Error (rad)')
plt.xlabel('Noise Level')
plt.grid('on')
plt.tight_layout()
plt.show()
