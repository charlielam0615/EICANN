import numpy as np
from numpy.fft import fft, ifft
import matplotlib.pyplot as plt
import pdb

N = 100
a = np.pi/6
X = np.linspace(-np.pi, np.pi, N)
# weight matrix for Fourier Transform
W_EE = 1/np.sqrt(N) * np.exp(-np.minimum(X-(-np.pi), np.pi-X)**2/(2*a**2))
# weight matrix for matrix product
W_EE_full = 1/np.sqrt(N) * np.exp(-np.minimum(np.abs(X[None,]-X[:,None]), 2*np.pi-np.abs(X[None,]-X[:,None]))**2/(2*a**2))

center = np.pi/4
r_E = np.exp(-np.minimum(center-X, np.pi-center+X-(-np.pi))**2/(2*a**2))

W_IE = -np.ones(N) * (1/N)
W_IE_full = -np.ones([N,N]) * (1/N)
r_I = np.random.rand(N)

shunting_input_normal = (W_EE_full @ r_E) * (W_IE_full @ r_I)
shunting_input_fft = ifft(fft(W_EE)*fft(r_E)) * ifft(fft(W_IE)*fft(r_I))

plt.subplot(2,1,1)
plt.plot(shunting_input_normal)
plt.subplot(2,1,2)
plt.plot(shunting_input_fft)
plt.show()