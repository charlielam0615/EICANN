import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
from scipy import signal

plt.style.use('ggplot')
dt = 0.1
duration = 1000.
N = int(duration/dt)
freq = 100.
fs = 1/dt*100
ts = bm.arange(0, N) * dt
time = ts / 1000
gain = 0.5*(bm.sin(2*bm.pi*freq*time)+1) * 0.001
spike = bm.random.rand(N, 750) < gain[:, None]

fig, gs = bp.visualize.get_figure(3, 3, 2, 2)
stair_dur = 1000.
stair_num = 1
frE = bp.measure.firing_rate(spike, width=5, dt=dt)
# E raster plots
fig.add_subplot(gs[0, :3])
bp.visualize.raster_plot(ts, spike, xlim=(0., 1000.), markersize=1., alpha=0.5)
# E firing rates
fig.add_subplot(gs[1, :3])
bp.visualize.line_plot(ts, frE, xlim=(0., 1000.))
# spectral analysis
ax = fig.add_subplot(3, 3, 7)
start = 0
end = N
sig = frE[start+2000:end-2000]
sig = sig - bm.mean(sig)
# f, Pxx_den = signal.welch(sig, fs=1/global_dt*100, nperseg=10*1024)
f, Pxx_den = signal.periodogram(sig, fs=1/dt*1000)
ax.semilogy(f, Pxx_den)
ax.set_xlim([-10, 500])
plt.tight_layout()
plt.show()