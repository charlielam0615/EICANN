import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d
# import synapse

import pdb

bp.math.set_platform('cpu')

size_E, size_I = 128, 32
vth = -50
vreset = -60
tau_Es = 30
tau_Is = 10
gEE = 0.045
gEI = 0.0175
gIE = 0.1603
gII = 0.0082
ve, vi = 0, -70

class LIF(bp.dyn.NeuGroup):
    def __init__(self, size, cm, gl, vl, vth, vreset, tau_ref, **kwargs):
        super(LIF, self).__init__(size, **kwargs)

        # parameters
        self.size = size
        self.cm = cm
        self.gl = gl
        self.vl = vl
        self.vth = vth
        self.vreset = vreset
        self.tau_ref = tau_ref

        # variables
        self.V = bm.Variable(bm.random.randn(self.size) + vreset)
        self.input = bm.Variable(bm.zeros(self.size))
        self.t_last_spike = bm.Variable(bm.ones(self.size) * -1e7)
        self.refractory = bm.Variable(bm.zeros(self.size, dtype=bool))
        self.spike = bm.Variable(bm.zeros(self.size, dtype=bool))

        # integral
        # self.integral = bp.odeint(f=self.derivative, method='exp_auto')
        self.integral = bp.odeint(f=self.derivative, method='euler')

    def derivative(self, V, t, inputs):
        dvdt = (-self.gl*(-self.vl+V) + inputs) / self.cm
        return dvdt

    def update(self, dti):
        _t, _dt = dti.t, dti.dt
        refractory = (_t - self.t_last_spike) <= self.tau_ref
        V = self.integral(self.V, _t, self.input, dt=_dt)
        V = bm.where(refractory, self.V, V)
        spike = self.vth <= V
        self.spike.value = spike
        self.t_last_spike.value = bm.where(spike, _t, self.t_last_spike)
        self.V.value = bm.where(spike, self.vreset, V)
        self.refractory.value = bm.logical_or(refractory, spike)
        self.input[:] = 0.


class SCANN(bp.dyn.Network):
    def __init__(self):
        self.a = bm.sqrt(2*bm.pi) * (bm.pi/6)
        self.J = 4.
        self.A = 1.
        self.x = bm.linspace(-bm.pi, bm.pi, size_E)

        # neurons
        E = LIF(size=size_E, cm=0.5, gl=0.012, vl=-70, vth=vth, vreset=vreset, tau_ref=5)
        I = LIF(size=size_I, cm=0.2, gl=0.008, vl=-70, vth=vth, vreset=vreset, tau_ref=1)

        E.V[:] = bm.random.random(size_E) * (vth - vreset) + vreset
        I.V[:] = bm.random.random(size_I) * (vth - vreset) + vreset

        w_ee, w_ei, w_ie, w_ii = self.make_conn(self.x)

        # synapses
        self.E2E = bp.dyn.ExpCOBA(pre=E, post=E, conn=bp.connect.All2All(), tau=tau_Es, E=ve, g_max=gEE*w_ee)
        self.E2I = bp.dyn.ExpCOBA(pre=E, post=I, conn=bp.connect.All2All(), tau=tau_Es, E=ve, g_max=gEI*w_ei)
        self.I2I = bp.dyn.ExpCOBA(pre=I, post=I, conn=bp.connect.All2All(), tau=tau_Is, E=vi, g_max=gII*w_ii)
        self.I2E = bp.dyn.ExpCOBA(pre=I, post=E, conn=bp.connect.All2All(), tau=tau_Is, E=vi, g_max=gIE*w_ie)

        super(SCANN, self).__init__(self.E2E, self.E2I, self.I2E, self.I2I, E=E, I=I)

    def dist(self, d):
        d = bm.remainder(d, 2*bm.pi)
        d = bm.where(d > bm.pi, d - 2*bm.pi, d)
        return d

    def make_conn(self, x):
        x_left = bm.reshape(x, (-1,1))
        x_right = bm.repeat(x.reshape((1,-1)), len(x), axis=0)
        d = self.dist(x_left-x_right)
        w_ee_ = self.J * bm.exp(-bm.pi * bm.square(d / self.a))
        w_ee = w_ee_ / bm.sum(w_ee_, axis=-1, keepdims=True)
        const_conn = lambda w, p: (w > 1-p) / np.maximum(np.sum(w > 1-p, axis=-1, keepdims=True), 1)
        w_ei = const_conn(np.random.rand(size_E, size_I), 1.0)
        w_ii = const_conn(np.random.rand(size_I, size_I), 0.3)
        w_ie = const_conn(np.random.rand(size_I, size_E), 1.0)

        return w_ee, w_ei, w_ie, w_ii

    def get_stimulus_by_pos(self, pos):
        if bm.ndim(pos) <= 1:
            x = self.x
        elif bm.ndim(pos) == 2:
            x = self.x[None,]
        I = self.A * (-0.08 + 0.48*bm.exp(-bm.pi*bm.square(self.dist(x - pos) / self.a)))
        return I


net = SCANN()

# # ==== Persistent Activity ==== 
# dur = 1500
# n_step = int(0.5*dur / 0.01)
# pos = bm.linspace(-bm.pi/4, 0, n_step)[:,None]
# inputs = bm.zeros([n_step*2, size_E], dtype=bm.float32)
# inputs[:n_step] = net.get_stimulus_by_pos(pos)


# ===== Moving Bump ====
dur = 4000
n_step = int(dur / 0.01)
pos = bm.linspace(-bm.pi/2, 10*bm.pi/2, n_step)[:,None]
inputs = net.get_stimulus_by_pos(pos)



# ===== Persistent Activity ====
# inputs = net.get_stimulus_by_pos(0.)
# inputs, dur = bp.inputs.section_input(values=[inputs, 0.],
#                                          durations=[500., 1000.],
#                                          return_length=True)



runner = bp.dyn.DSRunner(net,
                         jit=True,
                         monitors=['E2I.g', 'E2E.g', 'I2E.g', 'I2I.g', 'I.V', 'E.V', 'E.spike', 'I.spike'],
                         inputs=[('E.input', inputs, 'iter'),
                                 ('I.input', 0.)],
                         dt=0.01)
t = runner(dur)


e2i_inp = runner.mon['E2I.g'] * (ve - runner.mon['I.V'])
e2e_inp = runner.mon['E2E.g'] * (ve - runner.mon['E.V'])
i2e_inp = runner.mon['I2E.g'] * (vi - runner.mon['E.V'])
i2i_inp = runner.mon['I2I.g'] * (vi - runner.mon['I.V'])


# # visualization
fig, gs = bp.visualize.get_figure(3, 1, 1.5, 10)

fig.add_subplot(gs[:3, 0])
bp.visualize.raster_plot(runner.mon.ts, runner.mon['E.spike'], xlim=(0, dur), ylim=[0,size_E], markersize=1.)
plt.show()
# fig.add_subplot(gs[3:, 0])
# bp.visualize.raster_plot(runner.mon.ts, runner.mon['I.spike'], xlim=(0, dur), ylim=[0,size_I], show=True)  

# Time window: 100ms with dt=0.01 ms
# firing_rate = convolve2d(runner.mon['E.spike'].astype(np.float32), np.ones([10000,1],dtype=np.float32), mode='same') / (10000*0.01/1000)
#
# bp.visualize.animate_1D(
#   dynamical_vars=[{'ys': 0.005*firing_rate, 'xs': net.x, 'legend': 'spike'},
#                   {'ys': inputs, 'xs': net.x, 'legend': 'Iext'}],
#   frame_step=1000,
#   frame_delay=50,
#   show=True,
#   save_path='cann-decoding.gif'
# )
