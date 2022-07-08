import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d

from brainpy.dyn import TwoEndConn, DynamicalSystem


import pdb

bp.math.set_platform('cpu')

size_E, size_I = 1000, 250
vth = 10
vreset = 0
tau_Es = 6. 
tau_Is = 5. 
tau_E = 15.0
tau_I = 10.0
gEE = 0.065 * 300 * 7
gEI = 0.0175 * 750 * 4.5
gIE = -0.1603 * 250 * 0.8
gII = -0.0082 * 750 * 3
shunting_k = 1.0


class Shunting(TwoEndConn):
    def __init__(self, E2Esyn, I2Esyn, k):
        super().__init__(pre=E2Esyn.pre, post=I2Esyn.post, conn=None)
        self.E2Esyn = E2Esyn
        self.I2Esyn = I2Esyn
        self.k = k

        assert self.E2Esyn.post == self.I2Esyn.post
        self.post = self.E2Esyn.post

    def update(self, t, dt):
        E_inp = self.E2Esyn.output_value
        I_inp = self.I2Esyn.output_value
        self.post.input += self.k * E_inp * I_inp


class LIF(bp.dyn.NeuGroup):
    def __init__(self, size, tau, vth, vreset, tau_ref, **kwargs):
        super(LIF, self).__init__(size, **kwargs)

        # parameters
        self.size = size
        self.tau = tau
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
        dvdt = inputs / self.tau
        return dvdt

    def update(self, _t, _dt):
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
        # self.A = 0.48
        self.A = 2.
        self.x = bm.linspace(-bm.pi, bm.pi, size_E)

        # neurons
        E = LIF(size=size_E, tau=tau_E, vth=vth, vreset=vreset, tau_ref=5)
        I = LIF(size=size_I, tau=tau_I, vth=vth, vreset=vreset, tau_ref=1)

        E.V[:] = bm.random.random(size_E) * (vth - vreset) + vreset
        I.V[:] = bm.random.random(size_I) * (vth - vreset) + vreset

        w_ee, w_ei, w_ie, w_ii = self.make_conn(self.x)

        # synapses
        self.E2E = bp.dyn.ExpCUBA(pre=E, post=E, conn=bp.connect.All2All(), tau=tau_Es, g_max=gEE*w_ee)
        self.E2I = bp.dyn.ExpCUBA(pre=E, post=I, conn=bp.connect.All2All(), tau=tau_Es, g_max=gEI*w_ei)
        self.I2I = bp.dyn.ExpCUBA(pre=I, post=I, conn=bp.connect.All2All(), tau=tau_Is, g_max=gII*w_ii)
        self.I2E = bp.dyn.ExpCUBA(pre=I, post=E, conn=bp.connect.All2All(), tau=tau_Is, g_max=gIE*w_ie)
        self.ESI = Shunting(E2Esyn=self.E2E, I2Esyn=self.I2E, k=shunting_k)

        super(SCANN, self).__init__(self.E2E, self.E2I, self.I2E, self.I2I, self.ESI, E=E, I=I)

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
        I = self.A * bm.exp(-bm.pi*bm.square(self.dist(x - pos) / self.a))
        return I


net = SCANN()


# ===== Moving Bump ====
# dur = 2000
# n_step = int(dur / 0.01)
# pos = bm.linspace(-bm.pi/2, bm.pi/2, n_step)[:,None]
# inputs = net.get_stimulus_by_pos(pos)
# name = 'cann-moving.gif'


# ===== Persistent Activity ====
inputs = net.get_stimulus_by_pos(0.)
inputs, dur = bp.inputs.section_input(values=[inputs, 0.],
                                         durations=[500., 500.],
                                         return_length=True,
                                         dt=0.01)
name = 'cann-persistent.gif'



runner = bp.dyn.DSRunner(net,
                         jit=True,
                         monitors=['E2I.g', 'E2E.g', 'I2E.g', 'I2I.g', 'I.V', 'E.V', 'E.spike', 'I.spike'],
                         inputs=[('E.input', inputs, 'iter'),
                                 ('I.input', 0.)],
                         dt=0.01)
t = runner(dur)

# e2i_inp = runner.mon['E2I.g'] * (ve - runner.mon['I.V'])
# e2e_inp = runner.mon['E2E.g'] * (ve - runner.mon['E.V'])
# i2e_inp = runner.mon['I2E.g'] * (vi - runner.mon['E.V'])
# i2i_inp = runner.mon['I2I.g'] * (vi - runner.mon['I.V'])


# # visualization
fig, gs = bp.visualize.get_figure(5, 1, 1.5, 10)

fig.add_subplot(gs[:3, 0])
bp.visualize.raster_plot(runner.mon.ts, runner.mon['E.spike'], xlim=(0, dur), ylim=[0,size_E])

fig.add_subplot(gs[3:, 0])
bp.visualize.raster_plot(runner.mon.ts, runner.mon['I.spike'], xlim=(0, dur), ylim=[0,size_I], show=True)  



# # Time window: 100ms with dt=0.01 ms
# firing_rate = convolve2d(runner.mon['E.spike'].astype(np.float32), np.ones([10000,1],dtype=np.float32), mode='same') / (10000*0.01/1000)

# bp.visualize.animate_1D(
#   dynamical_vars=[{'ys': 0.005*firing_rate, 'xs': net.x, 'legend': 'spike'},
#                   {'ys': inputs, 'xs': net.x, 'legend': 'Iext'}],
#   frame_step=1000,
#   frame_delay=50,
#   show=True,
#   save_path=name,
# )  
