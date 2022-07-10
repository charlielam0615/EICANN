import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d

from brainpy.dyn import TwoEndConn, DynamicalSystem


import pdb

bp.math.set_platform('cpu')

size_E, size_Ip, size_Id, size_ff = 1000, 250, 250, 250
vth = 10
vreset = 0
tau_Es = 6. 
tau_Is = 5. 
tau_E = 15.0
tau_I = 10.0
gEE = 0.065 * 300 * 4
gEIp = 0.0175 * 750 * 8
gIpE = -0.1603 * 250 * 2
gIpIp = -0.0082 * 750 * 30
shunting_k = 1.0

f_E = 6
f_I = 4
mu_f = bm.array(1.0)
# gEE = 0.25 / bm.sqrt(size_E)
gIdE = -1. / bm.sqrt(size_E)
gEId = 0.4 / bm.sqrt(size_E)
gIdId = -1. / bm.sqrt(size_E)
prob = 0.25


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


class EICANN(bp.dyn.Network):
    def __init__(self):
        self.a = bm.sqrt(2*bm.pi) * (bm.pi/6)
        self.J = 4.
        # self.A = 0.48
        self.A = 2.
        self.x = bm.linspace(-bm.pi, bm.pi, size_E)

        # neurons
        self.E = LIF(size=size_E, tau=tau_E, vth=vth, vreset=vreset, tau_ref=0)
        self.Ip = LIF(size=size_Ip, tau=tau_I, vth=vth, vreset=vreset, tau_ref=0)
        self.Id = LIF(size=size_Id, tau=tau_I, vth=vth, vreset=vreset, tau_ref=0)

        self.E.V[:] = bm.random.random(size_E) * (vth - vreset) + vreset
        self.Ip.V[:] = bm.random.random(size_Ip) * (vth - vreset) + vreset
        self.Id.V[:] = bm.random.random(size_Id) * (vth - vreset) + vreset

        w_ee, w_ei, w_ie, w_ii = self.make_conn(self.x)

        # synapses
        self.E2E = bp.dyn.ExpCUBA(pre=self.E, post=self.E, conn=bp.connect.All2All(), tau=tau_Es, g_max=gEE*w_ee)
        self.E2Ip = bp.dyn.ExpCUBA(pre=self.E, post=self.Ip, conn=bp.connect.All2All(), tau=tau_Es, g_max=gEIp*w_ei)
        self.Ip2Ip = bp.dyn.ExpCUBA(pre=self.Ip, post=self.Ip, conn=bp.connect.All2All(), tau=tau_Is, g_max=gIpIp*w_ii)
        self.Ip2E = bp.dyn.ExpCUBA(pre=self.Ip, post=self.E, conn=bp.connect.All2All(), tau=tau_Is, g_max=gIpE*w_ie)
        self.ESI = Shunting(E2Esyn=self.E2E, I2Esyn=self.Ip2E, k=shunting_k)

        self.E2Id = bp.dyn.ExpCUBA(pre=self.E, post=self.Id, conn=bp.conn.FixedProb(prob), tau=tau_Es, g_max=gEId)
        self.Id2Id = bp.dyn.ExpCUBA(pre=self.Id, post=self.Id, conn=bp.conn.FixedProb(prob), tau=tau_Is, g_max=gIdId)
        self.Id2E = bp.dyn.ExpCUBA(pre=self.Id, post=self.E, conn=bp.conn.FixedProb(prob), tau=tau_Is, g_max=gIdE)

        # super(SCANN, self).__init__(self.E2E, self.E2Ip, self.Ip2E, self.Ip2Ip, self.ESI, self.E2Id, self.Id2Id, self.Id2E, E=E, Ip=Ip, Id=Id)
        super(EICANN, self).__init__()

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
        w_ei = const_conn(np.random.rand(size_E, size_Ip), 1.0)
        w_ii = const_conn(np.random.rand(size_Ip, size_Ip), 1.0)
        w_ie = const_conn(np.random.rand(size_Ip, size_E), 1.0)

        return w_ee, w_ei, w_ie, w_ii

    def get_stimulus_by_pos(self, pos):
        if bm.ndim(pos) <= 1:
            x = self.x
        elif bm.ndim(pos) == 2:
            x = self.x[None,]
        I = self.A * bm.exp(-bm.pi*bm.square(self.dist(x - pos) / self.a))
        return I


net = EICANN()


WEF = f_E/bm.sqrt(size_ff)
WIF = f_I/bm.sqrt(size_ff)
Einp_scale = size_ff * WEF
Iinp_scale = size_ff * WIF


# ===== Moving Bump ====
dur = 2000
n_step = int(dur / 0.01)
pos = bm.linspace(-bm.pi/2, bm.pi/2, n_step)[:,None]
E_inputs = net.get_stimulus_by_pos(pos)
I_inputs = bm.mean(E_inputs, axis=-1, keepdims=True)
name = 'cann-moving.gif'


# ===== Persistent Activity ====
# inputs = net.get_stimulus_by_pos(0.)
# E_inputs, dur = bp.inputs.section_input(values=[inputs, 0.],
#                                          durations=[500., 500.],
#                                          return_length=True,
#                                          dt=0.01)
# I_inputs = bm.mean(E_inputs, axis=-1, keepdims=True)
# name = 'cann-persistent.gif'

runner = bp.dyn.DSRunner(net,
                         jit=True,
                         monitors=['E2E.g', 'Ip2E.g', 'Id2E.g', 'Ip.V', 'E.V', 'E.spike', 'Ip.spike', 'Id.spike', 'E.input'],
                         inputs=[('E.input',  Einp_scale*E_inputs, 'iter'),
                                 ('Id.input', Iinp_scale*I_inputs, 'iter')],
                         dt=0.01)
t = runner(dur)



# ===== Raster Plot =====
# fig, gs = bp.visualize.get_figure(5, 1, 1.5, 10)

# fig.add_subplot(gs[:3, 0])
# bp.visualize.raster_plot(runner.mon.ts, runner.mon['E.spike'], xlim=(0, dur), ylim=[0,size_E])
# bp.visualize.line_plot(runner.mon.ts, bm.argmax(E_inputs, axis=1), xlim=(0, dur), legend='input peak')

# fig.add_subplot(gs[3:, 0])
# bp.visualize.raster_plot(runner.mon.ts, runner.mon['Ip.spike'], xlim=(0, dur), ylim=[0,size_Ip], show=True)  



# === GIF Time window: 100ms with dt=0.01 ms ======
# firing_rate = convolve2d(runner.mon['E.spike'].astype(np.float32), np.ones([10000,1],dtype=np.float32), mode='same') / (10000*0.01/1000)

# bp.visualize.animate_1D(
#   dynamical_vars=[{'ys': 0.005*firing_rate, 'xs': net.x, 'legend': 'spike'},
#                   {'ys': inputs, 'xs': net.x, 'legend': 'Iext'}],
#   frame_step=1000,
#   frame_delay=50,
#   show=True,
#   save_path=name,
# )  



# ===== Current Visualization =====
total_inp = runner.mon['E2E.g'] + runner.mon['Ip2E.g'] + shunting_k*runner.mon['E2E.g']*runner.mon['Ip2E.g'] + runner.mon['Id2E.g'] + Einp_scale*E_inputs
Ec_inp = runner.mon['E2E.g']
Ic_inp = runner.mon['Id2E.g'] + runner.mon['Ip2E.g'] + shunting_k*runner.mon['E2E.g']*runner.mon['Ip2E.g']
Fc_inp = Einp_scale*E_inputs


fig, gs = bp.visualize.get_figure(3, 1, 1.5, 7)
fig.add_subplot(gs[:1, 0])
bp.visualize.line_plot(runner.mon.ts, total_inp[:,500], xlim=(0, dur), legend='Total')  
bp.visualize.line_plot(runner.mon.ts, Ec_inp[:,500]+Fc_inp[:,500], xlim=(0, dur), legend='Excitatory') 
bp.visualize.line_plot(runner.mon.ts, Ic_inp[:,500], xlim=(0, dur), legend='Inhibitory')  

fig.add_subplot(gs[1:2, 0])
bp.visualize.line_plot(runner.mon.ts, runner.mon['E.V'][:,500], xlim=(0, dur), legend='membrane potential')  

fig.add_subplot(gs[2:3, 0])
bp.visualize.raster_plot(runner.mon.ts, runner.mon['E.spike'][:, 495:505], xlim=(0, dur), show=True)

# fig.add_subplot(gs[3:4, 0])
# bp.visualize.raster_plot(runner.mon.ts, runner.mon['Ip.spike'], xlim=(0, dur))

# fig.add_subplot(gs[4:, 0])
# bp.visualize.raster_plot(runner.mon.ts, runner.mon['Id.spike'], xlim=(0, dur), show=True)


