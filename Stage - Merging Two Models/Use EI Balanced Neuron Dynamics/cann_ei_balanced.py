import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d
from UnitExpCUBA import UnitExpCUBA

from brainpy.dyn import TwoEndConn, DynamicalSystem


import pdb

global_dt = 0.01

bp.math.set_platform('cpu')

size_E, size_I, size_ff = 750, 250, 1000
vth = 3
vreset = 0
tau_scale = 1.5 # 1
tau_Es = 0.6 *tau_scale
tau_Is = 0.5 *tau_scale
tau_E = 1.5 *tau_scale
tau_I = 1.0 *tau_scale

scale = 0.0125 * 6 # *2
gEE = 0.065 * 300 * 8 * scale # *7
gEI = 0.0175 * 750 * 6 * scale #*6
gIE = -0.1603 * 250 * 4 * scale
gII = -0.0082 * 750 * 30 * scale

# gl = 0.012 * 100 * scale
gl = 0.000268 * 20 * bm.sqrt(size_E+size_I)

shunting_k = 0.3
f_E = 1.1 * 0.15
f_I = 1.0 * 0.15
input_amp = 0.5


class Shunting(TwoEndConn):
    def __init__(self, E2Esyn, I2Esyn, k, EGroup):
        super().__init__(pre=E2Esyn.pre, post=I2Esyn.post, conn=None)
        self.E2Esyn = E2Esyn
        self.I2Esyn = I2Esyn
        self.EGroup = EGroup
        self.k = k

        assert self.E2Esyn.post == self.I2Esyn.post
        self.post = self.E2Esyn.post

    def update(self, t, dt):
        E_inp = self.E2Esyn.output_value + self.EGroup.ext_input
        I_inp = self.I2Esyn.output_value
        self.post.input += self.k * E_inp * I_inp


class LIF(bp.dyn.NeuGroup):
    def __init__(self, size, tau, gl, vth, vreset, tau_ref, **kwargs):
        super(LIF, self).__init__(size, **kwargs)

        # parameters
        self.size = size
        self.tau = tau
        self.vth = vth + (bm.random.rand(self.size) - 0.5) * 4
        self.gl = gl
        self.vreset = vreset
        self.tau_ref = tau_ref

        # variables
        self.V = bm.Variable(bm.zeros(self.size))
        self.input = bm.Variable(bm.zeros(self.size))
        self.t_last_spike = bm.Variable(bm.ones(self.size) * -1e7)
        self.refractory = bm.Variable(bm.zeros(self.size, dtype=bool))
        self.spike = bm.Variable(bm.zeros(self.size, dtype=bool))
        self.ext_input = bm.Variable(bm.zeros(self.size))

        # integral
        # self.integral = bp.odeint(f=self.derivative, method='exp_auto')
        self.integral = bp.odeint(f=self.derivative, method='euler')

    def derivative(self, V, t, inputs, ext_input):
        dvdt = (-self.gl*V + inputs + ext_input) / self.tau
        return dvdt

    def update(self, _t, _dt):
        refractory = (_t - self.t_last_spike) <= self.tau_ref
        V = self.integral(self.V, _t, self.input, self.ext_input, dt=_dt)
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
        self.A = input_amp
        self.x = bm.linspace(-bm.pi, bm.pi, size_E)

        # neurons
        self.E = LIF(size=size_E, gl=gl, tau=tau_E, vth=vth, vreset=vreset, tau_ref=0)
        self.I = LIF(size=size_I, gl=gl, tau=tau_I, vth=vth, vreset=vreset, tau_ref=0)

        self.E.V[:] = bm.random.random(size_E) * (self.E.vth - vreset) + vreset
        self.I.V[:] = bm.random.random(size_I) * (self.I.vth - vreset) + vreset

        w_ee, w_ei, w_ie, w_ii = self.make_conn(self.x)

        # synapses
        self.E2E = UnitExpCUBA(pre=self.E, post=self.E, conn=bp.connect.All2All(), tau=tau_Es, g_max=gEE*w_ee)
        self.E2I = UnitExpCUBA(pre=self.E, post=self.I, conn=bp.connect.All2All(), tau=tau_Es, g_max=gEI*w_ei)
        self.I2I = UnitExpCUBA(pre=self.I, post=self.I, conn=bp.connect.All2All(), tau=tau_Is, g_max=gII*w_ii)
        self.I2E = UnitExpCUBA(pre=self.I, post=self.E, conn=bp.connect.All2All(), tau=tau_Is, g_max=gIE*w_ie)
        self.ESI = Shunting(E2Esyn=self.E2E, I2Esyn=self.I2E, k=shunting_k, EGroup=self.E)

        super(SCANN, self).__init__()

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
        w_ii = const_conn(np.random.rand(size_I, size_I), 1.0)
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
# dur = 1000
# n_step = int(dur / global_dt)
# pos = bm.linspace(-2*bm.pi/3, 50*bm.pi/3, n_step)[:,None]
# E_inputs = net.get_stimulus_by_pos(pos)
# name = 'cann-moving.gif'


# ===== Persistent Activity ====
E_inputs = net.get_stimulus_by_pos(0.)
bg_inputs = 0.
E_inputs, dur = bp.inputs.section_input(values=[bg_inputs, bg_inputs+E_inputs, bg_inputs],
                                         durations=[20., 50., 50.],
                                         return_length=True,
                                         dt=global_dt)

I_inputs = bm.mean(E_inputs, axis=-1, keepdims=True)
noise = bm.random.randn(*E_inputs.shape) * 0
Einp_scale = size_ff * f_E / bm.sqrt(size_ff)
Iinp_scale = size_ff * f_I / bm.sqrt(size_ff)
name = 'cann-persistent.gif'

runner = bp.dyn.DSRunner(net,
                         jit=True,
                         monitors=['E2I.g', 'E2E.g', 'I2E.g', 'I2I.g', 'I.V', 'E.V', 'E.spike', 'I.spike'],
                         inputs=[('E.ext_input', Einp_scale * (E_inputs+noise), 'iter', '='),
                                 ('I.ext_input', Iinp_scale * I_inputs, 'iter', '=')],
                         dt=global_dt)
t = runner(dur)


# ==== raster plot =====
# fig, gs = bp.visualize.get_figure(2, 1, 1.5, 10)

# fig.add_subplot(gs[:1, 0])
# bp.visualize.raster_plot(runner.mon.ts, runner.mon['E.spike'], xlim=(0, dur), ylim=[0,size_E])
# plt.plot([20, 70], [350, 350], label='input peak', color='red')

# fig.add_subplot(gs[1:, 0])
# bp.visualize.raster_plot(runner.mon.ts, runner.mon['I.spike'], xlim=(0, dur), ylim=[0,size_I], show=True) 



# ===== Current Visualization =====
# Ec_inp = runner.mon['E2E.g']
# Fc_inp = Einp_scale * (E_inputs+noise)
# shunting_inp = shunting_k*(runner.mon['E2E.g']+Fc_inp)*runner.mon['I2E.g']
# r_SI = shunting_k*runner.mon['E2E.g']*runner.mon['I2E.g']
# Ic_inp = runner.mon['I2E.g'] + shunting_inp
# total_inp = Ec_inp + Ic_inp + Fc_inp

# fig, gs = bp.visualize.get_figure(4, 1, 1.5, 7)

# neuron_index = 375
# fig.add_subplot(gs[:2, 0])
# bp.visualize.line_plot(runner.mon.ts, total_inp[:,neuron_index], xlim=(0, dur), legend='Total')  
# bp.visualize.line_plot(runner.mon.ts, Fc_inp[:,neuron_index], xlim=(0, dur), legend='Fc')  
# bp.visualize.line_plot(runner.mon.ts, Ec_inp[:,neuron_index], xlim=(0, dur), legend='Exc Rec') 
# bp.visualize.line_plot(runner.mon.ts, (Ic_inp-shunting_inp)[:,neuron_index], xlim=(0, dur), legend='Inh (ex. SI)')  
# bp.visualize.line_plot(runner.mon.ts, shunting_inp[:,neuron_index], xlim=(0, dur), legend='SI')
# bp.visualize.line_plot(runner.mon.ts, -gl * runner.mon['E.V'][:,neuron_index], xlim=(0, dur), legend='Leak')

# fig.add_subplot(gs[2:, 0])
# bp.visualize.line_plot(runner.mon.ts, (Ic_inp-shunting_inp)[:,neuron_index], xlim=(0, dur), legend='I')  
# bp.visualize.line_plot(runner.mon.ts, r_SI[:,neuron_index], xlim=(0, dur), legend='SI (ex. Fc)', show=True) 




# ===== heat map =====

def moving_average(a, n, axis):
    ret = bm.cumsum(a, axis=axis, dtype=bm.float32)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

plt.figure()
T = 100
ma = moving_average(runner.mon['E.spike'], n=T, axis=0) # average window: 1 ms
bump_activity = bm.vstack([bm.sum(ma * bm.cos(net.x[None,]), axis=1),bm.sum(ma * bm.sin(net.x[None,]), axis=1)])
readout = bm.array([[1., 0.]]) @ bump_activity
firing_rate = ma / (T * global_dt / 1000) 
plt.subplot(2,1,1)
nm_readout = readout.T / bm.max(readout)
plt.plot(nm_readout)
plt.plot(E_inputs[T-1:,375] / bm.max(E_inputs[T-1:,375]))
# plt.xlim([int(runner.mon.ts.shape[0]*0.6), runner.mon.ts.shape[0]])
plt.xlim([0, runner.mon.ts.shape[0]])

print(bm.argmax(nm_readout>0.5))

plt.subplot(2,1,2)
plt.imshow(firing_rate.T, aspect='auto', cmap='gray_r')
plt.plot(bm.argmax(E_inputs, axis=1)[T-1:], label='input peak', color='red')
# plt.xlim([int(runner.mon.ts.shape[0]*0.6), runner.mon.ts.shape[0]])
plt.xlim([0, runner.mon.ts.shape[0]])
plt.show()


# ====== membrane potential ======
fig, gs = bp.visualize.get_figure(2, 2, 1.5, 7)
fig.add_subplot(gs[:1, 0])
neuron_index = 375
bp.visualize.line_plot(runner.mon.ts, runner.mon['E.V'][:,neuron_index], xlim=(0, dur), legend='E mem potential')

fig.add_subplot(gs[1:, 0])
neuron_index = 740
bp.visualize.line_plot(runner.mon.ts, runner.mon['E.V'][:,neuron_index], xlim=(0, dur), legend='E mem potential', show=True)




# # Time window: 100ms with dt=0.01 ms
# firing_rate = convolve2d(runner.mon['E.spike'].astype(np.float32), np.ones([10000,1],dtype=np.float32), mode='same') / (10000*global_dt/1000)

# bp.visualize.animate_1D(
#   dynamical_vars=[{'ys': 0.005*firing_rate, 'xs': net.x, 'legend': 'spike'},
#                   {'ys': E_inputs, 'xs': net.x, 'legend': 'Iext'}],
#   frame_step=1000,
#   frame_delay=50,
#   show=True,
#   save_path=name,
# )  
