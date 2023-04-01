import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
import numpy as np
from UnitExpCUBA import UnitExpCUBA

from brainpy.dyn import TwoEndConn, DynamicalSystem


import pdb

global_dt = 0.01

bp.math.set_platform('cpu')

size_E, size_I, size_ff = 750, 250, 1000
num = size_E + size_I
vth = 6
vreset = 0
tau_scale = 4
tau_Fs = 1.0
tau_Es = 0.5 * tau_scale
tau_Is = 0.25 * tau_scale
tau_F = 1.0
tau_E = 3 * tau_scale
tau_I = 2 * tau_scale

scale = 3.0 # 50
input_scale = 1.0
f_E = 1.1 * scale
f_I = 1. * scale
jie = -1. * scale * 1
jii = -1. * scale * 1
jee = 0.35 * scale * 1
jei = 0.4 * scale * 1
JIE = jie / bm.sqrt(num)
JII = jii / bm.sqrt(num)
JEE = jee / bm.sqrt(num)
JEI = jei / bm.sqrt(num)
JFE = f_E / bm.sqrt(size_ff)
JFI = f_I / bm.sqrt(size_ff)

gl = 0.000268 * 1 * bm.sqrt(size_E+size_I)

input_amp = 1.0
bg_str = 0.1

sigma_rec = bm.pi/6
sigma_ff = bm.pi/5
sigma_inp = bm.pi/12


class LIF(bp.dyn.NeuGroup):
    def __init__(self, size, tau, gl, vth, vreset, tau_ref, **kwargs):
        super(LIF, self).__init__(size, **kwargs)

        # parameters
        self.size = size
        self.tau = tau
        self.vth = vth + (bm.random.rand(self.size) - 0.5) * 0
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


class RCANN(bp.dyn.Network):
    def __init__(self):
        self.rec_a = bm.sqrt(2*bm.pi) * sigma_rec
        self.ff_a = bm.sqrt(2*bm.pi) * sigma_ff
        self.input_a = bm.sqrt(2*bm.pi) * sigma_inp
        self.A = input_amp
        self.Ex = bm.linspace(-bm.pi, bm.pi, size_E)
        self.Ix = bm.linspace(-bm.pi, bm.pi, size_I)
        self.Fx = bm.linspace(-bm.pi, bm.pi, size_ff)

        # neurons
        self.E = LIF(size=size_E, gl=gl, tau=tau_E, vth=vth, vreset=vreset, tau_ref=0)
        self.I = LIF(size=size_I, gl=gl, tau=tau_I, vth=vth, vreset=vreset, tau_ref=0)
        self.F = LIF(size=size_ff, gl=gl, tau=tau_F, vth=1, vreset=vreset, tau_ref=0)

        self.E.V[:] = bm.random.random(size_E) * (self.E.vth - vreset) + vreset
        self.I.V[:] = bm.random.random(size_I) * (self.I.vth - vreset) + vreset
        self.F.V[:] = bm.random.random(size_ff) * (self.F.vth - vreset) + vreset

        w_ee, w_ei, w_ie, w_ii, w_fe, w_fi = self.make_conn(size_E, size_I, size_ff)

        # synapses
        self.F2E = UnitExpCUBA(pre=self.F, post=self.E, conn=bp.connect.All2All(), tau=tau_Fs, g_max=JFE*w_fe)
        self.F2I = UnitExpCUBA(pre=self.F, post=self.I, conn=bp.connect.All2All(), tau=tau_Fs, g_max=JFI*w_fi)
        self.E2E = UnitExpCUBA(pre=self.E, post=self.E, conn=bp.connect.All2All(), tau=tau_Es, g_max=JEE*w_ee)
        self.E2I = UnitExpCUBA(pre=self.E, post=self.I, conn=bp.connect.All2All(), tau=tau_Es, g_max=JEI*w_ei)
        self.I2I = UnitExpCUBA(pre=self.I, post=self.I, conn=bp.connect.All2All(), tau=tau_Is, g_max=JII*w_ii)
        self.I2E = UnitExpCUBA(pre=self.I, post=self.E, conn=bp.connect.All2All(), tau=tau_Is, g_max=JIE*w_ie)

        super(RCANN, self).__init__()

    def dist(self, d):
        d = bm.remainder(d, 2*bm.pi)
        d = bm.where(d > bm.pi, d - 2*bm.pi, d)
        return d

    def make_conn(self, size_E, size_I, size_ff):
        
        def get_p(x_left, x_right, a):
            d = self.dist(x_left-x_right)
            p = bm.exp(-bm.pi * bm.square(d / a))
            return p

        def get_w(p, r):
            conn = (r > (1-p)).astype(bm.float32)
            return conn

        x_left = bm.reshape(self.Ex, (-1,1))
        x_right = bm.reshape(self.Ex, (1,-1))
        p_ee = get_p(x_left, x_right, self.rec_a)
        w_ee = get_w(p_ee, bm.random.rand(size_E, size_E))

        x_left = bm.reshape(self.Ex, (-1,1))
        x_right = bm.reshape(self.Ix, (1,-1))
        p_ei = get_p(x_left, x_right, self.rec_a)
        w_ei = get_w(p_ei, bm.random.rand(size_E, size_I))

        x_left = bm.reshape(self.Ix, (-1,1))
        x_right = bm.reshape(self.Ex, (1,-1))
        p_ie = get_p(x_left, x_right, self.rec_a)
        w_ie = get_w(p_ie, bm.random.rand(size_I, size_E))

        x_left = bm.reshape(self.Ix, (-1,1))
        x_right = bm.reshape(self.Ix, (1,-1))
        p_ii = get_p(x_left, x_right, self.rec_a)
        w_ii = get_w(p_ii, bm.random.rand(size_I, size_I))

        x_left = bm.reshape(self.Fx, (-1,1))
        x_right = bm.reshape(self.Ex, (1,-1))
        p_fe = get_p(x_left, x_right, self.ff_a)
        w_fe = get_w(p_fe, bm.random.rand(size_ff, size_E))

        x_left = bm.reshape(self.Fx, (-1,1))
        x_right = bm.reshape(self.Ix, (1,-1))
        p_fi = get_p(x_left, x_right, self.ff_a)
        w_fi = get_w(p_fi, bm.random.rand(size_ff, size_I))

        return w_ee, w_ei, w_ie, w_ii, w_fe, w_fi

    def get_stimulus_by_pos(self, pos):
        if bm.ndim(pos) <= 1:
            Fx = self.Fx
        elif bm.ndim(pos) == 2:
            Fx = self.Fx[None,]
        I_F = self.A/self.input_a * bm.exp(-bm.pi*bm.square(self.dist(Fx - pos) / self.input_a))

        return I_F


net = RCANN()


# ===== Persistent Activity ====
F_bump = net.get_stimulus_by_pos(0.)
bg_input = bg_str * net.A
F_inputs, dur = bp.inputs.section_input(values=[bg_input, bg_input+F_bump, bg_input],
                                         durations=[20., 50., 50.],
                                         return_length=True,
                                         dt=global_dt)
F_inputs = input_scale * F_inputs
F_noise = bm.random.randn(*F_inputs.shape) * bm.sqrt(F_inputs)
F_inp = F_inputs + F_noise


runner = bp.dyn.DSRunner(net,
                         jit=True,
                         monitors=['E2I.g', 'E2E.g', 'I2E.g', 'I2I.g', 'I.V', 'E.V', 'E.spike', 'I.spike', 'F.spike', 'F.V', 'F2E.g'],
                         inputs=[('F.ext_input', F_inp, 'iter', '=')],
                         dt=global_dt)
t = runner(dur)

# ==== debug =====
# fig, gs = bp.visualize.get_figure(4, 1, 1.5, 10)
#
# fig.add_subplot(gs[:2, 0])
# bp.visualize.raster_plot(runner.mon.ts, runner.mon['F.spike'], xlim=(0, dur), ylim=[0,size_ff])
# plt.plot([20, 70], [500, 500], label='input peak', color='red')
# fig.add_subplot(gs[2:, 0])
# bp.visualize.line_plot(runner.mon.ts, runner.mon['F.V'][:,500], xlim=(0, dur), show=True)



# ==== raster plot =====
fig, gs = bp.visualize.get_figure(4, 1, 1.5, 10)

fig.add_subplot(gs[:2, 0])
bp.visualize.raster_plot(runner.mon.ts, runner.mon['E.spike'], xlim=(0, dur), ylim=[0,size_E])
plt.plot([20, 70], [375, 375], label='input peak', color='red')

fig.add_subplot(gs[2:, 0])
bp.visualize.raster_plot(runner.mon.ts, runner.mon['I.spike'], xlim=(0, dur), ylim=[0,size_I], show=True) 


# ==== membrane potential hist =====
# plt.hist(bm.reshape(runner.mon['E.V'][1000, :], [-1]), bins=100, density=True)
# plt.show()


# ===== Current Visualization =====
Ec_inp = runner.mon['E2E.g']
Fc_inp = runner.mon['F2E.g']
Ic_inp = runner.mon['I2E.g'] 
total_inp = Ec_inp + Ic_inp + Fc_inp - gl * runner.mon['E.V']
total_E = Ec_inp + Fc_inp - gl * runner.mon['E.V']*(runner.mon['E.V']<0)
total_I = Ic_inp - gl * runner.mon['E.V']*(runner.mon['E.V']>0)

fig, gs = bp.visualize.get_figure(2, 1, 1.5, 7)

neuron_index = 375
fig.add_subplot(gs[:2, 0])
# bp.visualize.line_plot(runner.mon.ts, total_inp[:,neuron_index], xlim=(0, dur), legend='Total', alpha=0.5)  
# bp.visualize.line_plot(runner.mon.ts, Fc_inp[:,neuron_index], xlim=(0, dur), legend='Fc', alpha=0.5)  
# bp.visualize.line_plot(runner.mon.ts, Ec_inp[:,neuron_index], xlim=(0, dur), legend='Exc Rec', alpha=0.5) 
# bp.visualize.line_plot(runner.mon.ts, Ic_inp[:,neuron_index], xlim=(0, dur), legend='Inh Rec', alpha=0.5)  
# bp.visualize.line_plot(runner.mon.ts, -gl * runner.mon['E.V'][:,neuron_index], xlim=(0, dur), legend='Leak', alpha=0.5)
bp.visualize.line_plot(runner.mon.ts, total_E[:,neuron_index], xlim=(0, dur), legend='E', alpha=0.5)
bp.visualize.line_plot(runner.mon.ts, total_I[:,neuron_index], xlim=(0, dur), legend='I', alpha=0.5)
bp.visualize.line_plot(runner.mon.ts, total_inp[:,neuron_index], xlim=(0, dur), legend='Total', alpha=0.5)

plt.show()



# ===== heat map =====

# def moving_average(a, n, axis):
#     ret = bm.cumsum(a, axis=axis, dtype=bm.float32)
#     ret[n:] = ret[n:] - ret[:-n]
#     return ret[n - 1:] / n

# plt.figure()
# T = 100
# ma = moving_average(runner.mon['E.spike'], n=T, axis=0) # average window: 1 ms
# bump_activity = bm.vstack([bm.sum(ma * bm.cos(net.x[None,]), axis=1),bm.sum(ma * bm.sin(net.x[None,]), axis=1)])
# readout = bm.array([[1., 0.]]) @ bump_activity
# firing_rate = ma / (T * global_dt / 1000) 
# plt.subplot(2,1,1)
# nm_readout = readout.T / bm.max(readout)
# plt.plot(nm_readout)
# plt.plot(E_inp[T-1:,375] / bm.max(E_inp[T-1:,375]))
# # plt.xlim([int(runner.mon.ts.shape[0]*0.6), runner.mon.ts.shape[0]])
# plt.xlim([0, runner.mon.ts.shape[0]])

# print(bm.argmax(nm_readout>0.5))

# plt.subplot(2,1,2)
# plt.imshow(firing_rate.T, aspect='auto', cmap='gray_r')
# plt.plot(bm.argmax(E_inp, axis=1)[T-1:], label='input peak', color='red')
# # plt.xlim([int(runner.mon.ts.shape[0]*0.6), runner.mon.ts.shape[0]])
# plt.xlim([0, runner.mon.ts.shape[0]])
# plt.show()


# ====== membrane potential ======
# fig, gs = bp.visualize.get_figure(2, 2, 1.5, 7)
# fig.add_subplot(gs[:1, 0])
# neuron_index = 375
# bp.visualize.line_plot(runner.mon.ts, runner.mon['E.V'][:,neuron_index], xlim=(0, dur), legend='E mem potential')

# fig.add_subplot(gs[1:, 0])
# neuron_index = 740
# bp.visualize.line_plot(runner.mon.ts, runner.mon['E.V'][:,neuron_index], xlim=(0, dur), legend='E mem potential', show=True)


# ====== Membrane potential distribution ======
# plt.hist(bm.reshape(runner.mon['E.V'][1500:2000,], [-1]), bins=100, density=True, range=[-3, vth])
# plt.show()
# plt.hist(bm.reshape(runner.mon['E.V'][2500:3000,], [-1]), bins=100, density=True, range=[-3, vth])
# plt.show()



