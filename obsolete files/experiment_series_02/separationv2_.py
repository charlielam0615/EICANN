import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
import numpy as np
from UnitExpCUBA import UnitExpCUBA
from brainpy.dyn import TwoEndConn

import pdb

global_dt = 0.01
bp.math.set_platform('cpu')

# ==== Neuron parameters =====
n_scale = 1
size_E, size_Ip, size_Id, size_ff = 750*n_scale, 250*n_scale, 250*n_scale, 1000*n_scale
num = size_E + size_Id
num_ff = num
prob = 0.5
tau_scale = 10
tau_E = 2 * tau_scale
tau_I = 1 * tau_scale
V_reset = 0.
V_threshold = 1.

# ==== Balanced Parameters =====
tau_Ef = 3 * tau_scale
tau_If = 1 * tau_scale
ei_scale = 20 #20
f_E = 0.1
f_I = 0.05
jie = -2 * ei_scale * 1
jii = -1. * ei_scale * 1
jee = 0.25 * ei_scale * 3
jei = 0.4 * ei_scale * 3
JIE = jie / bm.sqrt(num)
JII = jii / bm.sqrt(num)
JEE = jee / bm.sqrt(num)
JEI = jei / bm.sqrt(num)
gl = 0.000268 * 1 * bm.sqrt(num)

# ===== CANN Parameters =====
cann_scale = 0.01
# tau_Es = 15 * tau_scale
# tau_Is = 5 * tau_scale
tau_Es = 15 * tau_scale
tau_Is = 5 * tau_scale
gEE = 0.065 * 3 * 130 * cann_scale
gEIp = 0.0175 * 7.5 * 4000 * cann_scale
gIpE = -0.1603 * 2.5 * 8000 * cann_scale
gIpIp = -0.0082 * 7.5 * 0 * cann_scale
shunting_k = 0.1
bg_str = 0.1


class Shunting(TwoEndConn):
    def __init__(self, E2Esyn_s, E2Esyn_f, I2Esyn_s, k, EGroup):
        super().__init__(pre=E2Esyn_s.pre, post=I2Esyn_s.post, conn=None)
        self.E2Esyn_s = E2Esyn_s
        self.E2Esyn_f = E2Esyn_f
        self.I2Esyn_s = I2Esyn_s
        self.EGroup = EGroup
        self.k = k
        self.post = self.E2Esyn_s.post

    def update(self, tdi):
        t, dt = tdi.t, tdi.dt
        E_inp = self.E2Esyn_s.output_value + self.E2Esyn_f.output_value + self.EGroup.ext_input
        I_inp = self.I2Esyn_s.output_value
        self.post.input += self.k * E_inp * I_inp


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
        self.V = bm.Variable(bm.random.rand(self.size) * (self.vth-vreset) + vreset)
        self.input = bm.Variable(bm.zeros(self.size))
        self.t_last_spike = bm.Variable(bm.ones(self.size) * -1e7)
        self.refractory = bm.Variable(bm.zeros(self.size, dtype=bool))
        self.spike = bm.Variable(bm.zeros(self.size, dtype=bool))
        self.ext_input = bm.Variable(bm.zeros(self.size))

        # integral
        self.integral = bp.odeint(f=self.derivative, method='euler')

    def derivative(self, V, t, inputs, ext_input):
        I = -self.gl*V + inputs + ext_input
        dvdt = I / self.tau
        return dvdt

    def update(self, tdi):
        _t, _dt = tdi.t, tdi.dt
        refractory = (_t - self.t_last_spike) <= self.tau_ref
        V = self.integral(self.V, _t, self.input, self.ext_input, dt=_dt)
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
        self.J = 1.
        self.A = 1.
        self.x = bm.linspace(-bm.pi, bm.pi, size_E)

        # neurons
        self.E = LIF(size=size_E, gl=gl, tau=tau_E, vth=V_threshold, vreset=V_reset, tau_ref=0)
        self.Ip = LIF(size=size_Ip, gl=gl, tau=tau_I, vth=V_threshold, vreset=V_reset, tau_ref=0)
        self.Id = LIF(size=size_Id, gl=gl, tau=tau_I, vth=V_threshold, vreset=V_reset, tau_ref=0)

        w_ee, w_ei, w_ie, w_ii = self.make_conn(self.x)

        # synapses
        self.E2E_gaus = gEE * w_ee

        # ======== EI balance =====
        self.E2E_f = UnitExpCUBA(pre=self.E, post=self.E, conn=bp.conn.FixedProb(prob), tau=tau_Ef, g_max=JEE)
        self.E2I_f = UnitExpCUBA(pre=self.E, post=self.Id, conn=bp.conn.FixedProb(prob), tau=tau_Ef, g_max=JEI)
        self.I2I_f = UnitExpCUBA(pre=self.Id, post=self.Id, conn=bp.conn.FixedProb(prob), tau=tau_If, g_max=JII)
        self.I2E_f = UnitExpCUBA(pre=self.Id, post=self.E, conn=bp.conn.FixedProb(prob), tau=tau_If, g_max=JIE)
        # ======= CANN =====
        self.E2E_s = UnitExpCUBA(pre=self.E, post=self.E, conn=bp.connect.All2All(), tau=tau_Es, g_max=self.E2E_gaus)
        self.E2I_s = UnitExpCUBA(pre=self.E, post=self.Ip, conn=bp.connect.All2All(), tau=tau_Es, g_max=gEIp*w_ei)
        self.I2I_s = UnitExpCUBA(pre=self.Ip, post=self.Ip, conn=bp.connect.All2All(), tau=tau_Is, g_max=gIpIp*w_ii)
        self.I2E_s = UnitExpCUBA(pre=self.Ip, post=self.E, conn=bp.connect.All2All(), tau=tau_Is, g_max=gIpE*w_ie)
        self.ESI = Shunting(E2Esyn_s=self.E2E_s, E2Esyn_f=self.E2E_f, I2Esyn_s=self.I2E_s, k=shunting_k, EGroup=self.E)

        super(EICANN, self).__init__()

        print('[Weights]')
        print("------------- EI Balanced ---------------")
        print("|  E2E   |  E2I    |  I2I     |  I2E    | ")
        print(f"|{JEE.max():.5f} | {JEI.max():.5f} | {JII.min():.5f} | {JIE.min():.5f}|")
        print("---------------- CANN -------------------")
        print("|  E2E   |  E2I    |  I2I     |  I2E    | ")
        print(f"|{self.E2E_gaus.max():.5f} | {(gEIp*w_ei).max():.5f} | {(gIpIp*w_ii).min():.5f}  | {(gIpE*w_ie).min():.5f}|")
        super(EICANN, self).__init__()

    def dist(self, d):
        d = bm.remainder(d, 2*bm.pi)
        d = bm.where(d > bm.pi, d - 2*bm.pi, d)
        return d

    def make_conn(self, x):
        x_left = bm.reshape(x, (-1,1))
        x_right = bm.reshape(x, (1,-1))
        d = self.dist(x_left-x_right)
        w_ee_ = self.J * bm.exp(-bm.pi * bm.square(d / self.a))

        # w_ee = w_ee_ / bm.sum(w_ee_, axis=-1, keepdims=True)
        w_ee = w_ee_

        const_conn = lambda w, p: (w > 1-p) / np.maximum(np.sum(w > 1-p, axis=-1, keepdims=True), 1)
        w_ei = const_conn(np.random.rand(size_E, size_Ip), prob)
        w_ii = const_conn(np.random.rand(size_Ip, size_Ip), prob)
        w_ie = const_conn(np.random.rand(size_Ip, size_E), prob)
 
        return w_ee, w_ei, w_ie, w_ii

    def get_stimulus_by_pos(self, pos, size_n):
        x = bm.linspace(-bm.pi, bm.pi, size_n)
        if bm.ndim(pos) == 2:
            x = x[None, ]
        I = self.A * bm.exp(-bm.pi * bm.square(self.dist(x - pos) / (2*self.a)))
        return I


net = EICANN()
Einp_scale = num_ff * f_E / bm.sqrt(num_ff)
Iinp_scale = num_ff * f_I / bm.sqrt(num_ff)

# ===== Persistent Activity ====
bg_amp = bg_str * net.A
st_amp = bp.inputs.section_input(values=[[bg_amp]], durations=[10000.], dt=global_dt)
ramp_amp = bp.inputs.ramp_input(c_start=0, c_end=1-bg_str, duration=1., dt=global_dt)
hold_amp = bp.inputs.section_input(values=[[1-bg_str]], durations=[999.], dt=global_dt)
remove_amp = bp.inputs.section_input(values=[[bg_amp]], durations=[1500.], dt=global_dt)

E_inputs = bm.concatenate([
    st_amp*bm.ones([1, size_E]),
    ramp_amp[:,None]*net.get_stimulus_by_pos(0., size_E)[None,]+bg_amp*bm.ones([1, size_E]),
    hold_amp*net.get_stimulus_by_pos(0., size_E)[None,]+bg_amp*bm.ones([1, size_E]),
    remove_amp*bm.ones([1, size_E]),
])


# E_inputs = bm.concatenate([
#     st_amp*bm.ones([1, size_E]),
#     ramp_amp[:,None]*net.get_stimulus_by_pos(-bm.pi/2, size_E)[None,] + 0.7*ramp_amp[:,None]*net.get_stimulus_by_pos(bm.pi/2, size_E)[None,] + bg_amp*bm.ones([1, size_E]),
#     hold_amp*net.get_stimulus_by_pos(-bm.pi/2, size_E)[None,] + 0.7*hold_amp*net.get_stimulus_by_pos(bm.pi/2, size_E)[None,] + bg_amp*bm.ones([1, size_E]),
#     remove_amp*bm.ones([1, size_E]),
# ])


I_inputs = bm.concatenate([
    st_amp*bm.ones([1, size_Id]),
    ramp_amp[:,None]*net.get_stimulus_by_pos(0., size_Id)[None,]+bg_amp*bm.ones([1, size_Id]),
    hold_amp*net.get_stimulus_by_pos(0., size_Id)[None,]+bg_amp*bm.ones([1, size_Id]),
    remove_amp*bm.ones([1, size_Id]),
])

dur = 3500+9000

# =========== Moving Inputs ========
# dur = 3000
# n_step = int(dur / global_dt)
# pos = bm.linspace(-2*bm.pi/2, 50*bm.pi/2, n_step)[:,None]
# E_inputs = net.get_stimulus_by_pos(pos, size_E)
# I_inputs = bm.zeros([n_step, size_Id])


# ===================================

# E_noise = bm.random.randn(*E_inputs.shape) * bm.sqrt(E_inputs)
# I_noise = bm.random.randn(*mean_inputs.shape) * bm.sqrt(inputs)
E_noise = 0
I_noise = 0

E_inp = Einp_scale*E_inputs + bm.sqrt(Einp_scale)*E_noise
I_inp = Iinp_scale*I_inputs + bm.sqrt(Iinp_scale)*I_noise

runner = bp.dyn.DSRunner(net,
                         jit=True,
                         monitors=['Id.V', 'Ip.V', 'E.V', 'E.spike', 'Ip.spike', 'Id.spike',
                                   'E2E_f.g', 'E2E_s.g', 'E2I_f.g', 'E2I_s.g',
                                   'I2I_f.g', 'I2I_s.g', 'I2E_f.g', 'I2E_s.g'
                                   ],
                         inputs=[('E.ext_input',  E_inp, 'iter', '='),
                                 ('Id.ext_input', I_inp*0, 'iter', '='),
                                 ('Ip.ext_input', I_inp*0, 'iter', '=')],
                         dt=global_dt)
t = runner(dur)


# # ===== Raster Plot =====
fig, gs = bp.visualize.get_figure(3, 1, 1.5, 10)

fig.add_subplot(gs[:1, 0])
bp.visualize.raster_plot(runner.mon.ts, runner.mon['E.spike'], xlim=(0, dur), ylim=[0, size_E], markersize=1.)
# plt.plot([1000, 2000], [int(size_E/2), int(size_E/2)], label='input peak', color='red')
plt.plot([10000, 11000], [int(size_E/2), int(size_E/2)], label='input peak', color='red')
# plt.plot(bm.arange(dur), bm.argmax(E_inputs[::100], axis=1), label='input peak', color='red')
plt.xlim([0, runner.mon.ts[-1]])

fig.add_subplot(gs[1:2, 0])
bp.visualize.raster_plot(runner.mon.ts, runner.mon['Ip.spike'], xlim=(0, dur), ylim=[0, size_Ip])
plt.xlim([0, runner.mon.ts[-1]])

fig.add_subplot(gs[2:3, 0])
bp.visualize.raster_plot(runner.mon.ts, runner.mon['Id.spike'], xlim=(0, dur), ylim=[0, size_Id])
plt.xlim([0, runner.mon.ts[-1]])
plt.show()

# ===== Total Current Visualization =====
# Ec_inp = runner.mon['E2E_f.g']+runner.mon['E2E_s.g']
# Fc_inp = E_inp
# shunting_inp = shunting_k*(runner.mon['E2E_s.g']+runner.mon['E2E_f.g']+Fc_inp)*runner.mon['I2E_s.g']
# Ic_inp = runner.mon['I2E_f.g'] + runner.mon['I2E_s.g'] + shunting_inp
# total_inp = Ec_inp + Ic_inp + Fc_inp - gl * runner.mon['E.V']
#
# fig, gs = bp.visualize.get_figure(2, 1, 1.5, 7)
#
# neuron_index = int(size_E/2)
# # neuron_index = 530
# fig.add_subplot(gs[:2, 0])
# bp.visualize.line_plot(runner.mon.ts, (Ec_inp+Fc_inp)[:,neuron_index], xlim=(0, dur), legend='E', alpha=0.5)
# bp.visualize.line_plot(runner.mon.ts, (Ic_inp-gl * runner.mon['E.V'])[:,neuron_index], xlim=(0, dur), legend='I', alpha=0.5)
# bp.visualize.line_plot(runner.mon.ts, total_inp[:,neuron_index], xlim=(0, dur), legend='Total', alpha=0.5)
# plt.show()
#
# print(f'[Current] neuron {neuron_index}')
# print("------------- EI Balanced ---------------")
# print("|  E2E   |  I2E    |          |         | ")
# print(f"|{runner.mon['E2E_f.g'][250000:290000, neuron_index].max():.5f} |{runner.mon['I2E_f.g'][250000:290000, neuron_index].min():.5f} |         |         |")
# print("---------------- CANN -------------------------------")
# print("|  E2E   |  I2E    |  F2E     |  SI      |   E2I    |")
# print(f"|{runner.mon['E2E_s.g'][250000:290000, neuron_index].max():.5f} | {runner.mon['I2E_s.g'][250000:290000, neuron_index].min():.5f} |  {Fc_inp[250000:290000, neuron_index].max():.5f} | {shunting_inp[250000:290000, neuron_index].min():.5f} | {runner.mon['E2I_s.g'].max():.5f}")


# ===== Balanced Current Visualization =====
# Ec_inp = runner.mon['E2E_f.g']+runner.mon['E2E_s.g']
# Fc_inp = E_inp
# shunting_inp = shunting_k*(runner.mon['E2E_s.g']+runner.mon['E2E_f.g']+Fc_inp)*runner.mon['I2E_s.g']
# Ic_inp = runner.mon['I2E_f.g'] + runner.mon['I2E_s.g'] + shunting_inp
#
# total_inp = Ec_inp + Ic_inp + Fc_inp - gl * runner.mon['E.V']
# ext_slow = Fc_inp + runner.mon['E2E_s.g'] + runner.mon['I2E_s.g'] + shunting_inp
# fig, gs = bp.visualize.get_figure(2, 1, 1.5, 7)
#
# neuron_index = int(size_E/2)
#
# fig.add_subplot(gs[:2, 0])
# bp.visualize.line_plot(runner.mon.ts, ext_slow[:,neuron_index], xlim=(0, dur), legend='slow+ext', alpha=0.5)
# bp.visualize.line_plot(runner.mon.ts, total_inp[:,neuron_index], xlim=(0, dur), legend='Total', alpha=0.5)
# plt.show()




# # ===== heat map =====
# plt.figure()
# T = 100
# neuron_index = int(size_E/2)
# x = bm.linspace(-bm.pi, bm.pi, size_E)
# ma = moving_average(runner.mon['E.spike'], n=T, axis=0)  # average window: 1 ms
# bump_activity = bm.vstack([bm.sum(ma * bm.cos(x[None,]), axis=1), bm.sum(ma * bm.sin(x[None,]), axis=1)])
# readout = bm.array([[1., 0.]]) @ bump_activity
# firing_rate = ma / (T * global_dt / 1000)
# plt.subplot(1, 1, 1)
# nm_readout = readout.T / bm.mean(readout[:, 100000:355000])
# plt.plot(nm_readout)
# plt.plot(E_inp[T-1:, neuron_index] / bm.max(E_inp[T-1:, neuron_index]))
# plt.xlim([0, runner.mon.ts.shape[0]])
# # plt.subplot(2, 1, 2)
# # plt.imshow(firing_rate.T, aspect='auto', cmap='gray_r')
# # plt.plot(bm.argmax(E_inp, axis=1)[T-1:], label='input peak', color='red')
# # plt.xlim([0, runner.mon.ts.shape[0]])
# plt.show()