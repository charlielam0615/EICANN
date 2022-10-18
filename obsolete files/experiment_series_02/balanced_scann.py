import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
import numpy as np
from UnitExpCUBA import UnitExpCUBA
from brainpy.dyn import TwoEndConn

import pdb

global_dt = 0.01
bp.math.set_platform('cpu')

size_E, size_Ip, size_Id, size_ff = 750, 250, 250, 1000
num = size_E + size_Id
vreset = 0
tau_scale = 1.5
tau_Es = 1.5 * tau_scale
tau_Is = 0.3 * tau_scale 
tau_E = 0.6 * tau_scale
tau_I = 0.3 * tau_scale

cann_scale = 0.1 # 0.1
gEE_G = 0.065 * 3 * 8 * cann_scale
gEIp = 0.0175 * 7.5 * 6 * cann_scale
gIpE = -0.1603 * 2.5 * 4 * cann_scale
gIpIp = -0.0082 * 7.5 * 30 * cann_scale
shunting_k = 0.1

vth = 10
f_E = 1.1 * 0.5
f_I = 1.0 * 0.5
bg_str = 0.25
ei_scale = 1
jie = -1 * ei_scale
jii = -1 * ei_scale
jee = 0.25 * ei_scale * 3
jei = 0.4 * ei_scale * 3
gIdE = jie / bm.sqrt(num)
gIdId = jii / bm.sqrt(num)
gEE_R = jee / bm.sqrt(num)
gEId = jei / bm.sqrt(num)
gl = 0.000268 * ei_scale * bm.sqrt(num)

prob = 0.25


class Shunting(TwoEndConn):
    def __init__(self, E2Esyn, I2Esyn, k, EGroup):
        super().__init__(pre=E2Esyn.pre, post=I2Esyn.post, conn=None)
        self.E2Esyn = E2Esyn
        self.I2Esyn = I2Esyn
        self.EGroup = EGroup
        self.k = k

        assert self.E2Esyn.post == self.I2Esyn.post
        self.post = self.E2Esyn.post

    def update(self, tdi):
        t, dt = tdi.t, tdi.dt
        E_inp = self.E2Esyn.output_value + self.EGroup.ext_input
        I_inp = self.I2Esyn.output_value
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
        self.E = LIF(size=size_E, gl=gl, tau=tau_E, vth=vth, vreset=vreset, tau_ref=0)
        self.Ip = LIF(size=size_Ip, gl=gl, tau=tau_I, vth=vth, vreset=vreset, tau_ref=0)
        self.Id = LIF(size=size_Id, gl=gl, tau=tau_I, vth=vth, vreset=vreset, tau_ref=0)

        w_ee, w_ei, w_ie, w_ii = self.make_conn(self.x)

        # synapses
        self.E2E_rand = (bm.random.rand(size_E, size_E) < prob).astype(bm.float32) * gEE_R
        self.E2E_gaus = gEE_G*w_ee
        E2E = self.E2E_rand + self.E2E_gaus

        self.E2E = UnitExpCUBA(pre=self.E, post=self.E, conn=bp.connect.All2All(), tau=tau_Es, g_max=E2E)
        self.E2Ip = UnitExpCUBA(pre=self.E, post=self.Ip, conn=bp.connect.All2All(), tau=tau_Es, g_max=gEIp*w_ei)
        self.Ip2Ip = UnitExpCUBA(pre=self.Ip, post=self.Ip, conn=bp.connect.All2All(), tau=tau_Is, g_max=gIpIp*w_ii)
        self.Ip2E = UnitExpCUBA(pre=self.Ip, post=self.E, conn=bp.connect.All2All(), tau=tau_Is, g_max=gIpE*w_ie)
        self.ESI = Shunting(E2Esyn=self.E2E, I2Esyn=self.Ip2E, k=shunting_k, EGroup=self.E)

        self.E2Id = UnitExpCUBA(pre=self.E, post=self.Id, conn=bp.conn.FixedProb(prob), tau=tau_Es, g_max=gEId)
        self.Id2Id = UnitExpCUBA(pre=self.Id, post=self.Id, conn=bp.conn.FixedProb(prob), tau=tau_Is, g_max=gIdId)
        self.Id2E = UnitExpCUBA(pre=self.Id, post=self.E, conn=bp.conn.FixedProb(prob), tau=tau_Is, g_max=gIdE)
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

        w_ee = w_ee_ / bm.sum(w_ee_, axis=-1, keepdims=True)
        const_conn = lambda w, p: (w > 1-p) / np.maximum(np.sum(w > 1-p, axis=-1, keepdims=True), 1)
        w_ei = const_conn(np.random.rand(size_E, size_Ip), prob)
        w_ii = const_conn(np.random.rand(size_Ip, size_Ip), prob)
        w_ie = const_conn(np.random.rand(size_Ip, size_E), prob)
 
        return w_ee, w_ei, w_ie, w_ii

    def get_stimulus_by_pos(self, pos):
        if bm.ndim(pos) <= 1:
            x = self.x
        elif bm.ndim(pos) == 2:
            x = self.x[None,]
        I = self.A * bm.exp(-bm.pi*bm.square(self.dist(x - pos) / self.a))
        return I


net = EICANN()

Einp_scale = size_ff * f_E / bm.sqrt(num)
Iinp_scale = size_ff * f_I / bm.sqrt(num) * 0.5

# ===== Persistent Activity ====
name = 'cann-persistent.gif'
bg_amp = bg_str * net.A

st_amp = bp.inputs.section_input(values=[[bg_amp]], durations=[20.], dt=global_dt)
ramp_amp = bp.inputs.ramp_input(c_start=0, c_end=1., duration=5., dt=global_dt)
hold_amp = bp.inputs.section_input(values=[[1.]], durations=[45.], dt=global_dt)
inputs = bm.concatenate([
    st_amp*bm.ones([1, size_E]),
    ramp_amp[:,None]*net.get_stimulus_by_pos(0.)[None,]+bg_amp*bm.ones([1, size_E]),
    hold_amp*net.get_stimulus_by_pos(0.)[None,]+bg_amp*bm.ones([1, size_E])
])

dur = 70.

mean_inputs = bm.mean(inputs, axis=-1, keepdims=True)
# E_noise = bm.random.randn(*inputs.shape) * bm.sqrt(inputs)
# I_noise = bm.random.randn(*mean_inputs.shape) * bm.sqrt(mean_inputs)
E_noise = 0
I_noise = 0
E_inp = Einp_scale*inputs + E_noise
I_inp = Iinp_scale*mean_inputs + I_noise


runner = bp.dyn.DSRunner(net,
                         jit=True,
                         monitors=['E2E.g', 'Ip2E.g', 'Id2E.g', 'Ip.V', 'E.V', 'E.spike', 'Ip.spike', 'Id.spike'],
                         inputs=[('E.ext_input',  E_inp, 'iter', '='),
                                 ('Id.ext_input', I_inp, 'iter', '='),
                                 ('Ip.ext_input', I_inp, 'iter', '=')],
                         dt=global_dt)
t = runner(dur)

# ====== Plot weight matrix ======
# mean_rand = bm.mean(net.E2E_rand[375, ])
# gaus = net.E2E_gaus[375, ]
# weights = mean_rand+gaus
# plt.figure()
# plt.plot(weights)
# plt.ylim([0, bm.max(weights)*1.2])
# plt.show()

# ===== Raster Plot =====
fig, gs = bp.visualize.get_figure(3, 1, 1.5, 10)

fig.add_subplot(gs[:1, 0])
bp.visualize.raster_plot(runner.mon.ts, runner.mon['E.spike'], xlim=(0, dur), ylim=[0,size_E])
# bp.visualize.line_plot(runner.mon.ts, bm.argmax(inputs, axis=1), xlim=(0, dur), legend='input peak')
plt.plot([20, 70], [375, 375], label='input peak', color='red')
plt.xlim([0, runner.mon.ts[-1]])

fig.add_subplot(gs[1:2, 0])
bp.visualize.raster_plot(runner.mon.ts, runner.mon['Ip.spike'], xlim=(0, dur), ylim=[0,size_Ip])
plt.xlim([0, runner.mon.ts[-1]])

fig.add_subplot(gs[2:3, 0])
bp.visualize.raster_plot(runner.mon.ts, runner.mon['Id.spike'], xlim=(0, dur), ylim=[0,size_Id])
plt.xlim([0, runner.mon.ts[-1]])
plt.show()

# ===== Current Visualization =====
# Ec_inp = runner.mon['E2E.g']
# Fc_inp = E_inp
# shunting_inp = shunting_k*(runner.mon['E2E.g']+Fc_inp)*runner.mon['Ip2E.g']
# r_SI = shunting_k*runner.mon['E2E.g']*runner.mon['Ip2E.g']
# Ic_inp = runner.mon['Id2E.g'] + runner.mon['Ip2E.g'] + shunting_inp
# total_inp = Ec_inp + Ic_inp + Fc_inp - gl * runner.mon['E.V']

# fig, gs = bp.visualize.get_figure(4, 1, 1.5, 7)

# neuron_index = 375
# fig.add_subplot(gs[:2, 0])
# # bp.visualize.line_plot(runner.mon.ts, total_inp[:,neuron_index], xlim=(0, dur), legend='Total')  
# # bp.visualize.line_plot(runner.mon.ts, Fc_inp[:,neuron_index], xlim=(0, dur), legend='Fc')  
# # bp.visualize.line_plot(runner.mon.ts, Ec_inp[:,neuron_index], xlim=(0, dur), legend='Exc Rec') 
# # bp.visualize.line_plot(runner.mon.ts, (Ic_inp-shunting_inp)[:,neuron_index], xlim=(0, dur), legend='Inh (ex. SI)')  
# # bp.visualize.line_plot(runner.mon.ts, shunting_inp[:,neuron_index], xlim=(0, dur), legend='SI')
# # bp.visualize.line_plot(runner.mon.ts, -gl * runner.mon['E.V'][:,neuron_index], xlim=(0, dur), legend='Leak')
# bp.visualize.line_plot(runner.mon.ts, (Ec_inp+Fc_inp)[:,neuron_index], xlim=(0, dur), legend='E', alpha=0.5)
# bp.visualize.line_plot(runner.mon.ts, (Ic_inp-gl * runner.mon['E.V'])[:,neuron_index], xlim=(0, dur), legend='I', alpha=0.5)
# bp.visualize.line_plot(runner.mon.ts, total_inp[:,neuron_index], xlim=(0, dur), legend='Total', alpha=0.5)


# fig.add_subplot(gs[2:, 0])
# bp.visualize.line_plot(runner.mon.ts, (Ic_inp-shunting_inp)[:,neuron_index], xlim=(0, dur), legend='Id+Ip', alpha=0.5)  
# bp.visualize.line_plot(runner.mon.ts, r_SI[:,neuron_index], xlim=(0, dur), legend='SI (ex. Fc)', alpha=0.5, show=True) 


# ==== membrane potential and ff input ====
# Fc_inp = E_inp
# fig, gs = bp.visualize.get_figure(2, 1, 1.5, 7)
# neuron_index = 375
# fig.add_subplot(gs[:1, 0])
# bp.visualize.line_plot(runner.mon.ts, Fc_inp[:,neuron_index], xlim=(0, dur), legend='ff inp')
# fig.add_subplot(gs[1:2, 0])
# bp.visualize.line_plot(runner.mon.ts, runner.mon['E.V'][:,neuron_index], xlim=(0, dur), legend='mem potential', show=True)

# fig, gs = bp.visualize.get_figure(2, 1, 1.5, 7)
# neuron_index = 740
# fig.add_subplot(gs[:1, 0])
# bp.visualize.line_plot(runner.mon.ts, Fc_inp[:,neuron_index], xlim=(0, dur), legend='ff inp')
# fig.add_subplot(gs[1:2, 0])
# bp.visualize.line_plot(runner.mon.ts, runner.mon['E.V'][:,neuron_index], xlim=(0, dur), legend='mem potential', show=True)

# ==== membrane potential hist =====
# plt.hist(bm.reshape(runner.mon['E.V'][1500:2000,], [-1]), bins=100, density=True, range=[-3, vth])
# plt.show()
# plt.hist(bm.reshape(runner.mon['E.V'][2500:3000,], [-1]), bins=100, density=True, range=[-3, vth])
# plt.show()



# ===== heat map =====

def moving_average(a, n, axis):
    ret = bm.cumsum(a, axis=axis, dtype=bm.float32)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


plt.figure()
T = 100
ma = moving_average(runner.mon['E.spike'], n=T, axis=0) # average window: 1ms
bump_activity = bm.vstack([bm.sum(ma * bm.cos(net.x[None,]), axis=1),bm.sum(ma * bm.sin(net.x[None,]), axis=1)])
readout = bm.array([[1., 0.]]) @ bump_activity
firing_rate = ma / (T * global_dt / 1000)
plt.subplot(2,1,1)
nm_readout = readout.T / bm.max(readout)
plt.plot(nm_readout)
plt.plot(E_inp[T-1:,375] / bm.max(E_inp[T-1:,375]))
# plt.xlim([int(runner.mon.ts.shape[0]*0.6), runner.mon.ts.shape[0]])
plt.xlim([0, runner.mon.ts.shape[0]])

print(bm.argmax(nm_readout>0.4))

plt.subplot(2,1,2)
plt.imshow(firing_rate.T, aspect='auto', cmap='gray_r')
plt.plot(bm.argmax(E_inp, axis=1)[T-1:], label='input peak', color='red')
# plt.xlim([int(runner.mon.ts.shape[0]*0.6), runner.mon.ts.shape[0]])
plt.xlim([0, runner.mon.ts.shape[0]])
plt.show()



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


