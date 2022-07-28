import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d
from brainpy.dyn import NeuGroup, TwoEndConn, DynamicalSystem

from typing import Union, Dict, Callable
from brainpy.connect import TwoEndConnector, All2All, One2One
from brainpy.types import Tensor
from brainpy.initialize import Initializer, init_param
from brainpy.integrators import odeint, JointEq

import pdb

global_dt = 0.01

# ======= Copied and modified from brainpy source ======

class ExpCUBA(TwoEndConn):
    def __init__(
            self,
            pre: NeuGroup,
            post: NeuGroup,
            conn: Union[TwoEndConnector, Tensor, Dict[str, Tensor]],
            conn_type: str = 'sparse',
            g_max: Union[float, Tensor, Initializer, Callable] = 1.,
            delay_step: Union[int, Tensor, Initializer, Callable] = None,
            tau: Union[float, Tensor] = 8.0,
            name: str = None,
            method: str = 'exp_auto',
    ):
        super(ExpCUBA, self).__init__(pre=pre, post=post, conn=conn, name=name)
        self.check_pre_attrs('spike')
        self.check_post_attrs('input', 'V')

        # parameters
        self.tau = tau
        if bm.size(self.tau) != 1:
            raise ValueError(f'"tau" must be a scalar or a tensor with size of 1. '
                                             f'But we got {self.tau}')

        # connections and weights
        self.conn_type = conn_type
        if conn_type not in ['sparse', 'dense']:
            raise ValueError(f'"conn_type" must be in "sparse" and "dense", but we got {conn_type}')
        if self.conn is None:
            raise ValueError(f'Must provide "conn" when initialize the model {self.name}')
        if isinstance(self.conn, One2One):
            self.g_max = init_param(g_max, (self.pre.num,), allow_none=False)
            self.weight_type = 'heter' if bm.size(self.g_max) != 1 else 'homo'
        elif isinstance(self.conn, All2All):
            self.g_max = init_param(g_max, (self.pre.num, self.post.num), allow_none=False)
            if bm.size(self.g_max) != 1:
                self.weight_type = 'heter'
                bm.fill_diagonal(self.g_max, 0.)
            else:
                self.weight_type = 'homo'
        else:
            if conn_type == 'sparse':
                self.pre2post = self.conn.require('pre2post')
                self.g_max = init_param(g_max, self.pre2post[1].shape, allow_none=False)
                self.weight_type = 'heter' if bm.size(self.g_max) != 1 else 'homo'
            elif conn_type == 'dense':
                self.g_max = init_param(g_max, (self.pre.num, self.post.num), allow_none=False)
                self.weight_type = 'heter' if bm.size(self.g_max) != 1 else 'homo'
                if self.weight_type == 'homo':
                    self.conn_mat = self.conn.require('conn_mat')
            else:
                raise ValueError(f'Unknown connection type: {conn_type}')

        # variables
        self.g = bm.Variable(bm.zeros(self.post.num))
        self.output_value = bm.Variable(bm.zeros(self.post.num))

        self.delay_step = self.register_delay(f"{self.pre.name}.spike", delay_step, self.pre.spike)

        # function
        self.integral = odeint(lambda g, t: -g / self.tau, method=method)

    def reset(self):
        self.g.value = bm.zeros(self.post.num)
        if self.delay_step is not None:
            self.reset_delay(f"{self.pre.name}.spike", self.pre.spike)

    def update(self, t, dt):
        # delays
        if self.delay_step is None:
            pre_spike = self.pre.spike
        else:
            pre_spike = self.get_delay_data(f"{self.pre.name}.spike", self.delay_step)
            self.update_delay(f"{self.pre.name}.spike", self.pre.spike)

        # post values
        assert self.weight_type in ['homo', 'heter']
        assert self.conn_type in ['sparse', 'dense']
        if isinstance(self.conn, All2All):
            pre_spike = pre_spike.astype(bm.float_)
            if self.weight_type == 'homo':
                post_vs = bm.sum(pre_spike)
                if not self.conn.include_self:
                    post_vs = post_vs - pre_spike
                post_vs = self.g_max * post_vs
            else:
                post_vs = pre_spike @ self.g_max
        elif isinstance(self.conn, One2One):
            pre_spike = pre_spike.astype(bm.float_)
            post_vs = pre_spike * self.g_max
        else:
            if self.conn_type == 'sparse':
                post_vs = bm.pre2post_event_sum(pre_spike,
                                                self.pre2post,
                                                self.post.num,
                                                self.g_max)
            else:
                pre_spike = pre_spike.astype(bm.float_)
                if self.weight_type == 'homo':
                    post_vs = self.g_max * (pre_spike @ self.conn_mat)
                else:
                    post_vs = pre_spike @ self.g_max

        # updates
        self.g.value = self.integral(self.g.value, t, dt=dt) + post_vs
        self.output_value.value = self.output(self.g)
        self.post.input += self.output_value

    def output(self, g_post):
        return g_post


# =========================


bp.math.set_platform('cpu')

size_E, size_Ip, size_Id, size_ff = 1000, 250, 250, 250
vreset = 0
tau_Es = 6. 
tau_Is = 5. 
tau_E = 15.0
tau_I = 10.0

cann_scale = 1.0
gEE_G = 0.065 * 300 * 8 * cann_scale # 4
gEIp = 0.0175 * 750 * 11 * cann_scale # 11
gIpE = -0.1603 * 250 * 2 * cann_scale
gIpIp = -0.0082 * 750 * 30 * cann_scale

# gl = 0.012 * 100 * cann_scale
gl = 0.012 * 3 * cann_scale

shunting_k = 0.25 # 1.0
input_amp = 2.


vth = 9
f_E = 2 /bm.sqrt(size_ff) * 0.6 * 2 # *1
f_I = 1 /bm.sqrt(size_ff) * 0.6
bg_str = 0.125

ei_scale = 70
gIdE = -2.0 / bm.sqrt(size_E) * ei_scale
gIdId = -2.0 / bm.sqrt(size_E) * ei_scale
gEE_R = 0.25 / bm.sqrt(size_E) * ei_scale *2 # *1
gEId = 1.0 / bm.sqrt(size_E) * ei_scale # 0.9

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
        self.vth = vth
        self.gl = gl
        self.vreset = vreset
        self.tau_ref = tau_ref

        # variables
        self.V = bm.Variable(bm.random.rand(self.size) * (vth-vreset) + vreset)
        self.input = bm.Variable(bm.zeros(self.size))
        self.t_last_spike = bm.Variable(bm.ones(self.size) * -1e7)
        self.refractory = bm.Variable(bm.zeros(self.size, dtype=bool))
        self.spike = bm.Variable(bm.zeros(self.size, dtype=bool))
        self.ext_input = bm.Variable(bm.zeros(self.size))

        # integral
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


class EICANN(bp.dyn.Network):
    def __init__(self):
        self.a = bm.sqrt(2*bm.pi) * (bm.pi/6)
        self.J = 4.
        self.A = input_amp
        self.x = bm.linspace(-bm.pi, bm.pi, size_E)

        # neurons
        self.E = LIF(size=size_E, gl=gl, tau=tau_E, vth=vth, vreset=vreset, tau_ref=0)
        self.Ip = LIF(size=size_Ip, gl=gl, tau=tau_I, vth=vth, vreset=vreset, tau_ref=0)
        self.Id = LIF(size=size_Id, gl=gl, tau=tau_I, vth=vth, vreset=vreset, tau_ref=0)

        self.E.V[:] = bm.random.random(size_E) * (vth - vreset) + vreset
        self.Ip.V[:] = bm.random.random(size_Ip) * (vth - vreset) + vreset
        self.Id.V[:] = bm.random.random(size_Id) * (vth - vreset) + vreset

        w_ee, w_ei, w_ie, w_ii = self.make_conn(self.x)

        # synapses
        E2E_rand = (bm.random.rand(size_E, size_E) > prob).astype(bm.float32) * gEE_R
        E2E = E2E_rand + gEE_G*w_ee # mean: 0.41457522, 0.039
        
        self.E2E = ExpCUBA(pre=self.E, post=self.E, conn=bp.connect.All2All(), tau=tau_Es, g_max=E2E) # mean: 0.4917864
        self.E2Ip = ExpCUBA(pre=self.E, post=self.Ip, conn=bp.connect.All2All(), tau=tau_Es, g_max=gEIp*w_ei) # mean: 0.42
        self.Ip2Ip = ExpCUBA(pre=self.Ip, post=self.Ip, conn=bp.connect.All2All(), tau=tau_Is, g_max=gIpIp*w_ii) # mean: -0.738
        self.Ip2E = ExpCUBA(pre=self.Ip, post=self.E, conn=bp.connect.All2All(), tau=tau_Is, g_max=gIpE*w_ie) # mean: -0.08015
        self.ESI = Shunting(E2Esyn=self.E2E, I2Esyn=self.Ip2E, k=shunting_k, EGroup=self.E)

        self.E2Id = ExpCUBA(pre=self.E, post=self.Id, conn=bp.conn.FixedProb(prob), tau=tau_Es, g_max=gEId) # mean: 0.8854378
        self.Id2Id = ExpCUBA(pre=self.Id, post=self.Id, conn=bp.conn.FixedProb(prob), tau=tau_Is, g_max=gIdId) # mean: -4.427189
        self.Id2E = ExpCUBA(pre=self.Id, post=self.E, conn=bp.conn.FixedProb(prob), tau=tau_Is, g_max=gIdE) # mean: -4.427189

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
Einp_scale = size_ff * f_E
Iinp_scale = size_ff * f_I

# ===== Persistent Activity ====
inputs = net.get_stimulus_by_pos(0.)
bg_inputs = bg_str * net.A
E_inputs, dur = bp.inputs.section_input(values=[bg_inputs, inputs+bg_inputs, bg_inputs],
                                         durations=[300., 600., 500.],
                                         return_length=True,
                                         dt=global_dt)

I_inputs = bm.mean(E_inputs, axis=-1, keepdims=True)
name = 'cann-persistent.gif'

runner = bp.dyn.DSRunner(net,
                         jit=True,
                         monitors=['E2E.g', 'Ip2E.g', 'Id2E.g', 'Ip.V', 'E.V', 'E.spike', 'Ip.spike', 'Id.spike'],
                         inputs=[('E.ext_input',  Einp_scale*E_inputs, 'iter', '='),
                                 ('Id.ext_input', Iinp_scale*I_inputs, 'iter', '=')],
                         dt=global_dt)
t = runner(dur)



# ===== Raster Plot =====
fig, gs = bp.visualize.get_figure(3, 1, 1.5, 10)

fig.add_subplot(gs[:1, 0])
bp.visualize.raster_plot(runner.mon.ts, runner.mon['E.spike'], xlim=(0, dur), ylim=[0,size_E])
bp.visualize.line_plot(runner.mon.ts, bm.argmax(E_inputs, axis=1), xlim=(0, dur), legend='input peak')
# plt.xlim([runner.mon.ts[-1]*0.6, runner.mon.ts[-1]])
plt.xlim([0, runner.mon.ts[-1]])

fig.add_subplot(gs[1:2, 0])
bp.visualize.raster_plot(runner.mon.ts, runner.mon['Ip.spike'], xlim=(0, dur), ylim=[0,size_Ip])  
# plt.xlim([runner.mon.ts[-1]*0.6, runner.mon.ts[-1]])
plt.xlim([0, runner.mon.ts[-1]])

fig.add_subplot(gs[2:3, 0])
bp.visualize.raster_plot(runner.mon.ts, runner.mon['Id.spike'], xlim=(0, dur), ylim=[0,size_Id]) 
# plt.xlim([runner.mon.ts[-1]*0.6, runner.mon.ts[-1]])
plt.xlim([0, runner.mon.ts[-1]])
plt.show()


# ===== Current Visualization =====
Ec_inp = runner.mon['E2E.g']
Fc_inp = Einp_scale*E_inputs
shunting_inp = shunting_k*(runner.mon['E2E.g']+Fc_inp)*runner.mon['Ip2E.g']
r_SI = shunting_k*runner.mon['E2E.g']*runner.mon['Ip2E.g']
Ic_inp = runner.mon['Id2E.g'] + runner.mon['Ip2E.g'] + shunting_inp
total_inp = Ec_inp + Ic_inp + Fc_inp

fig, gs = bp.visualize.get_figure(4, 1, 1.5, 7)

neuron_index = 500
fig.add_subplot(gs[:2, 0])
bp.visualize.line_plot(runner.mon.ts, total_inp[:,neuron_index], xlim=(0, dur), legend='Total')  
bp.visualize.line_plot(runner.mon.ts, Fc_inp[:,neuron_index], xlim=(0, dur), legend='Fc')  
bp.visualize.line_plot(runner.mon.ts, Ec_inp[:,neuron_index], xlim=(0, dur), legend='Exc Rec') 
bp.visualize.line_plot(runner.mon.ts, (Ic_inp-shunting_inp)[:,neuron_index], xlim=(0, dur), legend='Inh (ex. SI)')  
bp.visualize.line_plot(runner.mon.ts, shunting_inp[:,neuron_index], xlim=(0, dur), legend='SI')
bp.visualize.line_plot(runner.mon.ts, -gl * runner.mon['E.V'][:,neuron_index], xlim=(0, dur), legend='Leak')

fig.add_subplot(gs[2:, 0])
bp.visualize.line_plot(runner.mon.ts, (Ic_inp-shunting_inp)[:,neuron_index], xlim=(0, dur), legend='Id+Ip')  
bp.visualize.line_plot(runner.mon.ts, r_SI[:,neuron_index], xlim=(0, dur), legend='SI (ex. Fc)', show=True) 


# ==== membrane potential and ff input ====
Fc_inp = Einp_scale*E_inputs
fig, gs = bp.visualize.get_figure(2, 1, 1.5, 7)
neuron_index = 500
fig.add_subplot(gs[:1, 0])
bp.visualize.line_plot(runner.mon.ts, Fc_inp[:,neuron_index], xlim=(0, dur), legend='ff inp')
fig.add_subplot(gs[1:2, 0])
bp.visualize.line_plot(runner.mon.ts, runner.mon['E.V'][:,neuron_index], xlim=(0, dur), legend='mem potential', show=True)

# neuron_index = 900
# fig.add_subplot(gs[:1, 1])
# bp.visualize.line_plot(runner.mon.ts, Fc_inp[:,neuron_index], xlim=(0, dur), legend='ff inp')
# fig.add_subplot(gs[1:2, 1])
# bp.visualize.line_plot(runner.mon.ts, runner.mon['E.V'][:,neuron_index], xlim=(0, dur), legend='mem potential', show=True)


# ===== heat map =====

def moving_average(a, n, axis):
    ret = bm.cumsum(a, axis=axis, dtype=bm.float32)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


plt.figure()
T = 1000
ma = moving_average(runner.mon['E.spike'], n=T, axis=0) # average window: 10ms
bump_activity = bm.vstack([bm.sum(ma * bm.cos(net.x[None,]), axis=1),bm.sum(ma * bm.sin(net.x[None,]), axis=1)])
readout = bm.array([[1., 0.]]) @ bump_activity
firing_rate = ma / (T * global_dt / 1000) 
plt.subplot(2,1,1)
plt.plot(readout.T / bm.max(readout))
plt.plot(E_inputs[T-1:,500] / bm.max(E_inputs[T-1:,500]))
# plt.xlim([int(runner.mon.ts.shape[0]*0.6), runner.mon.ts.shape[0]])
plt.xlim([0, runner.mon.ts.shape[0]])

plt.subplot(2,1,2)
plt.imshow(firing_rate.T, aspect='auto', cmap='gray_r')
plt.plot(bm.argmax(E_inputs, axis=1)[T-1:], label='input peak', color='red')
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



