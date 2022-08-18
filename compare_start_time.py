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
input_amp = 0.5

cann_scale = 0.1
gEE_G = 0.065 * 3 * 8 * cann_scale
gEIp = 0.0175 * 7.5 * 6 * cann_scale
gIpE = -0.1603 * 2.5 * 4 * cann_scale
gIpIp = -0.0082 * 7.5 * 30 * cann_scale
shunting_k = 0.1

vth = 10
f_E = 1.1 * 0.5
f_I = 1.0 * 0.5
bg_str = 0.025
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

        w_ee, w_ei, w_ie, w_ii = self.make_conn(self.x)

        # synapses
        E2E_rand = (bm.random.rand(size_E, size_E) < prob).astype(bm.float32) * gEE_R
        E2E = E2E_rand + gEE_G*w_ee 
        # E2E rand mean: 0.03956277, E2E guas max: 0.02181133

        # print(f"E2E rand mean: {bm.mean(E2E_rand):.6f}, "
        #       f"E2E guas max: {bm.max(gEE_G*w_ee):.6f}")

        self.E2E = UnitExpCUBA(pre=self.E, post=self.E, conn=bp.connect.All2All(), tau=tau_Es, g_max=E2E) # mean: 0.04402414
        self.E2Ip = UnitExpCUBA(pre=self.E, post=self.Ip, conn=bp.connect.All2All(), tau=tau_Es, g_max=gEIp*w_ei) # mean: 0.007875
        self.Ip2Ip = UnitExpCUBA(pre=self.Ip, post=self.Ip, conn=bp.connect.All2All(), tau=tau_Is, g_max=gIpIp*w_ii) # mean: -0.01845
        self.Ip2E = UnitExpCUBA(pre=self.Ip, post=self.E, conn=bp.connect.All2All(), tau=tau_Is, g_max=gIpE*w_ie) # mean: -0.00534333
        self.ESI = Shunting(E2Esyn=self.E2E, I2Esyn=self.Ip2E, k=shunting_k, EGroup=self.E)

        self.E2Id = UnitExpCUBA(pre=self.E, post=self.Id, conn=bp.conn.FixedProb(prob), tau=tau_Es, g_max=gEId) # mean: 0.0632455575
        self.Id2Id = UnitExpCUBA(pre=self.Id, post=self.Id, conn=bp.conn.FixedProb(prob), tau=tau_Is, g_max=gIdId) # mean: -0.158113875
        self.Id2E = UnitExpCUBA(pre=self.Id, post=self.E, conn=bp.conn.FixedProb(prob), tau=tau_Is, g_max=gIdE) # mean: -0.158113875

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


class SCANN(bp.dyn.Network):
    def __init__(self):
        self.a = bm.sqrt(2*bm.pi) * (bm.pi/6)
        self.J = 4.
        self.A = input_amp
        self.x = bm.linspace(-bm.pi, bm.pi, size_E)

        # neurons
        self.E = LIF(size=size_E, gl=gl, tau=tau_E, vth=vth, vreset=vreset, tau_ref=0)
        self.I = LIF(size=size_Ip, gl=gl, tau=tau_I, vth=vth, vreset=vreset, tau_ref=0)

        self.E.V[:] = bm.random.random(size_E) * (self.E.vth - vreset) + vreset
        self.I.V[:] = bm.random.random(size_Ip) * (self.I.vth - vreset) + vreset

        w_ee, w_ei, w_ie, w_ii = self.make_conn(self.x)

        # synapses
        self.E2E = UnitExpCUBA(pre=self.E, post=self.E, conn=bp.connect.All2All(), tau=tau_Es, g_max=gEE_G*w_ee)
        self.E2I = UnitExpCUBA(pre=self.E, post=self.I, conn=bp.connect.All2All(), tau=tau_Es, g_max=gEIp*w_ei)
        self.I2I = UnitExpCUBA(pre=self.I, post=self.I, conn=bp.connect.All2All(), tau=tau_Is, g_max=gIpIp*w_ii)
        self.I2E = UnitExpCUBA(pre=self.I, post=self.E, conn=bp.connect.All2All(), tau=tau_Is, g_max=gIpE*w_ie)
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


def moving_average(a, n, axis):
    ret = bm.cumsum(a, axis=axis, dtype=bm.float32)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


eicann = EICANN()
scann = SCANN()

Einp_scale = size_ff * f_E / bm.sqrt(num)
Iinp_scale = size_ff * f_I / bm.sqrt(num)

# ===== EI CANN ====

eicann_bump_inp = eicann.get_stimulus_by_pos(0.)
eicann_bg_inputs = bg_str * eicann.A
eicann_inputs, eicann_dur = bp.inputs.section_input(values=[eicann_bg_inputs, eicann_bump_inp+eicann_bg_inputs, eicann_bg_inputs],
                                             durations=[20., 20., 10.],
                                             return_length=True,
                                             dt=global_dt)
eicann_mean_inputs = bm.mean(eicann_inputs, axis=-1, keepdims=True)
eicann_E_noise = bm.random.randn(*eicann_inputs.shape) * bm.sqrt(eicann_inputs)
eicann_I_noise = bm.random.randn(*eicann_mean_inputs.shape) * bm.sqrt(eicann_mean_inputs)

eicann_E_inp = Einp_scale*eicann_inputs + eicann_E_noise
eicann_I_inp = 0.5*Iinp_scale*eicann_mean_inputs + eicann_I_noise

# ===== SCANN ====

scann_bump_inp = scann.get_stimulus_by_pos(0.)
scann_bg_inputs = bg_str * scann.A
scann_inputs, scann_dur = bp.inputs.section_input(values=[scann_bg_inputs, scann_bg_inputs+scann_bump_inp, scann_bg_inputs],
                                         durations=[20., 20., 10.],
                                         return_length=True,
                                         dt=global_dt)
scann_mean_inputs = bm.mean(scann_inputs, axis=-1, keepdims=True)
scann_E_noise = bm.random.randn(*scann_inputs.shape) * bm.sqrt(scann_inputs)
scann_I_noise = bm.random.randn(*scann_mean_inputs.shape) * bm.sqrt(scann_mean_inputs)
scann_E_inp = Einp_scale*scann_inputs + scann_E_noise
scann_I_inp = Iinp_scale*scann_mean_inputs + scann_I_noise

eicann_runner = bp.dyn.DSRunner(eicann,
                                 jit=True,
                                 monitors=['E2E.g', 'Ip2E.g', 'Id2E.g', 'Ip.V', 'E.V', 'E.spike', 'Ip.spike', 'Id.spike'],
                                 inputs=[('E.ext_input',  eicann_E_inp, 'iter', '='),
                                         ('Id.ext_input', eicann_I_inp, 'iter', '='),
                                         ('Ip.ext_input', eicann_I_inp, 'iter', '=')],
                                 dt=global_dt)
eicann_t = eicann_runner(eicann_dur)

scann_runner = bp.dyn.DSRunner(scann,
                                 jit=True,
                                 monitors=['E2I.g', 'E2E.g', 'I2E.g', 'I2I.g', 'I.V', 'E.V', 'E.spike', 'I.spike'],
                                 inputs=[('E.ext_input', scann_E_inp, 'iter', '='),
                                         ('I.ext_input', scann_I_inp, 'iter', '=')],
                                 dt=global_dt)
scann_t = scann_runner(scann_dur)


T = 10
eicann_ma = moving_average(eicann_runner.mon['E.spike'], n=T, axis=0) # average window: 1ms
eicann_bump_activity = bm.vstack([bm.sum(eicann_ma * bm.cos(eicann.x[None,]), axis=1),bm.sum(eicann_ma * bm.sin(eicann.x[None,]), axis=1)])
eicann_readout = bm.array([[1., 0.]]) @ eicann_bump_activity
nm_eicann_readout = eicann_readout.T / bm.mean(eicann_readout[:,2000:4000])

scann_ma = moving_average(scann_runner.mon['E.spike'], n=T, axis=0) # average window: 1ms
scann_bump_activity = bm.vstack([bm.sum(scann_ma * bm.cos(scann.x[None,]), axis=1),bm.sum(scann_ma * bm.sin(scann.x[None,]), axis=1)])
scann_readout = bm.array([[1., 0.]]) @ scann_bump_activity
nm_scann_readout = scann_readout.T / bm.mean(scann_readout[:,2000:4000])

eicann_time = bm.argmax(nm_eicann_readout>0.5)
scann_time = bm.argmax(nm_scann_readout>0.5)
# print(eicann_time, scann_time)


# ==== raster plot =====
# fig, gs = bp.visualize.get_figure(4, 1, 1.5, 10)

# fig.add_subplot(gs[:2, 0])
# bp.visualize.raster_plot(eicann_runner.mon.ts, eicann_runner.mon['E.spike'], xlim=(0, eicann_dur), ylim=[0,size_E])

# fig.add_subplot(gs[2:, 0])
# bp.visualize.raster_plot(scann_runner.mon.ts, scann_runner.mon['E.spike'], xlim=(0, scann_dur), ylim=[0,size_E], show=True)  

# ===== population decoding ======
# T = 10
# plt.figure()
# plt.subplot(2,1,1)
# plt.plot(nm_eicann_readout)
# plt.plot(eicann_E_inp[T-1:,375] / bm.max(eicann_E_inp[T-1:,375]))
# plt.xlim([0, eicann_runner.mon.ts.shape[0]])
# plt.subplot(2,1,2)
# plt.plot(nm_scann_readout)
# plt.plot(scann_E_inp[T-1:,375] / bm.max(scann_E_inp[T-1:,375]))
# plt.xlim([0, scann_runner.mon.ts.shape[0]])
# plt.show()





import csv
with open(r'eicann_time.csv', 'a') as f:
    writer = csv.writer(f)
    writer.writerow([eicann_time])

with open(r'scann_time.csv', 'a') as f:
    writer = csv.writer(f)
    writer.writerow([scann_time])


