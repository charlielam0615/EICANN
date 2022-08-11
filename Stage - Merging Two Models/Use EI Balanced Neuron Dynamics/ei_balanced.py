import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
from UnitExpCUBA import UnitExpCUBA

import pdb

bp.math.set_platform('cpu')

# set parameters
num = 1000
qI = 0.25
qE = 1 - qI
num_inh = int(num * qI)
num_exc = int(num * qE)
num_ff = num
prob = 0.25
tau_E = 1.5
tau_I = 1.0
tau_Es = 0.6
tau_Is = 0.5

V_reset = 0.
V_threshold = 3.

mu_f = 0.5
scale = 20.

f_E = 1.1 
f_I = 1. 
jie = -1 * scale
jii = -1 * scale
jee = 0.25 * scale
jei = 0.4 * scale
JIE = jie / bm.sqrt(num) 
JII = jii / bm.sqrt(num)
JEE = jee / bm.sqrt(num)
JEI = jei / bm.sqrt(num)
gl = 0.000268 * scale * bm.sqrt(num)

class LIF(bp.dyn.NeuGroup):
    def __init__(self, size, tau, gl, vth, vreset, tau_ref, **kwargs):
        super(LIF, self).__init__(size, **kwargs)

        # parameters
        self.size = size
        self.tau = tau
        self.vth = vth + (bm.random.rand(self.size) - 0.5) * 0.5
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



class EINet(bp.dyn.Network):
    def __init__(self):
        # neurons
        E = LIF(num_exc, tau=tau_E, gl=gl, vth=V_threshold, vreset=V_reset, tau_ref=0)
        I = LIF(num_inh, tau=tau_I, gl=gl, vth=V_threshold, vreset=V_reset, tau_ref=0)

        # synapses
        self.E2I = UnitExpCUBA(pre=E, post=I, conn=bp.conn.FixedProb(prob), tau=tau_Es, g_max=JEI)
        self.E2E = UnitExpCUBA(pre=E, post=E, conn=bp.conn.FixedProb(prob), tau=tau_Es, g_max=JEE)
        self.I2I = UnitExpCUBA(pre=I, post=I, conn=bp.conn.FixedProb(prob), tau=tau_Is, g_max=JII)
        self.I2E = UnitExpCUBA(pre=I, post=E, conn=bp.conn.FixedProb(prob), tau=tau_Is, g_max=JIE)

        super(EINet, self).__init__(self.E2E, self.E2I, self.I2E, self.I2I, E=E, I=I)

w = lambda j, q: prob * j * q

def compute_re(mu):
    nume = f_E*w(jii, qI) - f_I*w(jie, qI) - 1/bm.sqrt(num)*f_E*V_threshold*tau_I
    deno = w(jie, qI)*w(jei, qE)-w(jee, qE)*w(jii, qI) \
        + 1/bm.sqrt(num)*V_threshold*(w(jee, qE)*tau_I+w(jii, qI)*tau_E) \
        - 1/num * V_threshold**2 * tau_E * tau_I
    re = nume / deno * mu
    return re

def compute_ri(mu):
    nume = f_I*w(jee, qE) - f_E*w(jei, qE) - 1/bm.sqrt(num)*f_I*V_threshold*tau_E
    deno = w(jie, qI)*w(jei, qE)-w(jee, qE)*w(jii, qI) \
        + 1/bm.sqrt(num)*V_threshold*(w(jee, qE)*tau_I+w(jii, qI)*tau_E) \
        - 1/num * V_threshold**2 * tau_E * tau_I
    ri = nume / deno * mu
    return ri


net = EINet()

# ==== Fast Response to Sin Input =====
duration = 100.
sin_inp = bp.inputs.sinusoidal_input(amplitude=mu_f, frequency=50., duration=duration, dt=0.01) + mu_f
sigma_F = 0.
noise = sigma_F * bm.random.randn(int(duration/0.01), num_exc+num_inh)
inputs = sin_inp[:,None] + noise

E_inp = num_ff * f_E / bm.sqrt(num) * inputs[:, :num_exc]
I_inp = num_ff * f_I / bm.sqrt(num) * inputs[:, num_exc:]

runner = bp.dyn.DSRunner(net,
                         monitors=['E2I.g', 'E2E.g', 'I2I.g', 'I2E.g', 'E.input', 'E.spike', 'I.spike', 'E.V'],
                         inputs=[('E.ext_input', E_inp, 'iter', '='),
                                 ('I.ext_input', I_inp, 'iter', '=')],
                         dt=0.01)

re = compute_re(sin_inp)
ri = compute_ri(sin_inp)


t = runner.run(duration)


fig, gs = bp.visualize.get_figure(6, 1, 1.5, 10)

fig.add_subplot(gs[:2, 0])
bp.visualize.raster_plot(runner.mon.ts, runner.mon['E.spike'], xlim=(0, duration))
fig.add_subplot(gs[2:4, 0])
bp.visualize.line_plot(runner.mon.ts, bm.mean(inputs, axis=1), xlim=(0, duration))
fig.add_subplot(gs[4:6, 0])
bp.visualize.line_plot(runner.mon.ts, bm.mean(runner.mon['E.spike'].astype(bm.float32),axis=1)/0.01, xlim=(0, duration), legend='firing rate')  
bp.visualize.line_plot(runner.mon.ts, re, xlim=(0, duration), legend='prediction', show=True)



# ===== Amplitude Tracking =====
# duration = 100.
# ramp_inp = bp.inputs.ramp_input(c_start=0.1*mu_f, c_end=1.5*mu_f, duration=duration/2, dt=0.01)
# inputs = bm.ones([int(duration/0.01), num_exc+num_inh]) * 0.1*mu_f
# inputs[int(duration/0.01/4):int(duration/0.01/4)+len(ramp_inp)] = ramp_inp[:,None]
# inputs[int(duration/0.01/4)+len(ramp_inp):] = 1.0 * mu_f
# sigma_F = inputs * 1.0
# noise = sigma_F * bm.random.randn(int(duration/0.01), num_exc+num_inh)

# E_inp = num_ff * f_E * (inputs[:, :num_exc] + noise[:, :num_exc])
# I_inp = num_ff * f_I * (inputs[:, num_exc:] + noise[:, num_exc:])

# runner = bp.dyn.DSRunner(net,
#                          monitors=['E2I.g', 'E2E.g', 'I2I.g', 'I2E.g', 'E.input', 'E.spike', 'I.spike', 'E.V'],
#                          inputs=[('E.input', E_inp, 'iter'),
#                                  ('I.input', I_inp, 'iter')],
#                          dt=0.01)

# t = runner.run(duration)

# fig, gs = bp.visualize.get_figure(2, 1, 1.5, 10)
# fig.add_subplot(gs[:, 0])
# bp.visualize.line_plot(runner.mon.ts, 2.5e2*bm.mean(runner.mon['E.spike'].astype(bm.float32),axis=1), xlim=(0, duration))  
# bp.visualize.line_plot(runner.mon.ts, bm.mean(inputs, axis=1), xlim=(0, duration), show=True)  


# ===== Current Visualization ====
# duration = 100.

# E_inp = num_ff * f_E * mu_f
# I_inp = num_ff * f_I * mu_f

Fc_inp = E_inp

# runner = bp.dyn.DSRunner(net,
#                          monitors=['E2I.g', 'E2E.g', 'I2I.g', 'I2E.g', 'E.input', 'E.spike', 'I.spike', 'E.V'],
#                          inputs=[('E.input', E_inp),
#                                  ('I.input', I_inp)],
#                          dt=0.01)

# t = runner.run(duration)
# Inspect an E neuron

total_inp = runner.mon['E2E.g'] + runner.mon['I2E.g'] + E_inp
Ec_inp = runner.mon['E2E.g']
Ic_inp = runner.mon['I2E.g']

fig, gs = bp.visualize.get_figure(4, 1, 1.5, 7)
fig.add_subplot(gs[:1, 0])
bp.visualize.line_plot(runner.mon.ts, total_inp[:,0], xlim=(0, duration), legend='Total')  
bp.visualize.line_plot(runner.mon.ts, (Ec_inp+Fc_inp)[:,0], xlim=(0, duration), legend='Excitatory') 
bp.visualize.line_plot(runner.mon.ts, Ic_inp[:,0], xlim=(0, duration), legend='Inhibitory')  

fig.add_subplot(gs[1:2, 0])
bp.visualize.line_plot(runner.mon.ts, runner.mon['E.V'], xlim=(0, duration), legend='membrane potential')  

fig.add_subplot(gs[2:3, 0])
bp.visualize.raster_plot(runner.mon.ts, runner.mon['E.spike'], xlim=(0, duration))

fig.add_subplot(gs[3:, 0])
bp.visualize.raster_plot(runner.mon.ts, runner.mon['I.spike'], xlim=(0, duration), show=True)

# ====== Membrane potential distribution ======
# duration = 100.

# E_inp = num_ff * f_E * mu_f
# I_inp = num_ff * f_I * mu_f

# Fc_inp = E_inp

# runner = bp.dyn.DSRunner(net,
#                          monitors=['E.V'],
#                          inputs=[('E.input', E_inp),
#                                  ('I.input', I_inp)],
#                          dt=0.01)

# t = runner.run(duration)

# plt.subplot(3,1,1)
# plt.hist(runner.mon['E.V'][100], bins=20)
# plt.subplot(3,1,2)
# plt.hist(runner.mon['E.V'][1000], bins=20)
# plt.subplot(3,1,3)
# plt.hist(runner.mon['E.V'][5000], bins=20)
# plt.show()


