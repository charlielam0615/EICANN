import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
import pdb

bp.math.set_platform('cpu')

# set parameters

num = 1000
num_inh = int(num * 0.2)
num_exc = num - num_inh
prob = 0.25
num_ff = int(num * prob)

tau_E = 15.
tau_I = 10.
tau_Es = 6.
tau_Is = 5.

V_reset = 0.
V_threshold = 10.

# Different from the original Tian et al. paper
# mu_f is not normalized in the paper
# mu_f = 10. / bm.sqrt(num_ff)
mu_f = bm.array(1.0)

f_E = 6
f_I = 4
JEE = 0.25 / bm.sqrt(num)
JIE = -1. / bm.sqrt(num)
JEI = 0.4 / bm.sqrt(num)
JII = -1. / bm.sqrt(num)


class LIF(bp.dyn.NeuGroup):
    def __init__(self, size, tau, **kwargs):
        super(LIF, self).__init__(size, **kwargs)

        # parameters
        self.tau = tau

        # variables
        self.V = bp.math.Variable(bp.math.zeros(size))
        self.spike = bp.math.Variable(bp.math.zeros(size, dtype=bool))
        self.input = bp.math.Variable(bp.math.zeros(size))

        # integral
        # self.integral = bp.odeint(lambda V, t, Isyn: (-V + Isyn) / self.tau)
        self.integral = bp.odeint(lambda V, t, Isyn: (Isyn) / self.tau)

    def update(self, _t, _dt):
        V = self.integral(self.V, _t, self.input, _dt)
        self.spike.value = V >= V_threshold
        self.V.value = bm.where(self.spike, V_reset, V)
        self.input[:] = 0.



class EINet(bp.dyn.Network):
    def __init__(self):
        # neurons
        E = LIF(num_exc, tau=tau_E)
        I = LIF(num_inh, tau=tau_I)
        E.V[:] = bm.random.random(num_exc) * (V_threshold - V_reset) + V_reset
        I.V[:] = bm.random.random(num_inh) * (V_threshold - V_reset) + V_reset

        # synapses
        self.E2I = bp.dyn.ExpCUBA(pre=E, post=I, conn=bp.conn.FixedProb(prob), tau=tau_Es, g_max=JEI)
        self.E2E = bp.dyn.ExpCUBA(pre=E, post=E, conn=bp.conn.FixedProb(prob), tau=tau_Es, g_max=JEE)
        self.I2I = bp.dyn.ExpCUBA(pre=I, post=I, conn=bp.conn.FixedProb(prob), tau=tau_Is, g_max=JII)
        self.I2E = bp.dyn.ExpCUBA(pre=I, post=E, conn=bp.conn.FixedProb(prob), tau=tau_Is, g_max=JIE)

        super(EINet, self).__init__(self.E2E, self.E2I, self.I2E, self.I2I, E=E, I=I)



net = EINet()

# ==== Fast Response to Sin Input =====
# duration = 100.
# sin_inp = bp.inputs.sinusoidal_input(amplitude=mu_f*0.1, frequency=50., duration=duration, dt=0.01) + 0.2*mu_f
# sigma_F = 0.
# noise = sigma_F * bm.random.randn(int(duration/0.01), num)
# inputs = sin_inp[:,None] + noise

# WEF = f_E/bm.sqrt(num_ff)
# WIF = f_I/bm.sqrt(num_ff)

# E_inp = num_ff * WEF * inputs[:, :num_exc]
# I_inp = num_ff * WIF * inputs[:, num_exc:]

# runner = bp.dyn.DSRunner(net,
#                          monitors=['E.spike', 'I.spike', 'E.V'],
#                          inputs=[('E.input', E_inp, 'iter'),
#                                  ('I.input', I_inp, 'iter')],
#                          dt=0.01)


# t = runner.run(duration)


# fig, gs = bp.visualize.get_figure(6, 1, 1.5, 10)

# fig.add_subplot(gs[:2, 0])
# bp.visualize.raster_plot(runner.mon.ts, runner.mon['E.spike'], xlim=(0, duration))
# fig.add_subplot(gs[2:4, 0])
# bp.visualize.line_plot(runner.mon.ts, bm.mean(inputs, axis=1), xlim=(0, duration))
# fig.add_subplot(gs[4:, 0])
# bp.visualize.line_plot(runner.mon.ts, bm.mean(runner.mon['E.spike'].astype(bm.float32),axis=1)/0.01*1000, xlim=(0, duration), show=True)  




# ===== Amplitude Tracking =====
# duration = 100.
# ramp_inp = bp.inputs.ramp_input(c_start=0.1*mu_f.numpy(), c_end=1.5*mu_f.numpy(), duration=duration/2, dt=0.01)
# inputs = bm.ones([int(duration/0.01), num]) * 0.1*mu_f.numpy()
# inputs[int(duration/0.01/4):int(duration/0.01/4)+len(ramp_inp)] = ramp_inp[:,None]
# inputs[int(duration/0.01/4)+len(ramp_inp):] = 1.0 * mu_f
# sigma_F = 0.
# noise = sigma_F * bm.random.randn(int(duration/0.01), num)

# WEF = f_E/bm.sqrt(num_ff)
# WIF = f_I/bm.sqrt(num_ff)

# E_inp = num_ff * WEF * (inputs[:, :num_exc] + noise[:, :num_exc])
# I_inp = num_ff * WIF * (inputs[:, num_exc:] + noise[:, num_exc:])

# runner = bp.dyn.DSRunner(net,
#                          monitors=['E2I.g', 'E2E.g', 'I2I.g', 'I2E.g', 'E.input', 'E.spike', 'I.spike', 'E.V'],
#                          inputs=[('E.input', E_inp, 'iter'),
#                                  ('I.input', I_inp, 'iter')],
#                          dt=0.01)


# t = runner.run(duration)

# fig, gs = bp.visualize.get_figure(2, 1, 1.5, 10)
# fig.add_subplot(gs[:, 0])
# bp.visualize.line_plot(runner.mon.ts, 1e2*bm.mean(runner.mon['E.spike'].astype(bm.float32),axis=1), xlim=(0, duration))  
# bp.visualize.line_plot(runner.mon.ts, bm.mean(inputs, axis=1), xlim=(0, duration), show=True)  


# ===== Current Visualization ====
duration = 100.

WEF = f_E/bm.sqrt(num_ff)
WIF = f_I/bm.sqrt(num_ff)

E_inp = num_ff * WEF * mu_f * 0.1
I_inp = num_ff * WIF * mu_f * 0.1

Fc_inp = E_inp

runner = bp.dyn.DSRunner(net,
                         monitors=['E2I.g', 'E2E.g', 'I2I.g', 'I2E.g', 'E.input', 'E.spike', 'I.spike', 'E.V'],
                         inputs=[('E.input', E_inp),
                                 ('I.input', I_inp)],
                         dt=0.01)

t = runner.run(duration)
# Inspect an E neuron
total_inp = runner.mon['E2E.g'] + runner.mon['I2E.g'] + E_inp
Ec_inp = runner.mon['E2E.g']
Ic_inp = runner.mon['I2E.g']

fig, gs = bp.visualize.get_figure(4, 1, 1.5, 7)
fig.add_subplot(gs[:1, 0])
bp.visualize.line_plot(runner.mon.ts, total_inp[:,0], xlim=(0, duration), legend='Total')  
bp.visualize.line_plot(runner.mon.ts, Ec_inp[:,0]+Fc_inp, xlim=(0, duration), legend='Excitatory') 
bp.visualize.line_plot(runner.mon.ts, Ic_inp[:,0], xlim=(0, duration), legend='Inhibitory')  

fig.add_subplot(gs[1:2, 0])
bp.visualize.line_plot(runner.mon.ts, runner.mon['E.V'], xlim=(0, duration), legend='membrane potential')  

fig.add_subplot(gs[2:3, 0])
bp.visualize.raster_plot(runner.mon.ts, runner.mon['E.spike'], xlim=(0, duration))

fig.add_subplot(gs[3:, 0])
bp.visualize.raster_plot(runner.mon.ts, runner.mon['I.spike'], xlim=(0, duration), show=True)



