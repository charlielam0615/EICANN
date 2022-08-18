import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
import pdb

bp.math.set_platform('cpu')

# set parameters

ff_num = 800
num = 20000
num_inh = int(num * 0.2)
num_exc = num - num_inh
Eprob = 0.05
Iprob = 0.2
V_reset = 0.
# V_threshold = 7.07
V_threshold = 3.
JEE = 0.25
JIE = -0.25
JEI = 0.25
JII = -0.25

tau_Es = 6.
tau_Is = 5.
tau_E = 15.
tau_I = 10.

f_E = 10.
f_I = 10.
mu_f = 1.


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
        self.integral = bp.odeint(lambda V, t, Isyn: (-V + Isyn) / self.tau)

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
        self.E2I = bp.dyn.ExpCUBA(pre=E, post=I, conn=bp.conn.FixedProb(Eprob), tau=tau_Es, g_max=JEI)
        self.E2E = bp.dyn.ExpCUBA(pre=E, post=E, conn=bp.conn.FixedProb(Eprob), tau=tau_Es, g_max=JEE)
        self.I2I = bp.dyn.ExpCUBA(pre=I, post=I, conn=bp.conn.FixedProb(Iprob), tau=tau_Is, g_max=JII)
        self.I2E = bp.dyn.ExpCUBA(pre=I, post=E, conn=bp.conn.FixedProb(Iprob), tau=tau_Is, g_max=JIE)

        super(EINet, self).__init__(self.E2E, self.E2I, self.I2E, self.I2I, E=E, I=I)



net = EINet()

# ==== Fast Response to Sin Input =====
# duration = 100.
# sin_inp = bp.inputs.sinusoidal_input(amplitude=0.3, frequency=50., duration=duration, dt=0.01) + mu_f
# sigma_F = 1.
# noise = sigma_F * bm.random.randn(int(duration/0.01), num)

# E_inp = f_E * (sin_inp[:,None] + noise[:, :num_exc])
# I_inp = f_I * (sin_inp[:,None] + noise[:, num_exc:])

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
# bp.visualize.line_plot(runner.mon.ts, bm.mean(E_inp,axis=1)/f_E, xlim=(0, duration))
# fig.add_subplot(gs[4:, 0])
# bp.visualize.line_plot(runner.mon.ts, bm.mean(runner.mon['E.spike'].astype(bm.float32),axis=1)/0.01*1000, xlim=(0, duration), ylim=(0, 100), show=True)  


# ===== Amplitude Tracking =====
# duration = 100.
# ramp_inp = bp.inputs.ramp_input(c_start=1., c_end=15.0, duration=duration/2, dt=0.01)
# inputs = bm.ones([int(duration/0.01), num])*1.
# inputs[int(duration/0.01/4):int(duration/0.01/4)+len(ramp_inp)] = ramp_inp[:,None]
# inputs[int(duration/0.01/4)+len(ramp_inp):] = 10.0
# sigma_F = 0.
# noise = sigma_F * bm.random.randn(int(duration/0.01), num)
# E_inp = f_E * (inputs[:, :num_exc] + noise[:, :num_exc])
# I_inp = f_I * (inputs[:, num_exc:] + noise[:, num_exc:])

# runner = bp.dyn.DSRunner(net,
#                          monitors=['E.spike', 'I.spike', 'E.V'],
#                          inputs=[('E.input', E_inp, 'iter'),
#                                  ('I.input', I_inp, 'iter')],
#                          dt=0.01)


# t = runner.run(duration)

# fig, gs = bp.visualize.get_figure(2, 1, 1.5, 10)
# fig.add_subplot(gs[:, 0])
# bp.visualize.line_plot(runner.mon.ts, 3e3*bm.mean(runner.mon['E.spike'].astype(bm.float32),axis=1), xlim=(0, duration))  
# bp.visualize.line_plot(runner.mon.ts, bm.mean(inputs, axis=1), xlim=(0, duration), show=True)  

# ===== Current Visualization ====

duration = 100.

E_inp = f_E * mu_f * 0.5
I_inp = f_I * mu_f * 0.5

runner = bp.dyn.DSRunner(net,
                         monitors=['E2I.g', 'E2E.g', 'I2I.g', 'I2E.g', 'E.input', 'E.spike', 'I.spike', 'E.V'],
                         inputs=[('E.input', E_inp),
                                 ('I.input', I_inp)],
                         dt=0.01)

t = runner.run(duration)

# Inspect an E neuron
total_inp = runner.mon['E2E.g'] + runner.mon['I2E.g'] + runner.mon['E.input']
Ec_inp = runner.mon['E2E.g']
Ic_inp = runner.mon['I2E.g']
Fc_inp = runner.mon['E.input']

fig, gs = bp.visualize.get_figure(4, 1, 1.5, 7)
fig.add_subplot(gs[:1, 0])
bp.visualize.line_plot(runner.mon.ts, total_inp[:,0], xlim=(0, duration), legend='Total')  
bp.visualize.line_plot(runner.mon.ts, Ec_inp[:,0], xlim=(0, duration), legend='Excitatory') 
bp.visualize.line_plot(runner.mon.ts, Ic_inp[:,0], xlim=(0, duration), legend='Inhibitory')  

fig.add_subplot(gs[1:2, 0])
bp.visualize.line_plot(runner.mon.ts, runner.mon['E.V'], xlim=(0, duration), legend='membrane potential')  

fig.add_subplot(gs[2:3, 0])
bp.visualize.raster_plot(runner.mon.ts, runner.mon['E.spike'], xlim=(0, duration))

fig.add_subplot(gs[3:, 0])
bp.visualize.raster_plot(runner.mon.ts, runner.mon['I.spike'], xlim=(0, duration), show=True)


