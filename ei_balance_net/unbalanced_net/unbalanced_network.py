import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
from UnitExpCUBA import UnitExpCUBA


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
        self.V = bm.Variable(bm.random.rand(self.size) * (self.vth-vreset) + vreset)
        self.input = bm.Variable(bm.zeros(self.size))
        self.t_last_spike = bm.Variable(bm.ones(self.size) * -1e7)
        self.refractory = bm.Variable(bm.zeros(self.size, dtype=bool))
        self.spike = bm.Variable(bm.zeros(self.size, dtype=bool))
        self.ext_input = bm.Variable(bm.zeros(self.size))

        # integral
        self.integral = bp.odeint(f=self.derivative, method='euler')

    def derivative(self, V, t, inputs, ext_input):
        dvdt = (self.gl*V + inputs + ext_input) / self.tau
        return dvdt

    def update(self, tdi):
        _t, _dt = tdi.t, tdi.dt
        refractory = (_t - self.t_last_spike) <= self.tau_ref
        V = self.integral(self.V, _t, self.input, self.ext_input, dt=_dt)
        V = bm.where(refractory, self.V, V)

        # no leak current, use a lower bound on membrane potential
        V = bm.where(V < -10, self.V, V)

        spike = self.vth <= V
        self.spike.value = spike
        self.t_last_spike.value = bm.where(spike, _t, self.t_last_spike)
        self.V.value = bm.where(spike, self.vreset, V)
        self.refractory.value = bm.logical_or(refractory, spike)
        self.input[:] = 0.


class UnbalanceNet(bp.dyn.Network):
    def __init__(self, num_exc, num_inh, tau_E, tau_I, tau_Ef, tau_If, V_threshold, V_reset, prob, gl, JEI, JEE, JII, JIE):
        # neurons
        self.E = LIF(num_exc, tau=tau_E, gl=gl, vth=V_threshold, vreset=V_reset, tau_ref=0)
        self.I = LIF(num_inh, tau=tau_I, gl=gl, vth=V_threshold, vreset=V_reset, tau_ref=0)

        # synapses
        self.E2I = UnitExpCUBA(pre=self.E, post=self.I, conn=bp.conn.FixedProb(prob), tau=tau_Ef, g_max=JEI)
        self.E2E = UnitExpCUBA(pre=self.E, post=self.E, conn=bp.conn.FixedProb(prob), tau=tau_Ef, g_max=JEE)
        self.I2I = UnitExpCUBA(pre=self.I, post=self.I, conn=bp.conn.FixedProb(prob), tau=tau_If, g_max=JII)
        self.I2E = UnitExpCUBA(pre=self.I, post=self.E, conn=bp.conn.FixedProb(prob), tau=tau_If, g_max=JIE)

        super(UnbalanceNet, self).__init__()
