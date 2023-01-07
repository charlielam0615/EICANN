import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
from UnitExpCUBA import UnitExpCUBA


class LIF2D(bp.neurons.LIF):
    def __init__(self, size, tau, tau_ref, gl, V_th, V_reset):
        super().__init__(size=size, V_reset=V_reset, V_th=V_th, tau=tau, tau_ref=None)
        self.gl = gl
        self.ext_input = bm.Variable(bm.zeros(self.size))

    def derivative(self, V, t, inputs, ext_input):
        return (self.gl*V + inputs + bm.reshape(ext_input, [-1])) / self.tau

    def update(self, tdi):
        _t, _dt = tdi.t, tdi.dt
        # integrate membrane potential
        V = self.integral(self.V, _t, self.input, self.ext_input, dt=_dt)
        # spike, spiking time, and membrane potential reset
        spike = V >= self.V_th
        V = bm.where(spike, self.V_reset, V)
        self.V.value = V
        self.spike.value = spike



class EINet2D(bp.dyn.Network):
    def __init__(self, exc_w, exc_h, inh_w, inh_h, tau_E, tau_I, tau_Ef, tau_If, V_threshold, V_reset, prob, gl, JEI, JEE, JII, JIE):
        # neurons
        self.E = LIF2D(size=(exc_w, exc_h), tau=tau_E, gl=gl, V_th=V_threshold, V_reset=V_reset, tau_ref=0)
        self.I = LIF2D(size=(inh_w, inh_h), tau=tau_I, gl=gl, V_th=V_threshold, V_reset=V_reset, tau_ref=0)

        # synapses
        self.E2I = UnitExpCUBA(pre=self.E, post=self.I, conn=bp.conn.FixedProb(prob), tau=tau_Ef, g_max=JEI)
        self.E2E = UnitExpCUBA(pre=self.E, post=self.E, conn=bp.conn.FixedProb(prob), tau=tau_Ef, g_max=JEE)
        self.I2I = UnitExpCUBA(pre=self.I, post=self.I, conn=bp.conn.FixedProb(prob), tau=tau_If, g_max=JII)
        self.I2E = UnitExpCUBA(pre=self.I, post=self.E, conn=bp.conn.FixedProb(prob), tau=tau_If, g_max=JIE)

        super(EINet2D, self).__init__()
