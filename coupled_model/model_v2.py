import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
import numpy as np
from UnitExpCUBA import UnitExpCUBA
from brainpy.dyn import TwoEndConn


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
        I = self.gl*V + inputs + ext_input
        dvdt = I / self.tau
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


class EICANN(bp.dyn.Network):
    def __init__(self, size_E, size_Ip, size_Id, tau_E, tau_I, tau_Es, tau_Is, tau_Ef, tau_If,
                 V_reset, V_threshold, prob, JEE, JEI, JII, JIE, gl, gEE, gEIp, gIpIp, gIpE, shunting_k):
        self.conn_a = 2 * (bm.pi/6)**2
        self.stim_a = 2 * (bm.pi/6)**2
        self.size_E, self.size_Ip, self.size_Id = size_E, size_Ip, size_Id
        self.shunting_k = shunting_k
        self.J = 1.
        self.A = 1.

        # neurons
        self.E = LIF(size=size_E, gl=gl, tau=tau_E, vth=V_threshold, vreset=V_reset, tau_ref=0)
        self.Ip = LIF(size=size_Ip, gl=gl, tau=tau_I, vth=V_threshold, vreset=V_reset, tau_ref=0)
        self.Id = LIF(size=size_Id, gl=gl, tau=tau_I, vth=V_threshold, vreset=V_reset, tau_ref=0)

        w = lambda size_pre, size_post, prob: self.make_gauss_conn(size_pre, size_post, prob)
        r = lambda size_pre, size_post, prob: self.make_rand_conn(size_pre, size_post, prob)

        E2E_fw, E2I_fw, I2I_fw, I2E_fw = JEE * r(size_E, size_E, prob), JEI * r(size_E, size_Id, prob), \
                                         JII * r(size_Id, size_Id, prob), JIE * r(size_Id, size_E, prob)

        # ======== EI balance =====
        self.E2E_f = UnitExpCUBA(pre=self.E, post=self.E, conn=bp.conn.FixedProb(prob), tau=tau_Ef, g_max=JEE)
        self.E2I_f = UnitExpCUBA(pre=self.E, post=self.Id, conn=bp.conn.FixedProb(prob), tau=tau_Ef, g_max=JEI)
        self.I2I_f = UnitExpCUBA(pre=self.Id, post=self.Id, conn=bp.conn.FixedProb(prob), tau=tau_If, g_max=JII)
        self.I2E_f = UnitExpCUBA(pre=self.Id, post=self.E, conn=bp.conn.FixedProb(prob), tau=tau_If, g_max=JIE)

        # ======= CANN =====
        E2E_sw, E2I_sw, I2I_sw, I2E_sw = gEE * w(size_E, size_E, 1.0), gEIp * r(size_E, size_Ip, prob), \
                                         gIpIp * r(size_Ip, size_Ip, prob), gIpE * r(size_Ip, size_E, prob)

        self.E2E_s = UnitExpCUBA(pre=self.E, post=self.E, conn=bp.connect.All2All(), tau=tau_Es, g_max=E2E_sw)
        self.E2I_s = UnitExpCUBA(pre=self.E, post=self.Ip, conn=bp.conn.FixedProb(prob), tau=tau_Es, g_max=gEIp)
        self.I2I_s = UnitExpCUBA(pre=self.Ip, post=self.Ip, conn=bp.conn.FixedProb(prob), tau=tau_Is, g_max=gIpIp)
        self.I2E_s = UnitExpCUBA(pre=self.Ip, post=self.E, conn=bp.conn.FixedProb(prob), tau=tau_Is, g_max=gIpE)
        self.ESI = Shunting(E2Esyn_s=self.E2E_s, E2Esyn_f=self.E2E_f, I2Esyn_s=self.I2E_s, k=shunting_k, EGroup=self.E)

        super(EICANN, self).__init__()

        print('[Weights]')
        print("------------- EI Balanced ---------------")
        print("|  E2E   |  E2I    |  I2I     |  I2E    | ")
        print(f"|{E2E_fw.max():.5f} | {E2I_fw.max():.5f} | {I2I_fw.min():.5f} | {I2E_fw.min():.5f}|")
        print("---------------- CANN -------------------")
        print("|  E2E   |  E2I    |  I2I     |  I2E    | ")
        print(f"|{E2E_sw.max():.5f} | {E2I_sw.max():.5f} | {I2I_sw.min():.5f}  | {I2E_sw.min():.5f}|")
        super(EICANN, self).__init__()

    def dist(self, d):
        d = bm.remainder(d, 2 * bm.pi)
        d = bm.where(d > bm.pi, d - 2 * bm.pi, d)
        return d

    def make_gauss_conn(self, size_pre, size_post, prob):
        x_left = bm.reshape(bm.linspace(-bm.pi, bm.pi, size_pre), (-1, 1))
        x_right = bm.reshape(bm.linspace(-bm.pi, bm.pi, size_post), (1, -1))
        d = self.dist(x_left - x_right)
        w_ = self.J * bm.exp(-bm.pi * bm.square(d / self.conn_a))
        # w = w_ / bm.sum(w_, axis=-1, keepdims=True)
        w = w_
        prob_mask = (bm.random.rand(size_pre, size_post) < prob).astype(bm.float32)
        w = w * prob_mask
        return w

    def make_rand_conn(self, size_pre, size_post, prob):
        const_conn = lambda w, p: (w > 1 - p)
        w = const_conn(bm.random.rand(size_pre, size_post), prob).astype(bm.float32)
        return w

    def get_stimulus_by_pos(self, pos, size_n):
        x = bm.linspace(-bm.pi, bm.pi, size_n)
        if bm.ndim(pos) == 2:
            x = x[None,]
        I = self.A * bm.exp(-bm.pi * bm.square(self.dist(x - pos) / (2 * self.a)))
        return I
