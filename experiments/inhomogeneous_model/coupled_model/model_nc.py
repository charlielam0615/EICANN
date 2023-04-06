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
        self.vth = vth
        self.gl = gl
        self.vreset = vreset
        self.tau_ref = tau_ref
        self.rng = bm.random.RandomState(seed=1024)

        # variables
        self.V = bm.Variable(self.rng.rand(self.size) * (self.vth - vreset) + vreset)
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

    def reset_state(self, batch_size=None):
        self.rng.seed(1024)
        self.V[:] = self.rng.rand(self.size) * (self.vth - self.vreset) + self.vreset
        self.input[:] = bm.zeros(self.size)
        self.t_last_spike[:] = bm.ones(self.size) * -1e7
        self.refractory[:] = bm.zeros(self.size, dtype=bool)
        self.spike[:] = bm.zeros(self.size, dtype=bool)
        self.ext_input[:] = bm.zeros(self.size)




class EIContinousAttractor(bp.Network):
    def __init__(self, size_E, size_Ip, size_Id, tau_E, tau_I, tau_Es, tau_Is, tau_Ef, tau_If, prob,
                 V_reset, V_threshold, JEE, JEI, JII, JIE, gl, gEE, gEIp, gIpIp, gIpE, shunting_k):

        self.conn_a = 2 * (bm.pi/6)**2
        self.stim_a = 2 * (bm.pi/6)**2
        self.size_E, self.size_Ip, self.size_Id = size_E, size_Ip, size_Id
        self.tau_E, self.tau_I = tau_E, tau_I
        self.tau_Es, self.tau_Is = tau_Es, tau_Is
        self.tau_Ef, self.tau_If = tau_Ef, tau_If
        self.gl, self.gEE, self.gEIp, self.gIpIp, self.gIpE = gl, gEE, gEIp, gIpIp, gIpE
        self.JEE, self.JEI, self.JII, self.JIE = JEE, JEI, JII, JIE
        self.shunting_k = shunting_k
        self.J = 1.
        self.A = 1.
        self.prob = prob

        # neurons
        self.rng = bm.random.RandomState(seed=1024)
        self.E = LIF(size=size_E, gl=gl, tau=tau_E, vth=V_threshold, vreset=V_reset, tau_ref=0)
        self.Ip = LIF(size=size_Ip, gl=gl, tau=tau_I, vth=V_threshold, vreset=V_reset, tau_ref=0)
        self.Id = LIF(size=size_Id, gl=gl, tau=tau_I, vth=V_threshold, vreset=V_reset, tau_ref=0)

        super().__init__()

    def dist(self, d):
        d = bm.remainder(d, 2 * bm.pi)
        d = bm.where(d > bm.pi, d - 2 * bm.pi, d)
        return d

    def make_gauss_conn(self, size_pre, size_post, prob):
        x_left = bm.reshape(bm.linspace(-bm.pi, bm.pi, size_pre), (-1, 1))
        x_right = bm.reshape(bm.linspace(-bm.pi, bm.pi, size_post), (1, -1))
        d = self.dist(x_left - x_right)
        w = self.J * bm.exp(-bm.square(d) / self.conn_a)
        prob_mask = (self.rng.rand(size_pre, size_post) < prob).astype(bm.float32)
        w = w * prob_mask

        return w

    def make_rand_conn(self, size_pre, size_post, prob):
        const_conn = lambda w, p: (w > 1 - p)
        w = const_conn(bm.random.rand(size_pre, size_post), prob).astype(bm.float32)
        return w

    def init_conn(self):
        seed = 1024
        prob = self.prob
        g_func = lambda size_pre, size_post, p: self.make_gauss_conn(size_pre, size_post, p)
        E2E_sw = self.gEE * g_func(self.size_E, self.size_E, 1.0)

        # ======== EI balance =====
        self.E2E_f = UnitExpCUBA(pre=self.E, post=self.E, conn=bp.connect.FixedProb(prob, seed=seed), tau=self.tau_Ef,
                                 g_max=self.JEE)
        self.E2I_f = UnitExpCUBA(pre=self.E, post=self.Id, conn=bp.connect.FixedProb(prob, seed=seed), tau=self.tau_Ef,
                                 g_max=self.JEI)
        self.I2I_f = UnitExpCUBA(pre=self.Id, post=self.Id, conn=bp.connect.FixedProb(prob, seed=seed), tau=self.tau_If,
                                 g_max=self.JII)
        self.I2E_f = UnitExpCUBA(pre=self.Id, post=self.E, conn=bp.connect.FixedProb(prob, seed=seed), tau=self.tau_If,
                                 g_max=self.JIE)

        # ======= CANN =====
        self.E2E_s = UnitExpCUBA(pre=self.E, post=self.E, conn=bp.connect.All2All(), tau=self.tau_Es, g_max=E2E_sw)
        self.E2I_s = UnitExpCUBA(pre=self.E, post=self.Ip, conn=bp.connect.FixedProb(prob, seed=seed), tau=self.tau_Es,
                                 g_max=self.gEIp)
        self.I2I_s = UnitExpCUBA(pre=self.Ip, post=self.Ip, conn=bp.connect.FixedProb(prob, seed=seed), tau=self.tau_Is,
                                 g_max=self.gIpIp)
        self.I2E_s = UnitExpCUBA(pre=self.Ip, post=self.E, conn=bp.connect.FixedProb(prob, seed=seed), tau=self.tau_Is,
                                 g_max=self.gIpE)
        self.ESI = Shunting(E2Esyn_s=self.E2E_s, E2Esyn_f=self.E2E_f, I2Esyn_s=self.I2E_s, k=self.shunting_k,
                            EGroup=self.E)

    def update(self, tdi, inputs):
        E_inp, Ip_inp, Id_inp = inputs
        self.E.ext_input[:] = E_inp
        self.Ip.ext_input[:] = Ip_inp
        self.Id.ext_input[:] = Id_inp
        super().update(tdi, inputs)

    def reset_state(self, batch_size=None):
        self.rng.seed(1024)
        super().reset_state(batch_size=batch_size)


class CosineWeightEIContinousAttractor(EIContinousAttractor):
    def __init__(self, size_E, size_Ip, size_Id, tau_E, tau_I, tau_Es, tau_Is, tau_Ef, tau_If, prob,
                 V_reset, V_threshold, JEE, JEI, JII, JIE, gl, gEE, gEIp, gIpIp, gIpE, shunting_k,
                 weight_scale_cos_freq=0.5 / bm.pi,
                 weight_scale_cos_phase=0.,
                 weight_scale_cos_range=(0.8, 1.2),
                 ):
        self.weight_scale_cos_freq = weight_scale_cos_freq
        self.weight_scale_cos_phase = weight_scale_cos_phase
        self.weight_scale_cos_range = weight_scale_cos_range

        super().__init__(
            size_E=size_E,
            size_Ip=size_Ip,
            size_Id=size_Id,
            tau_E=tau_E,
            tau_I=tau_I,
            tau_Es=tau_Es,
            tau_Is=tau_Is,
            tau_Ef=tau_Ef,
            tau_If=tau_If,
            V_reset=V_reset,
            V_threshold=V_threshold,
            prob=prob,
            JEE=JEE,
            JEI=JEI,
            JII=JII,
            JIE=JIE,
            gl=gl,
            gEE=gEE,
            gEIp=gEIp,
            gIpIp=gIpIp,
            gIpE=gIpE,
            shunting_k=shunting_k
        )

    def make_gauss_conn(self, size_pre, size_post, prob):
        x_left = bm.reshape(bm.linspace(-bm.pi, bm.pi, size_pre), (-1, 1))
        x_right = bm.reshape(bm.linspace(-bm.pi, bm.pi, size_post), (1, -1))
        w_freq = self.weight_scale_cos_freq
        w_phas = self.weight_scale_cos_phase
        wl, wu = self.weight_scale_cos_range

        # transform w_scale to weight_scale_cos_range
        w_scale = bm.cos(2*bm.pi * w_freq * x_right + w_phas)
        w_scale = (w_scale + 1.)/2 * (wu-wl) + wl
        d = self.dist(x_left - x_right)
        w = self.J * bm.exp(-bm.square(d) / self.conn_a)
        ws = w * w_scale
        prob_mask = (self.rng.rand(size_pre, size_post) < prob).astype(bm.float32)
        ws = ws * prob_mask

        return ws


class RandWeightEIContinousAttractor(EIContinousAttractor):
    def __init__(self, size_E, size_Ip, size_Id, tau_E, tau_I, tau_Es, tau_Is, tau_Ef, tau_If, prob,
                 V_reset, V_threshold, JEE, JEI, JII, JIE, gl, gEE, gEIp, gIpIp, gIpE, shunting_k,
                 weight_scale_range=(0.8, 1.2),
                 a_scale_range=(0.8, 1.2),
                 ):
        self.weight_scale_range = weight_scale_range
        self.a_scale_range = a_scale_range

        super().__init__(
            size_E=size_E,
            size_Ip=size_Ip,
            size_Id=size_Id,
            tau_E=tau_E,
            tau_I=tau_I,
            tau_Es=tau_Es,
            tau_Is=tau_Is,
            tau_Ef=tau_Ef,
            tau_If=tau_If,
            V_reset=V_reset,
            V_threshold=V_threshold,
            prob=prob,
            JEE=JEE,
            JEI=JEI,
            JII=JII,
            JIE=JIE,
            gl=gl,
            gEE=gEE,
            gEIp=gEIp,
            gIpIp=gIpIp,
            gIpE=gIpE,
            shunting_k=shunting_k
        )

    def make_gauss_conn(self, size_pre, size_post, prob):
        x_left = bm.reshape(bm.linspace(-bm.pi, bm.pi, size_pre), (-1, 1))
        x_right = bm.reshape(bm.linspace(-bm.pi, bm.pi, size_post), (1, -1))
        wl, wu = self.weight_scale_range
        al, au = self.a_scale_range

        # transform w_scale to weight_scale_cos_range
        w_scale = bm.random.rand(*x_right.shape) * (wu-wl) + wl
        a_scale = bm.random.rand(*x_right.shape) * (au - al) + al
        d = self.dist(x_left - x_right)
        wa = self.J * bm.exp(-bm.square(d) / (self.conn_a * a_scale))
        ws = wa * w_scale
        prob_mask = (self.rng.rand(size_pre, size_post) < prob).astype(bm.float32)
        ws = ws * prob_mask

        return ws