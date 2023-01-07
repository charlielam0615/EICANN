import brainpy as bp
import brainpy.math as bm
import numpy as np
import matplotlib.pyplot as plt
from UnitExpCUBA import UnitExpCUBA
from FFTUnitExpCUBA import FFTUnitExpCUBA
from brainpy.dyn import TwoEndConn


class Shunting(TwoEndConn):
    def __init__(self, E2Esyn_s, I2Esyn_s, k, EGroup):
        super().__init__(pre=E2Esyn_s.pre, post=I2Esyn_s.post, conn=None)
        self.E2Esyn_s = E2Esyn_s
        self.I2Esyn_s = I2Esyn_s
        self.EGroup = EGroup
        self.k = k
        self.post = self.E2Esyn_s.post
        self.output_value = bm.Variable(bm.zeros(self.post.num))

    def update(self, tdi):
        t, dt = tdi.t, tdi.dt
        E_inp = self.E2Esyn_s.output_value + bm.reshape(self.EGroup.ext_input,-1)
        I_inp = self.I2Esyn_s.output_value
        self.output_value.value = self.k * E_inp * I_inp
        self.post.input += self.output_value


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


class CANN2D(bp.dyn.Network):
    def __init__(self, size_E, size_Ip, tau_E, tau_I, tau_Es, tau_Is, V_reset, V_threshold, prob,
                 gl, gEE, gEIp, gIpIp, gIpE, shunting_k, a):
        # consider isotropic guassian
        self.conn_a = 2 * a**2
        self.stim_a = 2 * a**2
        self.size_E, self.size_Ip = size_E, size_Ip
        self.shunting_k = shunting_k
        self.J = 1.
        self.A = 1.

        w = lambda size: self.make_guass_kernel_2d(size)

        # neurons
        self.E = LIF2D(size_E, tau=tau_E, gl=gl, V_th=V_threshold, V_reset=V_reset, tau_ref=0)
        self.Ip = LIF2D(size_Ip, tau=tau_I, gl=gl, V_th=V_threshold, V_reset=V_reset, tau_ref=0)
        E, Ip = self.E, self.Ip

        # CANN synapse
        r = lambda size_pre, size_post, p: self.make_rand_conn_2d(size_pre, size_post, p)
        E2E_skernel = gEE*w(size_E)
        E2I_sw, I2I_sw, I2E_sw = gEIp*r(size_E, size_Ip, prob), \
                                 gIpIp*r(size_Ip, size_Ip, prob), \
                                 gIpE*r(size_Ip, size_E, prob)

        self.E2E_s = FFTUnitExpCUBA(pre=E, post=E, conn=bp.connect.All2All(), tau=tau_Es, g_max=E2E_skernel)
        self.E2I_s = UnitExpCUBA(pre=E, post=Ip, conn=bp.conn.FixedProb(prob), tau=tau_Es, g_max=gEIp)
        self.I2I_s = UnitExpCUBA(pre=Ip, post=Ip, conn=bp.conn.FixedProb(prob), tau=tau_Is, g_max=gIpIp)
        self.I2E_s = UnitExpCUBA(pre=Ip, post=E, conn=bp.conn.FixedProb(prob), tau=tau_Is, g_max=gIpE)
        self.ESI = Shunting(E2Esyn_s=self.E2E_s, I2Esyn_s=self.I2E_s, k=shunting_k, EGroup=self.E)

        print('[Weights]')
        print("---------------- CANN -------------------")
        print("|  E2E   |  E2I    |  I2I     |  I2E    | ")
        print(f"|{E2E_skernel.max():.5f} | {E2I_sw.max():.5f} | {I2I_sw.min():.5f}  | {I2E_sw.min():.5f}|")
        super(CANN2D, self).__init__()

    def dist(self, d):
        d = bm.remainder(d, 2 * bm.pi)
        d = bm.where(d > bm.pi, d - 2 * bm.pi, d)
        return d

    def make_gauss_conn_2d(self, size_pre, size_post):
        x_left = bm.reshape(bm.linspace(-bm.pi, bm.pi, size_pre[0]), (-1, 1))
        x_right = bm.reshape(bm.linspace(-bm.pi, bm.pi, size_post[0]), (1, -1))
        y_left = bm.reshape(bm.linspace(-bm.pi, bm.pi, size_pre[1]), (-1, 1))
        y_right = bm.reshape(bm.linspace(-bm.pi, bm.pi, size_post[1]), (1, -1))
        d_square = (self.dist(x_left-x_right)**2)[:, None, :, None] + (self.dist(y_left-y_right)**2)[None, :, None, :]
        w = self.J * bm.exp(-d_square/self.conn_a)
        w = bm.reshape(w, [np.prod(size_pre), np.prod(size_post)])
        return w

    def make_guass_kernel_2d(self, size):
        x_left  = bm.linspace(-bm.pi, bm.pi, size[0])
        x_right = bm.linspace(-bm.pi, bm.pi, size[1])
        x1, x2 = bm.meshgrid(x_left, x_right)
        value = bm.stack([x1.flatten(), x2.flatten()]).T
        d = self.dist(bm.abs(value[0] - value))
        d = bm.linalg.norm(d, axis=1)
        d = d.reshape(size)
        w = self.J * bm.exp(-bm.square(d)/self.conn_a)
        return w

    def make_rand_conn_2d(self, size_pre, size_post, prob):
        const_conn = lambda w, p: (w > 1 - p)
        w = const_conn(bm.random.rand(size_pre[0], size_pre[1], size_post[0], size_post[1]), prob).astype(bm.float32)
        w = bm.reshape(w, [np.prod(size_pre), np.prod(size_post)])
        return w
