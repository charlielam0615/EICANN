from brainpy.dyn import TwoEndConn
import brainpy.math as bm
from brainpy.integrators import odeint


class FFTUnitExpCUBA(TwoEndConn):
    def __init__(self, pre, post, conn, g_max, tau):
        super().__init__(pre=pre, post=post, conn=conn)
        self.g_max = g_max
        self.tau = tau
        self.fft_gmax = bm.fft.fft2(g_max/tau)  # Gaussian profile of the first neuron
        self.g = bm.Variable(bm.zeros(self.post.num))
        self.output_value = bm.Variable(bm.zeros(self.post.num))
        self.integral = odeint(lambda g, t: -g / self.tau, method='exp_auto')

    def update(self, tdi):
        t, dt = tdi.t, tdi.dt
        pre_spike = self.pre.spike
        pre_spike = pre_spike.astype(bm.float_).reshape(self.g_max.shape)
        fft_pre_spike = bm.fft.fft2(pre_spike)
        post_vs = bm.real(bm.fft.ifft2(fft_pre_spike * self.fft_gmax))
        post_vs = post_vs.reshape(self.pre.spike.shape)
        self.g.value = self.integral(self.g.value, t, dt=dt) + post_vs
        self.output_value.value = self.g
        self.post.input += self.output_value
