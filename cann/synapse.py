from brainpy.dyn import TwoEndConn
from brainpy.synapses import Exponential
import brainpy.math as bm


class UnitExpCUBA(Exponential):
    def __init__(self, pre, post, conn, tau, g_max, **kwargs):
        super().__init__(pre, post, conn, tau=tau, g_max=g_max/tau, **kwargs)
        self.output_value = bm.Variable(bm.zeros(self.post.num))

    def update(self, tdi, pre_spike=None):
        super().update(tdi, pre_spike=pre_spike)
        self.output_value.value = self.g
        self.post.input += self.g
        return 
    

class Shunting(TwoEndConn):
    def __init__(self, E2Esyn_s, I2Esyn_s, k, EGroup):
        super().__init__(pre=E2Esyn_s.pre, post=I2Esyn_s.post, conn=None)
        self.E2Esyn_s = E2Esyn_s
        self.I2Esyn_s = I2Esyn_s
        self.EGroup = EGroup
        self.k = k
        self.post = self.E2Esyn_s.post

    def update(self, tdi):
        t, dt = tdi.t, tdi.dt
        E_inp = self.E2Esyn_s.output_value + self.EGroup.ext_input
        I_inp = self.I2Esyn_s.output_value
        self.post.input += self.k * E_inp * I_inp
        return