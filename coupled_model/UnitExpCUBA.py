from brainpy.synapses import Exponential
import brainpy.math as bm


class UnitExpCUBA(Exponential):
    def __init__(self, pre, post, conn, tau, g_max, **kwargs):
        super().__init__(pre, post, conn, tau=tau, g_max=g_max/tau, **kwargs)
        # self.g_max = self.g_max / tau
        self.output_value = bm.Variable(bm.zeros(self.post.num))

    def update(self, tdi, pre_spike=None):
        super().update(tdi, pre_spike=pre_spike)
        self.output_value.value = self.g
        return self.output_value
