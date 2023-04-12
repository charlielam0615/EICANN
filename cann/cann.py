import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
from synapse import UnitExpCUBA, Shunting


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

        # for debug purposes
        # self._leak = bm.Variable(bm.zeros(self.size))
        # self._recinp = bm.Variable(bm.zeros(self.size))
        # self._ext = bm.Variable(bm.zeros(self.size))

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
        V = bm.where(V < -5, self.V, V)
        spike = self.vth <= V
        self.spike.value = spike
        self.t_last_spike.value = bm.where(spike, _t, self.t_last_spike)
        self.V.value = bm.where(spike, self.vreset, V)
        self.refractory.value = bm.logical_or(refractory, spike)
        
        # for debug purposes
        # self._leak.value = self.gl * self.V
        # self._recinp.value = self.input
        # self._ext.value = self.ext_input


        self.input[:] = 0.



class CANN(bp.dyn.Network):
    def __init__(self, config):
        self.conn_a = 2 * config.conn_a**2
        self.stim_a = 2 * config.stim_a**2
        self.size_E, self.size_Ip = config.size_E, config.size_Ip
        self.shunting_k = config.shunting_k
        self.J = 1.
        self.A = 1.
        self.name = "CANN"

        w = lambda size_pre, size_post, p: self.make_gauss_conn(size_pre, size_post, p)
        r = lambda size_pre, size_post, p: self.make_rand_conn(size_pre, size_post, p)

        # neurons
        self.E = LIF(config.size_E, tau=config.tau_E, gl=config.gl, vth=config.V_threshold, 
                     vreset=config.V_reset, tau_ref=5, name='E_Group')
        self.Ip = LIF(config.size_Ip, tau=config.tau_I, gl=config.gl, vth=config.V_threshold, 
                      vreset=config.V_reset, tau_ref=5, name='Ip_Group')

        # CANN synapse
        E2E_sw, E2I_sw, I2I_sw, I2E_sw = config.gEE*self.make_gauss_conn(config.size_E, config.size_E, 1.0), \
                                         config.gEIp*self.make_rand_conn(config.size_E, config.size_Ip, config.prob), \
                                         config.gIpIp*self.make_rand_conn(config.size_Ip, config.size_Ip, config.prob), \
                                         config.gIpE*self.make_rand_conn(config.size_Ip, config.size_E, config.prob)

        self.E2E_s = UnitExpCUBA(pre=self.E, post=self.E, conn=bp.connect.All2All(), tau=config.tau_Es, g_max=E2E_sw)
        self.E2I_s = UnitExpCUBA(pre=self.E, post=self.Ip, conn=bp.conn.FixedProb(config.prob), tau=config.tau_Es, g_max=config.gEIp)
        self.I2I_s = UnitExpCUBA(pre=self.Ip, post=self.Ip, conn=bp.conn.FixedProb(config.prob), tau=config.tau_Is, g_max=config.gIpIp)
        self.I2E_s = UnitExpCUBA(pre=self.Ip, post=self.E, conn=bp.conn.FixedProb(config.prob), tau=config.tau_Is, g_max=config.gIpE)
        self.ESI = Shunting(E2Esyn_s=self.E2E_s, I2Esyn_s=self.I2E_s, k=config.shunting_k, EGroup=self.E)

        print('[Weights]')
        print("---------------- CANN -------------------")
        print("|  E2E   |  E2I    |  I2I     |  I2E    | ")
        print(f"|{E2E_sw.max():.5f} | {E2I_sw.max():.5f} | {I2I_sw.min():.5f}  | {I2E_sw.min():.5f}|")
        super(CANN, self).__init__()

    def dist(self, d):
        d = bm.remainder(d, 2 * bm.pi)
        d = bm.where(d > bm.pi, d - 2 * bm.pi, d)
        return d

    def make_gauss_conn(self, size_pre, size_post, prob):
        x_left = bm.reshape(bm.linspace(-bm.pi, bm.pi, size_pre), (-1, 1))
        x_right = bm.reshape(bm.linspace(-bm.pi, bm.pi, size_post), (1, -1))
        d = self.dist(x_left - x_right)
        w_ = self.J * bm.exp(-bm.square(d) / self.conn_a)
        # w = w_ / bm.sum(w_, axis=-1, keepdims=True)
        w = w_
        prob_mask = (bm.random.rand(size_pre, size_post)<prob).astype(bm.float32)
        w = w * prob_mask
        return w

    def make_rand_conn(self, size_pre, size_post, prob):
        const_conn = lambda w, p: (w > 1 - p)
        w = const_conn(bm.random.rand(size_pre, size_post), prob).astype(bm.float32)
        return w

