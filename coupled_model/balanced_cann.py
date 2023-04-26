import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
import numpy as np
from synapse import UnitExpCUBA, Shunting
from functools import partial


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


class EICANN(bp.dyn.Network):
    def __init__(self, config):
        self.conn_a = 2 * config.conn_a**2
        self.stim_a = 2 * config.stim_a**2
        self.size_E, self.size_Ip, self.size_Id = config.size_E, config.size_Ip, config.size_Id
        self.shunting_k =  config.shunting_k
        self.J = 1.
        self.A = 1.
        self.name = 'EICANN'

        _plot_weight_dist = False

        # neurons
        self.E = LIF(size=config.size_E, gl=config.gl, tau=config.tau_E, vth=config.V_threshold, 
                     vreset=config.V_reset, tau_ref=config.tau_ref)
        self.Ip = LIF(size=config.size_Ip, gl=config.gl, tau=config.tau_I, vth=config.V_threshold, 
                      vreset=config.V_reset, tau_ref=config.tau_ref)
        self.Id = LIF(size=config.size_Id, gl=config.gl, tau=config.tau_I, vth=config.V_threshold, 
                      vreset=config.V_reset, tau_ref=config.tau_ref)


        # ======== EI balance =====
        if not _plot_weight_dist:
            # E/I balance weights, not used. Merely used for printing weight information.
            E2E_fw, E2I_fw, I2I_fw, I2E_fw = config.JEE*self.make_rand_conn(config.size_E, config.size_E, config.prob), \
                                            config.JEI*self.make_rand_conn(config.size_E, config.size_Id, config.prob), \
                                            config.JII*self.make_rand_conn(config.size_Id, config.size_Id, config.prob), \
                                            config.JIE*self.make_rand_conn(config.size_Id, config.size_E, config.prob)
            # weights from delta distribution
            self.E2E_f = UnitExpCUBA(pre=self.E, post=self.E, conn=bp.conn.FixedProb(config.prob), 
                                     tau=config.tau_Ef, g_max=config.JEE)
            self.E2I_f = UnitExpCUBA(pre=self.E, post=self.Id, conn=bp.conn.FixedProb(config.prob), 
                                     tau=config.tau_Ef, g_max=config.JEI)
            self.I2I_f = UnitExpCUBA(pre=self.Id, post=self.Id, conn=bp.conn.FixedProb(config.prob), 
                                     tau=config.tau_If, g_max=config.JII)
            self.I2E_f = UnitExpCUBA(pre=self.Id, post=self.E, conn=bp.conn.FixedProb(config.prob), 
                                     tau=config.tau_If, g_max=config.JIE)
        else:
            rv = lambda size_pre, size_post, p, mean: \
                self.make_rand_conn_with_variance(size_pre, size_post, p, mean, 0.1)
            E2E_fw, E2I_fw, I2I_fw, I2E_fw = rv(config.size_E, config.size_E, config.prob, config.JEE), \
                                            rv(config.size_E, config.size_Id, config.prob, config.JEI), \
                                            rv(config.size_Id, config.size_Id, config.prob, config.JII), \
                                            rv(config.size_Id, config.size_E, config.prob, config.JIE)

            # weights from gaussian distribution
            g_from_dist = lambda m: bp.init.Normal(mean=m, scale=0.1*bm.abs(m))
            self.E2E_f = UnitExpCUBA(pre=self.E, post=self.E, conn=bp.conn.FixedProb(config.prob), 
                                        tau=config.tau_Ef, g_max=g_from_dist(config.JEE))
            self.E2I_f = UnitExpCUBA(pre=self.E, post=self.Id, conn=bp.conn.FixedProb(config.prob), 
                                        tau=config.tau_Ef, g_max=g_from_dist(config.JEI))
            self.I2I_f = UnitExpCUBA(pre=self.Id, post=self.Id, conn=bp.conn.FixedProb(config.prob), 
                                        tau=config.tau_If, g_max=g_from_dist(config.JII))
            self.I2E_f = UnitExpCUBA(pre=self.Id, post=self.E, conn=bp.conn.FixedProb(config.prob), 
                                        tau=config.tau_If, g_max=g_from_dist(config.JIE))
        
        # ======= CANN =====
        E2E_sw, E2I_sw, I2I_sw, I2E_sw = config.gEE * self.make_gauss_conn(config.size_E, config.size_E, 1.0), \
                                         config.gEIp * self.make_rand_conn(config.size_E, config.size_Ip, config.prob), \
                                         config.gIpIp * self.make_rand_conn(config.size_Ip, config.size_Ip, config.prob), \
                                         config.gIpE * self.make_rand_conn(config.size_Ip, config.size_E, config.prob)

        self.E2E_s = UnitExpCUBA(pre=self.E, post=self.E, conn=bp.connect.All2All(), 
                                 tau=config.tau_Es, g_max=E2E_sw)
        self.E2I_s = UnitExpCUBA(pre=self.E, post=self.Ip, conn=bp.conn.FixedProb(config.prob), 
                                 tau=config.tau_Es, g_max=config.gEIp)
        self.I2I_s = UnitExpCUBA(pre=self.Ip, post=self.Ip, conn=bp.conn.FixedProb(config.prob), 
                                 tau=config.tau_Is, g_max=config.gIpIp)
        self.I2E_s = UnitExpCUBA(pre=self.Ip, post=self.E, conn=bp.conn.FixedProb(config.prob), 
                                 tau=config.tau_Is, g_max=config.gIpE)
        self.ESI = Shunting(E2Esyn_s=self.E2E_s, E2Esyn_f=self.E2E_f, I2Esyn_s=self.I2E_s, 
                            k=config.shunting_k, EGroup=self.E)

        super(EICANN, self).__init__()

        print('[Weights]')
        print("------------- EI Balanced ---------------")
        print("|  E2E   |  E2I    |  I2I     |  I2E    | ")
        print(f"|{E2E_fw.max():.5f} | {E2I_fw.max():.5f} | {I2I_fw.min():.5f} | {I2E_fw.min():.5f}|")
        print("---------------- CANN -------------------")
        print("|  E2E   |  E2I    |  I2I     |  I2E    | ")
        print(f"|{E2E_sw.max():.5f} | {E2I_sw.max():.5f} | {I2I_sw.min():.5f}  | {I2E_sw.min():.5f}|")
        super(EICANN, self).__init__()

        if _plot_weight_dist:
            self._plot_weight_distribution(ei_weight=E2E_fw, cann_weight=E2E_sw)
        return

    @staticmethod
    def _plot_weight_distribution(ei_weight, cann_weight):
        import seaborn as sns
        ei_w = ei_weight[ei_weight > 1e-5]
        cann_w = cann_weight[cann_weight > 1e-5]

        print(f"E/I : CANN = {bm.mean(ei_w)/bm.mean(cann_w):.4f}")

        sns.distplot(ei_w, hist=False, kde=True, label='E-INN weights',
                     hist_kws={'edgecolor': 'black'},
                     kde_kws={'fill': True, 'linewidth': 2})
        sns.distplot(cann_w, hist=False, kde=True, label='CANN weights',
                     hist_kws={'edgecolor': 'black'},
                     kde_kws={'fill': True, 'linewidth': 2})
        plt.legend(loc="upper right", fontsize=12)
        plt.show()

    @staticmethod
    def _plot_weight_distribution_broken_axis(ei_weight, cann_weight):
        import seaborn as sns
        f, (ax_top, ax_bottom) = plt.subplots(2, 1, sharex=True, gridspec_kw={'hspace': 0.1})
        ei_w = ei_weight[ei_weight > 1e-5]
        cann_w = cann_weight[cann_weight > 1e-5]

        print(f"E/I : CANN = {bm.mean(ei_w)/bm.mean(cann_w):.4f}")

        ax = sns.kdeplot(data=ei_w, label='E/I weights', fill=True, linewidth=2, bw_method=0.02, ax=ax_top,
                         color='gray')
        ax = sns.kdeplot(data=ei_w, label='E/I weights', fill=True, linewidth=2, bw_method=0.02, ax=ax_bottom,
                         color='gray')
        ax = sns.kdeplot(data=cann_w, label='CANN weights', fill=True, linewidth=2, bw_method=0.02, ax=ax_bottom,
                         color='blue')
        ax = sns.kdeplot(data=cann_w, label='CANN weights', fill=True, linewidth=2, bw_method=0.02, ax=ax_top,
                         color='blue')

        ax_bottom.set_ylim([0, 150])
        ax_top.set_ylim(bottom=450)

        sns.despine(ax=ax_top, top=False, bottom=True, left=False, right=False)
        sns.despine(ax=ax_bottom, top=True, bottom=False, left=False, right=False)
        ax_top.set_xticklabels([])

        ax = ax_top
        d = .015
        kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
        ax.plot((-d, +d), (-d, +d), **kwargs)

        ax2 = ax_bottom
        kwargs.update(transform=ax2.transAxes)
        ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)

        ax_top.legend()
        ax_top.grid()
        ax_bottom.grid()
        ax_bottom.set_ylabel("")

        plt.show()

    def dist(self, d):
        d = bm.remainder(d, 2 * bm.pi)
        d = bm.where(d > bm.pi, d - 2 * bm.pi, d)
        return d

    def make_rand_conn_with_variance(self, size_pre, size_post, prob, mean, v_ratio):
        const_conn = lambda w, p: (w > 1 - p)
        w = bm.random.randn(size_pre, size_post) * v_ratio * bm.abs(mean) + mean
        p_mask = const_conn(bm.random.rand(size_pre, size_post), prob).astype(bm.float32)
        w = w * p_mask
        return w

    def get_stimulus_by_pos(self, pos, size_n):
        x = bm.linspace(-bm.pi, bm.pi, size_n)
        if bm.ndim(pos) == 2:
            x = x[None,]
        I = self.A * bm.exp(-bm.pi * bm.square(self.dist(x - pos) / (2 * self.a)))
        return I

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
    