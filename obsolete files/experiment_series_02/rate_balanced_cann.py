import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt

bp.math.set_platform('cpu')
global_dt = 0.01

# ======== CANN ========
tau_Es = 150
tau_Ef = 30
tau_If = 10
k = 0.05
a = bm.pi/3
J0 = 1
A = 1
theta = 1.0

ei_scale = 0.0
wee = 1.3
jie = -1.4 * ei_scale
jii = -0.7 * ei_scale
jee = 0.7 * ei_scale
jei = 0.8 * ei_scale


class EICANN(bp.dyn.NeuGroup):
    def __init__(self, size, **kwargs):
        super(EICANN, self).__init__(size=size, **kwargs)

        # parameters
        self.tau_Es = tau_Es    # The synaptic time constant
        self.tau_Ef = tau_Ef
        self.tau_If = tau_If
        self.k = k    # Degree of the rescaled inhibition
        self.a = a    # Half-width of the range of excitatory connections
        self.A = A    # Magnitude of the external input
        self.J0 = J0    # maximum connection value
        self.theta = theta

        # variables
        self.U_E = bm.Variable(bm.zeros(size))
        self.V_E = bm.Variable(bm.zeros(size))
        self.V_I = bm.Variable(bm.zeros(size))
        self.U_I = bm.Variable(bm.zeros(size))
        self.input = bm.Variable(bm.zeros(size))

        # currents for book-keeping
        self.Irec_EEs = bm.Variable(bm.zeros(size))
        self.Irec_EEf = bm.Variable(bm.zeros(size))
        self.Irec_IEf = bm.Variable(bm.zeros(size))
        self.Irec_EIf = bm.Variable(bm.zeros(size))
        self.Irec_IIf = bm.Variable(bm.zeros(size))

        # rates for book-keeping
        self.r_E = bm.Variable(bm.zeros(size))
        self.r_I = bm.Variable(bm.zeros(size))

        # The connection matrix
        self.conn_mat = self.make_gauss_conn(size, size, 1.0)

        # function
        self.integral = bp.odeint(f=self.derivative, method='euler')

    def derivative(self, U_E, U_I, V_E, V_I, t, Iext):
        r1 = bm.square(U_E)
        r2 = 1.0 + self.k * bm.sum(r1)
        self.r_E.value = r1 / r2
        self.r_I.value = U_I / self.theta
        self.Irec_EEs.value = wee * bm.dot(self.conn_mat, self.r_E)
        self.Irec_EEf.value = jee * bm.dot(self.conn_mat, self.r_E)
        self.Irec_IEf.value = jie * bm.dot(self.conn_mat, self.r_I)
        self.Irec_EIf.value = jei * bm.dot(self.conn_mat, self.r_E)
        self.Irec_IIf.value = jii * bm.dot(self.conn_mat, self.r_I)
        dU_E = (-U_E + self.Irec_EEs + V_E + V_I + Iext) / self.tau_Es
        dU_I = (-U_I + self.Irec_EIf + self.Irec_IIf) / self.tau_If
        dV_E = (-V_E + self.Irec_EEf) / self.tau_Ef
        dV_I = (-V_I + self.Irec_IEf) / self.tau_If

        return dU_E, dU_I, dV_E, dV_I

    def dist(self, d):
        d = bm.remainder(d, 2 * bm.pi)
        d = bm.where(d > bm.pi, d - 2 * bm.pi, d)
        return d

    def make_gauss_conn(self, size_pre, size_post, prob):
        x_left = bm.reshape(bm.linspace(-bm.pi, bm.pi, size_pre), (-1, 1))
        x_right = bm.reshape(bm.linspace(-bm.pi, bm.pi, size_post), (1, -1))
        d = self.dist(x_left - x_right)
        w = self.J0 * bm.exp(-bm.pi * bm.square(d / self.a))
        prob_mask = (bm.random.rand(size_pre, size_post) < prob).astype(bm.float32)
        w = w * prob_mask
        return w

    def get_stimulus_by_pos(self, pos, size_n):
        x = bm.linspace(-bm.pi, bm.pi, size_n)
        if bm.ndim(pos) == 2:
            x = x[None,]
        I = self.A * bm.exp(-bm.pi * bm.square(self.dist(x - pos) / 2*self.a))
        return I

    def update(self, tdi):
        U_E, U_I, V_E, V_I = self.integral(self.U_E, self.U_I, self.V_E, self.V_I, tdi.t, self.input, tdi.dt)
        U_E = bm.where(U_E<0, self.U_E, U_E)
        U_I = bm.where(U_I<0, self.U_I, U_I)
        V_E = bm.where(V_E<0, self.V_E, V_E)
        V_I = bm.where(V_I>0, self.V_I, V_I)
        self.U_E.value, self.U_I.value, self.V_E.value, self.V_I.value = U_E, U_I, V_E, V_I
        self.input[:] = 0.

#-U_E + self.Irec_EEs + V_E + V_I + self.input

n_size = 512
eicann = EICANN(size=n_size)

I1 = eicann.get_stimulus_by_pos(0., n_size)
Iext, duration = bp.inputs.section_input(values=[0., I1, 0.],
                                         durations=[100., 2450., 850.],
                                         dt=global_dt,
                                         return_length=True)
runner = bp.dyn.DSRunner(eicann,
                         inputs=['input', Iext, 'iter'],
                         monitors=['U_E', 'U_I', 'V_E', 'V_I', 'Irec_EEs', 'Irec_EEf', 'Irec_IEf', 'Irec_EIf', 'Irec_IIf', 'r_E', 'r_I'],
                         jit=True,
                         dt=global_dt)
runner.run(duration)
x = bm.linspace(-bm.pi, bm.pi, n_size)
t_axis = bm.arange(0, duration, global_dt)
ma = runner.mon.r_E
bump_activity = bm.vstack([bm.sum(ma * bm.cos(x[None,]), axis=1), bm.sum(ma * bm.sin(x[None,]), axis=1)])
readout = bm.array([[1., 0.]]) @ bump_activity
nm_readout = readout.T / bm.mean(readout[:, 10000:50000])

fig = plt.figure()
ax1 = fig.add_subplot(4, 1, 1)
ax1.plot(t_axis, bm.max(Iext, axis=1), label='Iext')
ax1.plot(t_axis, nm_readout, label='readout')
ax1.legend()

ax2 = fig.add_subplot(4, 1, 2)
neuron_index = 256
E_current = runner.mon.Irec_EEs[:, neuron_index] + runner.mon.V_E[:, neuron_index] + Iext[:, neuron_index]
I_current = -runner.mon.U_E[:, neuron_index] + runner.mon.V_I[:, neuron_index]
total_current = E_current + I_current
ax2.plot(t_axis, E_current, label='E_current')
ax2.plot(t_axis, I_current, label='I_current')
ax2.plot(t_axis, total_current, label='total_current')
ax2.legend(loc='upper left')

ax3 = fig.add_subplot(4, 1, 3)
neuron_index = 256
V_E = runner.mon.V_E[:, neuron_index]
V_I = runner.mon.V_I[:, neuron_index]
ax3.plot(t_axis, V_E, label='V_E', alpha=0.5)
ax3.plot(t_axis, V_I, label='V_I', alpha=0.5)
ax3.legend(loc='upper left')

ax4 = fig.add_subplot(4, 1, 4)
im = ax4.imshow(ma.T, aspect='auto')
fig.colorbar(im, ax=ax4)

plt.show()