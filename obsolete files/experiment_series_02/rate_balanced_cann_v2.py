import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt

k = 1.0
ei_scale = 40.0 # 40


bp.math.set_platform('cpu')
global_dt = 0.01

# ======== CANN ========
tau_Es = 150
tau_Ef = 30
tau_If = 10
a = bm.pi/3
J0 = 1
A = 1
theta = 1.0
wee = 0.15

# ======== EI balance ========
jie = -1.4 * ei_scale
jii = -0.7 * ei_scale
jee = 0.7 * ei_scale
jei = 0.8 * ei_scale
gl = 0.1


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
        self.fftgaus_conn_mat = self.make_fft_gauss_conn(size)
        self.rand_conn_mat_eef = self.make_rand_conn(size, size, 0.5)
        self.rand_conn_mat_ief = self.make_rand_conn(size, size, 0.5)
        self.rand_conn_mat_eif = self.make_rand_conn(size, size, 0.5)
        self.rand_conn_mat_iif = self.make_rand_conn(size, size, 0.5)

        # function
        self.integral = bp.odeint(f=self.derivative, method='euler')

    def derivative(self, U_E, U_I, V_E, V_I, t, Iext):
        r1 = bm.square(bm.maximum(U_E, 0.0))
        r2 = 1.0 + self.k * bm.sum(r1)
        self.r_E.value = r1 / r2
        self.r_I.value = U_I / self.theta
        self.Irec_EEs.value = wee * bm.real(bm.fft.ifft(bm.fft.fft(self.fftgaus_conn_mat)*bm.fft.fft(self.r_E)))
        self.Irec_EEf.value = jee * bm.dot(self.rand_conn_mat_eef, self.r_E)
        self.Irec_IEf.value = jie * bm.dot(self.rand_conn_mat_ief, self.r_I)
        self.Irec_EIf.value = jei * bm.dot(self.rand_conn_mat_eif, self.r_E)
        self.Irec_IIf.value = jii * bm.dot(self.rand_conn_mat_iif, self.r_I)
        dU_E = (-gl*U_E + self.Irec_EEs + V_E + V_I + Iext) / self.tau_Es
        dU_I = (-U_I + self.Irec_EIf + self.Irec_IIf) / self.tau_If
        dV_E = (-V_E + self.Irec_EEf) / self.tau_Ef
        dV_I = (-V_I + self.Irec_IEf) / self.tau_If
        return dU_E, dU_I, dV_E, dV_I

    def dist(self, d):
        d = bm.remainder(d, 2 * bm.pi)
        d = bm.where(d > bm.pi, d - 2 * bm.pi, d)
        return d

    def make_rand_conn(self, size_pre, size_post, prob):
        const_conn = lambda w, p: (w > 1 - p) / bm.maximum(bm.sum(w > 1 - p, axis=-1, keepdims=True), 1)
        w = const_conn(bm.random.rand(size_pre, size_post), prob)
        return w

    def make_gauss_conn(self, size_pre, size_post, prob):
        x_left = bm.reshape(bm.linspace(-bm.pi, bm.pi, size_pre), (-1, 1))
        x_right = bm.reshape(bm.linspace(-bm.pi, bm.pi, size_post), (1, -1))
        d = self.dist(x_left - x_right)
        w = self.J0 * bm.exp(-bm.pi * bm.square(d / self.a))
        prob_mask = (bm.random.rand(size_pre, size_post) < prob).astype(bm.float32)
        w = w * prob_mask
        return w

    def make_fft_gauss_conn(self, size):
        x = bm.linspace(-bm.pi, bm.pi, size)
        x_ = bm.minimum(x+bm.pi, bm.pi-x)
        w = self.J0 * bm.exp(-bm.pi * bm.square(x_ / self.a))
        return w

    def get_stimulus_by_pos(self, pos, size_n):
        x = bm.linspace(-bm.pi, bm.pi, size_n)
        if bm.ndim(pos) == 2:
            x = x[None,]
        I = self.A * bm.exp(-bm.pi * bm.square(self.dist(x - pos) / (2*self.a)))
        return I

    def update(self, tdi):
        U_E, U_I, V_E, V_I = self.integral(self.U_E, self.U_I, self.V_E, self.V_I, tdi.t, self.input, tdi.dt)
        U_E = bm.where(U_E<0, self.U_E, U_E)
        U_I = bm.where(U_I<0, self.U_I, U_I)
        V_E = bm.where(V_E<0, self.V_E, V_E)
        V_I = bm.where(V_I>0, self.V_I, V_I)
        self.U_E.value, self.U_I.value, self.V_E.value, self.V_I.value = U_E, U_I, V_E, V_I
        self.input[:] = 0.

n_size = 512
eicann = EICANN(size=n_size)

# ======== Persistent Activity =======
I1 = eicann.get_stimulus_by_pos(0., n_size)
Iext, duration = bp.inputs.section_input(values=[0.1, I1*0.9+0.1, 0.1],
                                         durations=[800., 1000., 800.],
                                         dt=global_dt,
                                         return_length=True)

# =========== Moving Inputs ========
# duration = 2000
# n_step = int(duration / global_dt)
# pos = bm.linspace(-1*bm.pi/2, 1*bm.pi/2, n_step)[:,None]
# Iext = eicann.get_stimulus_by_pos(pos, n_size)


runner = bp.dyn.DSRunner(eicann,
                         inputs=['input', Iext, 'iter'],
                         monitors=['U_E', 'U_I', 'V_E', 'V_I', 'Irec_EEs', 'Irec_EEf', 'Irec_IEf', 'Irec_EIf', 'Irec_IIf', 'r_E', 'r_I'],
                         jit=True,
                         dt=global_dt)
runner.run(duration)

fig = plt.figure()
x = bm.linspace(-bm.pi, bm.pi, n_size)
t_axis = bm.arange(0, duration, global_dt)
ma = runner.mon.r_E
bump_activity = bm.vstack([bm.sum(ma * bm.cos(x[None,]), axis=1), bm.sum(ma * bm.sin(x[None,]), axis=1)])
readout = bm.array([[1., 0.]]) @ bump_activity
nm_readout = readout.T / bm.mean(readout[:, 80000:180000])

ax1 = fig.add_subplot(4, 1, 1)
ax1.plot(t_axis, bm.max(Iext, axis=1), label='Iext')
ax1.plot(t_axis, nm_readout, label='readout')
ax1.legend()

ax2 = fig.add_subplot(4, 1, 2)
neuron_index = 256
E_current = runner.mon.Irec_EEs[:, neuron_index] + runner.mon.V_E[:, neuron_index] + Iext[:, neuron_index]
I_current = -gl*runner.mon.U_E[:, neuron_index] + runner.mon.V_I[:, neuron_index]
total_current = E_current + I_current
ax2.plot(t_axis, E_current, label='E_current')
ax2.plot(t_axis, I_current, label='I_current')
ax2.plot(t_axis, total_current, label='total_current')
ax2.legend(loc='upper right')

ax3 = fig.add_subplot(4, 1, 3)
neuron_index = 256
V_E = runner.mon.V_E[:, neuron_index]
V_I = runner.mon.V_I[:, neuron_index]
ax3.plot(t_axis, V_E, label='V_E')
ax3.plot(t_axis, V_I, label='V_I')
ax3.legend(loc='upper right')

ax4 = fig.add_subplot(4, 1, 4)
# ax4 = fig.add_subplot(1, 1, 1)
im = ax4.imshow(ma.T, aspect='auto')
# # fig.colorbar(im, ax=ax4)
#
plt.show()
