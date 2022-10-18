import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt

bp.math.set_platform('cpu')
global_dt = 0.01

# ======== CANN ========
tau_Es = 150
tau_Ef = 30
tau_If = 10
k = 0.01
a = bm.pi/6
J0 = 1
A = 1
theta = 1.0

ei_scale = 1.0
wee = 1.0
jie = -1.4 * ei_scale
jii = -0.7 * ei_scale
jee = 0.5 * ei_scale
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
        self.U_Es = bm.Variable(bm.zeros(size))
        self.U_Ef = bm.Variable(bm.zeros(size))
        self.U_If = bm.Variable(bm.zeros(size))
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

    def derivative(self, U_Es, U_Ef, U_If, t, Iext):
        U_E = U_Es + U_Ef + Iext
        r1 = bm.square(U_E)
        r2 = 1.0 + self.k * bm.sum(r1)
        self.r_E.value = r1 / r2
        self.r_I.value = U_If / self.theta
        self.Irec_EEs.value = wee * bm.dot(self.conn_mat, self.r_E)
        self.Irec_EEf.value = jee * bm.dot(self.conn_mat, self.r_E)
        self.Irec_IEf.value = jie * bm.dot(self.conn_mat, self.r_I)
        self.Irec_EIf.value = jei * bm.dot(self.conn_mat, self.r_E)
        self.Irec_IIf.value = jii * bm.dot(self.conn_mat, self.r_I)
        dU_Es = (-U_Es + self.Irec_EEs) / self.tau_Es
        dU_Ef = (-U_Ef + self.Irec_EEf + self.Irec_IEf) / self.tau_Ef
        dU_If = (-U_If + self.Irec_EIf + self.Irec_IIf) / self.tau_If

        return dU_Es, dU_Ef, dU_If

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
        I = self.A * bm.exp(-bm.pi * bm.square(self.dist(x - pos) / self.a))
        return I

    def update(self, tdi):
        U_Es, U_Ef, U_If = self.integral(self.U_Es, self.U_Ef, self.U_If, tdi.t, self.input, tdi.dt)
        U_Es = bm.where(U_Es<0, self.U_Es, U_Es)
        U_Ef = bm.where(U_Ef<0, self.U_Ef, U_Ef)
        U_If = bm.where(U_If<0, self.U_If, U_If)
        self.U_Es.value, self.U_Ef.value, self.U_If.value = U_Es, U_Ef, U_If
        self.input[:] = 0.

n_size = 512
eicann = EICANN(size=n_size)

I1 = eicann.get_stimulus_by_pos(0., n_size)
Iext, duration = bp.inputs.section_input(values=[0.2, I1*0.8+0.2, 0.2],
                                         durations=[1000., 2550., 850.],
                                         dt=global_dt,
                                         return_length=True)
runner = bp.dyn.DSRunner(eicann,
                         inputs=['input', Iext, 'iter'],
                         monitors=['U_Es', 'U_Ef', 'Irec_EEs', 'Irec_EEf', 'Irec_IEf', 'Irec_EIf', 'Irec_IIf', 'r_E', 'r_I'],
                         dt=global_dt)
runner.run(duration)
x = bm.linspace(-bm.pi, bm.pi, n_size)
t_axis = bm.arange(0, duration, global_dt)
ma = runner.mon.r_E
bump_activity = bm.vstack([bm.sum(ma * bm.cos(x[None,]), axis=1), bm.sum(ma * bm.sin(x[None,]), axis=1)])
readout = bm.array([[1., 0.]]) @ bump_activity
nm_readout = readout.T / bm.mean(readout[:, 10000:60000])

fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(t_axis, bm.max(Iext, axis=1), label='Iext')
ax1.plot(t_axis, nm_readout, label='readout')
ax1.legend()

ax2 = fig.add_subplot(2, 1, 2)
neuron_index = 256
E_current = runner.mon.Irec_EEs[:, neuron_index] + runner.mon.Irec_EEf[:, neuron_index] + Iext[:, neuron_index]
I_current = runner.mon.Irec_IEf[:, neuron_index]
total_current = E_current + I_current
ax2.plot(t_axis, E_current, label='E_current')
ax2.plot(t_axis, I_current, label='I_current')
ax2.plot(t_axis, total_current, label='total_current')
ax2.legend()

plt.show()