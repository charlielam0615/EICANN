import brainpy as bp
import numpy as np
import brainpy.math as bm
from functools import partial
import mat73
from skimage.transform import resize
import matplotlib.pyplot as plt


n_scale = 2
global_dt = 0.01
qI = 0.25
qE = 1 - qI
w, h = 35*n_scale, 35*n_scale
E_w, E_h = int(w*np.sqrt(qE)), int(h*np.sqrt(qE))
num_E, size_E = E_w*E_h, (E_h,E_w)
Ip_w, Ip_h = int(w*np.sqrt(qI)), int(h*np.sqrt(qI))
num_Ip, size_Ip = Ip_w*Ip_h, (Ip_h,Ip_w)
Id_w, Id_h = int(w*np.sqrt(qI)), int(h*np.sqrt(qI))
num_Id, size_Id = Id_w*Id_h, (Id_h,Id_w)
stim_a = 2 * (bm.pi/12)**2


def generate_bump_stimulus(pos, size_w, size_h, stim_a, reshape=True):
    def dist(d):
        d = bm.remainder(d, 2 * bm.pi)
        d = bm.where(d > bm.pi, d - 2 * bm.pi, d)
        return d

    x = bm.linspace(-bm.pi, bm.pi, size_w)
    y = bm.linspace(-bm.pi, bm.pi, size_h)
    if bm.ndim(pos) == 2:
        x, y = x[None, ], y[None, ]
        dist_x = dist(x - pos[:, 0:1]) ** 2
        dist_y = dist(y - pos[:, 1:2]) ** 2
        d_square = dist_x[:, None] + dist_y[..., None]
        I = 1.0 * bm.exp(-d_square / stim_a) * (bm.sqrt(bm.pi/2)*stim_a)
        if reshape:
            I = bm.reshape(I, (I.shape[0],-1))
    elif bm.ndim(pos) == 1:
        d_square = dist(x[None,] - pos[0]) ** 2 + dist(y[:,None] - pos[1]) ** 2
        I = 1.0 * bm.exp(-d_square / stim_a) * (bm.sqrt(bm.pi/2)*stim_a)
        if reshape:
            I = bm.reshape(I, -1)

    I = I #/ (bm.sqrt(bm.pi/2)*stim_a)

    return I


def convert_rate_to_spike(rate_input, dt, max_rate):
    rate_input = rate_input / bm.max(rate_input) * max_rate * dt
    spike_input = bm.random.rand(*rate_input.shape)
    spike_input = (spike_input < rate_input).astype(bm.float32)
    return spike_input


def persistent_input_protocol(amplitude, duration, dt=global_dt):
    bg_str = 0.1
    st_amp = bp.inputs.section_input(values=[[bg_str]], durations=[500.], dt=global_dt)
    ramp_amp = bp.inputs.ramp_input(c_start=0, c_end=amplitude - bg_str, duration=10., dt=global_dt)
    hold_amp = bp.inputs.section_input(values=[[amplitude - bg_str]], durations=[390.], dt=global_dt)
    remove_amp = bp.inputs.section_input(values=[[bg_str]], durations=[duration - 900.], dt=global_dt)

    E_bump = generate_bump_stimulus([0., 0.], E_w, E_h, stim_a, reshape=True)
    I_bump = generate_bump_stimulus([0., 0.], Ip_w, Ip_h, stim_a, reshape=True)

    E_inputs = bm.concatenate([
        st_amp * bm.ones([1, num_E]),
        ramp_amp[:, None] * E_bump[None,] + bg_str * bm.ones([1, num_E]),
        hold_amp * E_bump[None,] + bg_str * bm.ones([1, num_E]),
        remove_amp * bm.ones([1, num_E]),
    ])

    # shouldn't matter, because Ip has random connection
    I_inputs = bm.concatenate([
        st_amp * bm.ones([1, num_Ip]),
        ramp_amp[:, None] * I_bump[None,] + bg_str * bm.ones([1, num_Ip]),
        hold_amp * I_bump[None,] + bg_str * bm.ones([1, num_Ip]),
        remove_amp * bm.ones([1, num_Ip]),
    ])

    E_inputs = bm.reshape(E_inputs, (E_inputs.shape[0], *size_E))
    I_inputs = bm.reshape(I_inputs, (I_inputs.shape[0], *size_Ip))

    return E_inputs, I_inputs, duration


def persistent_spiking_input_protocol(amplitude, duration, dt):
    E_inputs, I_inputs, duration = persistent_input_protocol(amplitude=amplitude, duration=duration, dt=dt)
    E_inputs = convert_rate_to_spike(E_inputs, dt, max_rate=10.)
    I_inputs = convert_rate_to_spike(I_inputs, dt, max_rate=10.)
    return E_inputs, I_inputs, duration


def tracking_input_protocol(amplitude, duration, n_period, warm_up_time, bg_str, dt):
    # warm-up stimulus
    st_amp = bp.inputs.section_input(values=[[bg_str]], durations=[warm_up_time], dt=global_dt)
    E_st = bm.reshape(st_amp * bm.ones([1, num_E]), (st_amp.shape[0], *size_E))
    I_st = bm.reshape(st_amp * bm.ones([1, num_Ip]), (st_amp.shape[0], *size_Ip))

    # moving stimulus
    bump_str = amplitude * 1.0 - bg_str
    n_step = int(duration / dt)
    pos_x = bm.linspace(-bm.pi*2/4, n_period*2*bm.pi, n_step)[:, None]
    pos_y = bm.linspace(0, 0, n_step)[:, None]
    pos = bm.hstack([pos_x, pos_y])
    E_inp = bump_str * generate_bump_stimulus(pos, E_w, E_h, stim_a, reshape=False) + bg_str
    I_inp = bump_str * generate_bump_stimulus(pos, Ip_w, Ip_h, stim_a, reshape=False) + bg_str

    E_inp = bm.concatenate([E_st, E_inp], axis=0)
    I_inp = bm.concatenate([I_st, I_inp], axis=0)

    duration = E_inp.shape[0] / 100.

    return E_inp, I_inp, duration


def tracking_spiking_input_protocol(amplitude, duration, n_period, warm_up_time, bg_str, dt):
    E_inputs, I_inputs, duration = tracking_input_protocol(amplitude=amplitude, duration=duration,
                                                           n_period=n_period, warm_up_time=warm_up_time,
                                                           bg_str=bg_str, dt=dt)
    E_inputs = convert_rate_to_spike(E_inputs, dt, max_rate=10.)
    I_inputs = convert_rate_to_spike(I_inputs, dt, max_rate=10.)
    return E_inputs, I_inputs, duration


def spiking_camera_input_protocol(amplitude, repeat_n, load_mat, warm_up_time, dt):
    if load_mat:
        file_name = "/Users/charlie/Local Documents/Projects/EI Balanced CANN/spiking camera data/" \
                    "diff_20200429_deep-sun-nei2.mat"
        data_dict = mat73.loadmat(file_name)
        inputs = bm.transpose(data_dict['diff_data'], [1, 0, 2])
        E_inp = resize(inputs, size_E)
        E_inp = E_inp / bm.max(E_inp, axis=[0, 1], keepdims=True)
        I_inp = resize(inputs, size_Ip)
        I_inp = I_inp / bm.max(I_inp, axis=[0, 1], keepdims=True)
        np.savez_compressed("data/E_inp_diff_20200429_deep-sun-nei2.npz", inputs=E_inp)
        np.savez_compressed("data/I_inp_diff_20200429_deep-sun-nei2.npz", inputs=I_inp)

    E_inp = np.load("data/E_inp_diff_20200429_deep-sun-nei2.npz")['inputs'] * amplitude
    I_inp = np.load("data/I_inp_diff_20200429_deep-sun-nei2.npz")['inputs'] * amplitude
    E_inp = bm.repeat(E_inp, repeat_n, axis=-1)
    I_inp = bm.repeat(I_inp, repeat_n, axis=-1)

    # warm-up inputs
    bg_str = 1.
    st_amp = bp.inputs.section_input(values=[[bg_str]], durations=[warm_up_time], dt=global_dt)
    E_st = bm.reshape(st_amp * bm.ones([1, num_E]), (st_amp.shape[0], *size_E))
    E_st = convert_rate_to_spike(E_st, dt, max_rate=1.0) * bg_str
    I_st = bm.reshape(st_amp * bm.ones([1, num_Ip]), (st_amp.shape[0], *size_Ip))
    I_st = convert_rate_to_spike(I_st, dt, max_rate=1.0) * bg_str

    # concate warm-up inputs with real inputsd
    E_inp = bm.transpose(E_inp, [2, 0, 1])
    I_inp = bm.transpose(I_inp, [2, 0, 1])
    E_inp = bm.concatenate([E_st, E_inp], axis=0)
    I_inp = bm.concatenate([I_st, I_inp], axis=0)

    duration = E_inp.shape[0] / 100.

    return E_inp, I_inp, duration


input_setup = {
    "persistent_input": partial(persistent_input_protocol, amplitude=1.0, duration=1300., dt=global_dt),

    "persistent_spiking_input": partial(persistent_spiking_input_protocol, amplitude=4.0, duration=1300., dt=global_dt),

    "tracking_input": partial(tracking_input_protocol, amplitude=4.0, duration=250, n_period=1.,
                              warm_up_time=500., bg_str=0.5, dt=global_dt),
    "tracking_spiking_input": partial(tracking_spiking_input_protocol, amplitude=4.0, duration=250, n_period=1.,
                                      warm_up_time=500., bg_str=0.5, dt=global_dt),
    "spiking_camera_input": partial(spiking_camera_input_protocol, amplitude=10., repeat_n=10, load_mat=False,
                                    warm_up_time=500., dt=global_dt),
}
