import brainpy as bp
import numpy as np
import brainpy.math as bm
import mat73
from functools import partial
from skimage.transform import resize
import matplotlib.pyplot as plt


n_scale = 3
global_dt = 0.01
qI = 0.25
qE = 1 - qI
w, h = 48*n_scale, 32*n_scale
E_w, E_h = int(w*np.sqrt(qE)), int(h*np.sqrt(qE))
num_E, size_E = E_w*E_h, (E_h,E_w)
Ip_w, Ip_h = int(w*np.sqrt(qI)), int(h*np.sqrt(qI))
num_Ip, size_Ip = Ip_w*Ip_h, (Ip_h,Ip_w)
stim_a = 2 * (bm.pi/6)**2


def moving_average(a, n, axis):
    ret = bm.cumsum(a, axis=axis, dtype=bm.float32)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


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

    I = I / (bm.sqrt(bm.pi/2)*stim_a)

    return I


def persistent_input_protocol(amplitude, duration, dt=global_dt):
    bg_str = amplitude * 0.1
    st_amp = bp.inputs.section_input(values=[[bg_str]], durations=[500.], dt=global_dt)
    ramp_amp = bp.inputs.ramp_input(c_start=0, c_end=1 - bg_str, duration=100., dt=global_dt)
    hold_amp = bp.inputs.section_input(values=[[1 - bg_str]], durations=[1400.], dt=global_dt)
    remove_amp = bp.inputs.section_input(values=[[bg_str]], durations=[duration-2000.], dt=global_dt)

    E_bump = generate_bump_stimulus([0., 0.], E_w, E_h, stim_a, reshape=True)
    I_bump = generate_bump_stimulus([0., 0.], Ip_w, Ip_h, stim_a, reshape=True)

    E_inputs = bm.concatenate([
        st_amp * bm.ones([1, num_E]),
        ramp_amp[:, None] * E_bump[None, ] + bg_str*bm.ones([1, num_E]),
        hold_amp * E_bump[None, ] + bg_str * bm.ones([1, num_E]),
        remove_amp*bm.ones([1, num_E]),
    ])

    # shouldn't matter, because Ip has random connection
    I_inputs = bm.concatenate([
        st_amp * bm.ones([1, num_Ip]),
        ramp_amp[:, None] * I_bump[None, ] + bg_str*bm.ones([1, num_Ip]),
        hold_amp * I_bump[None, ] + bg_str * bm.ones([1, num_Ip]),
        remove_amp*bm.ones([1, num_Ip]),
    ])

    E_inputs = bm.reshape(E_inputs, (E_inputs.shape[0], *size_E))
    I_inputs = bm.reshape(I_inputs, (I_inputs.shape[0], *size_Ip))

    return E_inputs, I_inputs, duration


def tracking_input_protocol(amplitude, duration, n_period, dt=global_dt):
    bg_str = amplitude * 0.1
    bump_str = amplitude * 1.0
    n_step = int(duration / dt)
    pos_x = bm.linspace(-bm.pi*2/4, n_period*2*bm.pi, n_step)[:, None]
    pos_y = bm.linspace(0, 0, n_step)[:, None]
    pos = bm.hstack([pos_x, pos_y])
    E_inputs = bump_str * generate_bump_stimulus(pos, E_w, E_h, stim_a, reshape=False) + bg_str
    I_inputs = bump_str * generate_bump_stimulus(pos, Ip_w, Ip_h, stim_a, reshape=False) + bg_str

    return E_inputs, I_inputs, duration


def spiking_camera_input_protocol(amplitude, repeat_n, dt):
    file_name = "/Users/charlie/Local Documents/Projects/EI Balanced CANN/spiking camera data/diff_20200429_deep-sun-nei2.mat"
    data_dict = mat73.loadmat(file_name)
    inputs = bm.transpose(data_dict['diff_data'], [1, 0, 2])
    inputs = bm.repeat(inputs, repeat_n, axis=-1)
    E_inp = resize(inputs, size_E)
    E_inp = E_inp / bm.max(E_inp, axis=[0, 1], keepdims=True) * amplitude
    I_inp = resize(inputs, size_Ip)
    I_inp = I_inp / bm.max(I_inp, axis=[0, 1], keepdims=True) * amplitude
    duration = inputs.shape[-1] / 100.

    E_inp = bm.transpose(E_inp, [2, 0, 1])
    I_inp = bm.transpose(I_inp, [2, 0, 1])

    return E_inp, I_inp, duration


input_setup = {
    "persistent_input": partial(persistent_input_protocol, amplitude=1.0, duration=4000., dt=global_dt),
    "tracking_input": partial(tracking_input_protocol, amplitude=1.0, duration=3000, n_period=1., dt=global_dt),
    "spiking_camera_input": partial(spiking_camera_input_protocol, amplitude=1.0, repeat_n=3, dt=global_dt),
}


