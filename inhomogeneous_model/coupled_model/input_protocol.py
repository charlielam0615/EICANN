import brainpy as bp
import brainpy.math as bm
from functools import partial
import matplotlib.pyplot as plt

global_dt = 0.01


def generate_bump_stimulus(pos, size_n, stim_a):
    def dist(d):
        d = bm.remainder(d, 2 * bm.pi)
        d = bm.where(d > bm.pi, d - 2 * bm.pi, d)
        return d

    x = bm.linspace(-bm.pi, bm.pi, size_n)
    if bm.ndim(pos) == 2:
        x = x[None, ]
    I = 1.0 * bm.exp(-bm.square(dist(x-pos)) / stim_a)
    return I


def different_input_position_protocol(amplitude, duration, loop_ratio, dt):
    size_E, size_I, stim_a = 750, 250, 2 * (bm.pi / 6) ** 2
    bg_str = amplitude * 0.3
    pos = -bm.pi + loop_ratio * 2*bm.pi
    # pos = -bm.pi / 4 + loop_ratio * bm.pi / 2

    E_amp = bp.inputs.section_input(values=[[1 - bg_str]], durations=[duration], dt=dt)
    I_amp = bp.inputs.section_input(values=[[1 - bg_str]], durations=[duration], dt=dt)

    E_inputs = E_amp * generate_bump_stimulus(pos, size_E, stim_a) + bg_str
    I_inputs = I_amp * generate_bump_stimulus(pos, size_I, stim_a) + bg_str

    return E_inputs, I_inputs, duration


def different_input_position_sanity_check(amplitude, duration, loop_ratio, dt):
    return different_input_position_protocol(amplitude, duration, loop_ratio, dt)


input_setup = {
    "different_input_position_protocol": partial(different_input_position_protocol, amplitude=1.0, duration=1300., dt=global_dt),
    "sanity_check": partial(different_input_position_sanity_check, amplitude=1.0, duration=1300., dt=global_dt),
}


