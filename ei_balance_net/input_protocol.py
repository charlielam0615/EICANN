import brainpy as bp
import brainpy.math as bm
from functools import partial

global_dt = 0.01

def sin_input_protocol(amplitude, frequency=50., duration=500., dt=0.01):
    bg_inp = bp.inputs.section_input(values=[[amplitude*1.5]], durations=[100.], dt=dt)
    sin_inp = bp.inputs.sinusoidal_input(amplitude=amplitude, frequency=frequency, duration=duration-100., dt=dt) + amplitude*1.5
    inputs = bm.concatenate([bg_inp, sin_inp[:, None]], axis=0)
    E_inputs, I_inputs = inputs, inputs

    return E_inputs, I_inputs, duration

def linear_input_protocol(start_amp, end_amp, duration=150., dt=0.01):
    st_amp = bp.inputs.section_input(values=[[start_amp]], durations=[50.], dt=dt)
    ramp_amp = bp.inputs.ramp_input(c_start=start_amp, c_end=end_amp, duration=50., dt=dt)
    hold_amp = bp.inputs.section_input(values=[[end_amp]], durations=[duration-100.], dt=dt)
    inputs = bm.concatenate([st_amp, ramp_amp[:, None], hold_amp], axis=0)
    E_inputs, I_inputs = inputs, inputs

    return E_inputs, I_inputs, duration

def constant_input_protocol(amplitude, duration=100., dt=0.01):
    inputs = bp.inputs.section_input(values=[[amplitude]], durations=[duration], dt=dt)
    E_inputs, I_inputs = inputs, inputs

    return E_inputs, I_inputs, duration


def generate_bump_stimulus(pos, size_n, stim_a):
    def dist(d):
        d = bm.remainder(d, 2 * bm.pi)
        d = bm.where(d > bm.pi, d - 2 * bm.pi, d)
        return d

    x = bm.linspace(-bm.pi, bm.pi, size_n)
    if bm.ndim(pos) == 2:
        x = x[None, ]
    I = 1.0 * bm.exp(-bm.pi * bm.square(dist(x-pos)) / stim_a)
    return I


def localized_input_protocol(amplitude, duration=3000., dt=0.01):
    size_E, size_I, stim_a = 750, 250, 2*(bm.pi/6)**2
    bg_str = amplitude * 0.1
    st_amp = bp.inputs.section_input(values=[[bg_str]], durations=[1000.], dt=dt)
    ramp_amp = bp.inputs.ramp_input(c_start=0, c_end=amplitude - bg_str, duration=10., dt=dt)
    hold_amp = bp.inputs.section_input(values=[[amplitude - bg_str]], durations=[duration - 1010.], dt=dt)

    E_bump = generate_bump_stimulus(0., size_E, stim_a)
    I_bump = generate_bump_stimulus(0., size_I, stim_a)

    E_inputs = bm.concatenate([
        st_amp * bm.ones([1, size_E]),
        ramp_amp[:, None] * E_bump[None,] + bg_str * bm.ones([1, size_E]),
        hold_amp * E_bump[None,] + bg_str * bm.ones([1, size_E]),
    ])

    # shouldn't matter, because Ip has random connection
    I_inputs = bm.concatenate([
        st_amp * bm.ones([1, size_I]),
        ramp_amp[:, None] * I_bump[None,] + bg_str * bm.ones([1, size_I]),
        hold_amp * I_bump[None,] + bg_str * bm.ones([1, size_I]),
    ])

    return E_inputs, I_inputs, duration


def staircase_input_protocol(amplitude, dt):
    inputs, duration = bp.inputs.section_input(amplitude * bm.array([1., 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]), durations=[500] * 8,
                                               return_length=True, dt=dt)
    E_inputs, I_inputs = inputs, inputs
    return E_inputs, I_inputs, duration


def staircase_powerspec_input_protocol(amplitude, dt):
    inputs, duration = bp.inputs.section_input(amplitude * bm.array([1., 4., 8., 12., 16., 20., 24., 28.]),
                                               durations=[1500] * 8,
                                               return_length=True, dt=dt)
    # inputs, duration = bp.inputs.section_input(amplitude * bm.array([1.]),
    #                                            durations=[1000],
    #                                            return_length=True, dt=dt)
    E_inputs, I_inputs = inputs, inputs
    return E_inputs, I_inputs, duration


input_setup = {
    "sin_input": sin_input_protocol(amplitude=1.0, frequency=10., duration=500., dt=global_dt),
    "linear_input": linear_input_protocol(start_amp=1.0, end_amp=4., duration=150., dt=global_dt),
    "constant_input": constant_input_protocol(amplitude=1.0, duration=100., dt=global_dt),
    "localized_input": localized_input_protocol(amplitude=5.0, duration=3000., dt=global_dt),
    "staircase_input": staircase_input_protocol(amplitude=1.0, dt=global_dt),
    "staircase_powerspec_input": staircase_powerspec_input_protocol(amplitude=1.0, dt=global_dt),

}


