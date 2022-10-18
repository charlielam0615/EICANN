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

input_setup = {
    "sin_input": sin_input_protocol(amplitude=1.0, frequency=10., duration=500., dt=global_dt),
    "linear_input": linear_input_protocol(start_amp=1.0, end_amp=10.0, duration=150., dt=global_dt),
    "constant_input": constant_input_protocol(amplitude=1.0, duration=100., dt=0.01),
}


