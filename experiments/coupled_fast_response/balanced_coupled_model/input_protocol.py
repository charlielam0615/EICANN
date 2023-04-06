import brainpy as bp
import brainpy.math as bm
from functools import partial


global_dt = 0.01
n_size = 0.125
size_E, size_I, stim_a = int(800 * n_size), int(100*n_size), 2*(bm.pi/6)**2


def linear_input_protocol(start_amp, end_amp, duration, input_duration, dt):
    ramp_dur = input_duration[1]-input_duration[0]
    st_amp = bp.inputs.section_input(values=[[start_amp]], durations=[input_duration[0]], dt=dt)
    ramp_amp = bp.inputs.ramp_input(c_start=start_amp, c_end=end_amp, duration=ramp_dur, dt=dt)
    hold_amp = bp.inputs.section_input(values=[[end_amp]], durations=[duration-input_duration[1]], dt=dt)

    inputs = bm.concatenate([st_amp, ramp_amp[:, None], hold_amp], axis=0)
    E_inputs, I_inputs = inputs, inputs

    return E_inputs, I_inputs, duration, input_duration


input_setup = {
    "linear_input": partial(linear_input_protocol, start_amp=1.0, end_amp=2.0, duration=250., input_duration=[184., 195.], dt=global_dt),
    "linear_input_save": partial(linear_input_protocol, start_amp=1.0, end_amp=2.0, duration=250., input_duration=[184., 195.], dt=global_dt),
}
