import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import brainpy as bp
import brainpy.math as bm
from functools import partial
from utils.input_utils import generate_bump_stimulus
from configs import (fast_cann_config,
                     slow_cann_config,
                     turn_off_config)

config_and_name = {slow_cann_config: "slow_CANN",
                   fast_cann_config: "fast_CANN",
                   turn_off_config: "turn_off_with_excitation"}

config_file = slow_cann_config
config = config_file.config

global_dt = config.global_dt
size_E, size_I = config.size_E, config.size_Ip
stim_a = 2 * config.stim_a ** 2


def persistent_protocol(amplitude, dt):
    # === Protocol Paramters ====
    duration = 2400.
    input_duration = [399., 1400.]
    hold_dur = input_duration[1] - input_duration[0] - 1
    bg_str = amplitude * 0.1
    # ===========================

    st_amp = bp.inputs.section_input(values=[[bg_str]], durations=[input_duration[0]], dt=dt)
    ramp_amp = bp.inputs.ramp_input(c_start=0, c_end=1 - bg_str, duration=1., dt=dt)
    hold_amp = bp.inputs.section_input(values=[[1 - bg_str]], durations=[hold_dur], dt=dt)
    remove_amp = bp.inputs.section_input(values=[[bg_str]], durations=[duration-input_duration[1]], dt=dt)

    E_bump = generate_bump_stimulus(0., size_E, stim_a)
    I_bump = generate_bump_stimulus(0., size_I, stim_a)

    E_inputs = bm.concatenate([
        st_amp * bm.ones([1, size_E]),
        ramp_amp[:, None] * E_bump[None, ] + bg_str*bm.ones([1, size_E]),
        hold_amp * E_bump[None, ] + bg_str * bm.ones([1, size_E]),
        remove_amp*bm.ones([1, size_E]),
    ])

    I_inputs = bm.concatenate([
        st_amp * bm.ones([1, size_I]),
        ramp_amp[:, None] * I_bump[None, ] + bg_str*bm.ones([1, size_I]),
        hold_amp * I_bump[None, ] + bg_str * bm.ones([1, size_I]),
        remove_amp*bm.ones([1, size_I]),
    ])

    return E_inputs, I_inputs, duration, input_duration


def tracking_protocol(amplitude, duration, n_period, dt=global_dt):
    input_duration = [0., duration]
    bg_str = amplitude * 0.1
    bump_str = amplitude - bg_str
    n_step = int(duration / dt)
    pos = bm.linspace(-bm.pi/2, n_period*bm.pi/2, n_step)[:, None]
    E_inputs = bump_str * generate_bump_stimulus(pos, size_E, stim_a) + bg_str
    I_inputs = bump_str * generate_bump_stimulus(pos, size_I, stim_a) + bg_str

    return E_inputs, I_inputs, duration, input_duration


def convergence_rate_population_readout_protocol(amplitude, duration, dt=global_dt):
    input_duration = [500., duration]
    hold_dur = input_duration[1]-input_duration[0]-10.
    bg_str = amplitude * 0.1
    st_amp = bp.inputs.section_input(values=[[bg_str]], durations=[input_duration[0]], dt=global_dt)
    ramp_amp = bp.inputs.ramp_input(c_start=0, c_end=1 - bg_str, duration=10., dt=global_dt)
    hold_amp = bp.inputs.section_input(values=[[1 - bg_str]], durations=[hold_dur], dt=global_dt)

    E_bump = generate_bump_stimulus(0., size_E, stim_a)
    I_bump = generate_bump_stimulus(0., size_I, stim_a)

    E_inputs = bm.concatenate([
        st_amp * bm.ones([1, size_E]),
        ramp_amp[:, None] * E_bump[None, ] + bg_str*bm.ones([1, size_E]),
        hold_amp * E_bump[None, ] + bg_str * bm.ones([1, size_E]),
    ])
    I_inputs = bm.concatenate([
        st_amp * bm.ones([1, size_I]),
        ramp_amp[:, None] * I_bump[None, ] + bg_str*bm.ones([1, size_I]),
        hold_amp * I_bump[None, ] + bg_str * bm.ones([1, size_I]),
    ])

    return E_inputs, I_inputs, duration, input_duration


def convergence_rate_current_protocol(amplitude, duration, dt=global_dt):
    input_duration = [999., duration]
    hold_dur = input_duration[1]-input_duration[0]-1.
    bg_str = amplitude * 0.1
    st_amp = bp.inputs.section_input(values=[[bg_str]], durations=[input_duration[0]], dt=global_dt)
    ramp_amp = bp.inputs.ramp_input(c_start=0, c_end=1 - bg_str, duration=1., dt=global_dt)
    hold_amp = bp.inputs.section_input(values=[[1 - bg_str]], durations=[hold_dur], dt=global_dt)

    E_bump = generate_bump_stimulus(0., size_E, stim_a)
    I_bump = generate_bump_stimulus(0., size_I, stim_a)

    E_inputs = bm.concatenate([
        st_amp * bm.ones([1, size_E]),
        ramp_amp[:, None] * E_bump[None, ] + bg_str*bm.ones([1, size_E]),
        hold_amp * E_bump[None, ] + bg_str * bm.ones([1, size_E]),
    ])
    I_inputs = bm.concatenate([
        st_amp * bm.ones([1, size_I]),
        ramp_amp[:, None] * I_bump[None, ] + bg_str*bm.ones([1, size_I]),
        hold_amp * I_bump[None, ] + bg_str * bm.ones([1, size_I]),
    ])

    return E_inputs, I_inputs, duration, input_duration


def noise_sensitivity_protocol(signal_amplitude, noise_amplitude, noise_cv, duration, dt):
    input_duration = [0., duration]
    bg_str = noise_amplitude
    stimulus_amp = bp.inputs.section_input(values=[[signal_amplitude-bg_str]], durations=[duration], dt=global_dt)
    stimulus_T = stimulus_amp.shape[0]

    E_bump = generate_bump_stimulus(0., size_E, stim_a)
    I_bump = generate_bump_stimulus(0., size_I, stim_a)
    E_noise = bm.sqrt(stimulus_amp*E_bump[None,] + bg_str) * bm.sqrt(dt) * bm.random.randn(stimulus_T, size_E) * noise_cv
    I_noise = bm.sqrt(stimulus_amp*I_bump[None,] + bg_str) * bm.sqrt(dt) * bm.random.randn(stimulus_T, size_I) * noise_cv

    E_inputs = bm.maximum(stimulus_amp*E_bump[None,]+bg_str+E_noise, 0)
    I_inputs = bm.maximum(stimulus_amp*I_bump[None,]+bg_str+I_noise, 0)

    return E_inputs, I_inputs, duration, input_duration


def sudden_change_convergence_protocol(amplitude, wait_dur, sti_dur, dt):
    input_duration = None
    wait_amp = bp.inputs.section_input(values=[[amplitude]], durations=[wait_dur], dt=dt)
    sti_amp = bp.inputs.section_input(values=[[amplitude]], durations=[sti_dur], dt=dt)

    E_bump_w = generate_bump_stimulus(0., size_E, stim_a)
    E_bump_s = generate_bump_stimulus(bm.pi / 6., size_E, stim_a)
    I_bump_w = generate_bump_stimulus(0., size_I, stim_a)
    I_bump_s = generate_bump_stimulus(bm.pi / 6., size_I, stim_a)

    E_inputs = bm.concatenate([
        wait_amp * E_bump_w,
        sti_amp * E_bump_s,
    ])
    I_inputs = bm.concatenate([
        wait_amp * I_bump_w,
        sti_amp * I_bump_s,
    ])

    duration = wait_dur + sti_dur

    return E_inputs, I_inputs, duration, input_duration


def smooth_moving_lag_protocol(amplitude, duration, n_period, dt):
    input_duration = [0., duration]
    bg_str = amplitude * 0.1
    bump_str = amplitude * 1.0
    n_step = int(duration / dt)
    pos = bm.linspace(0, n_period * 2 * bm.pi, n_step)[:, None]
    E_inputs = bump_str * generate_bump_stimulus(pos, size_E, stim_a) + bg_str
    I_inputs = bump_str * generate_bump_stimulus(pos, size_I, stim_a) + bg_str
    return E_inputs, I_inputs, duration, input_duration


def turn_off_with_exicitation_protocol(amplitude, duration, dt):
    input_duration = [399., 1400]
    hold_dur = input_duration[1]-input_duration[0]-1.
    bg_str = amplitude * 0.1
    st_amp = bp.inputs.section_input(values=[[bg_str]], durations=[input_duration[0]], dt=global_dt)
    ramp_amp = bp.inputs.ramp_input(c_start=0, c_end=1 - bg_str, duration=1., dt=global_dt)
    hold_amp = bp.inputs.section_input(values=[[1 - bg_str]], durations=[hold_dur], dt=global_dt)
    remove_amp = bp.inputs.section_input(values=[[0.]], durations=[duration-input_duration[1]], dt=global_dt)

    E_bump = generate_bump_stimulus(0., size_E, stim_a)
    I_bump = generate_bump_stimulus(0., size_I, stim_a)

    E_inputs = bm.concatenate([
        st_amp * bm.ones([1, size_E]),
        ramp_amp[:, None] * E_bump[None, ] + bg_str*bm.ones([1, size_E]),
        hold_amp * E_bump[None, ] + bg_str * bm.ones([1, size_E]),
        remove_amp*bm.ones([1, size_E]),
    ])
    I_inputs = bm.concatenate([
        st_amp * bm.ones([1, size_I]),
        ramp_amp[:, None] * I_bump[None, ] + bg_str*bm.ones([1, size_I]),
        hold_amp * I_bump[None, ] + bg_str * bm.ones([1, size_I]),
        remove_amp*bm.ones([1, size_I]),
    ])

    return E_inputs, I_inputs, duration, input_duration


input_setup = {
    "persistent_input": partial(persistent_protocol, amplitude=1.0, dt=global_dt),
    "tracking_input": partial(tracking_protocol, amplitude=1.0, duration=1500, n_period=1., dt=global_dt),
    "convergence_rate_population_readout_input": partial(convergence_rate_population_readout_protocol, amplitude=1.0, duration=1500., dt=global_dt),
    "convergence_rate_current_input": partial(convergence_rate_current_protocol, amplitude=1.0, duration=3000., dt=global_dt),
    "noise_sensitivity_input": partial(noise_sensitivity_protocol, signal_amplitude=1.0,
                                               noise_amplitude=0.5, noise_cv=1.0, duration=2000., dt=global_dt),
    "sudden_change_convergence_input": partial(sudden_change_convergence_protocol, amplitude=1.0, wait_dur=300., sti_dur=300.,
                                               dt=global_dt),
    "smooth_moving_lag_input": partial(smooth_moving_lag_protocol, amplitude=1.0, duration=3000, n_period=2,
                                          dt=global_dt),
    "turn_off_with_exicitation_input": partial(turn_off_with_exicitation_protocol, amplitude=1.0, duration=2400., dt=global_dt),                                
}


