import brainpy as bp
import brainpy.math as bm
from functools import partial


global_dt = 0.01


def generate_bump_stimulus(pos, size_n, stim_a):
    def dist(d):
        d = bm.remainder(d, 2 * bm.pi)
        d = bm.where(d > bm.pi, d - 2 * bm.pi, d)
        return d

    x = bm.linspace(-bm.pi, bm.pi, size_n)
    if bm.ndim(pos) == 2:
        x = x[None, ]
    I = 1.0 * bm.exp(-bm.pi * bm.square(dist(x - pos) / stim_a))
    return I


def background_input_protocol(amplitude, duration, dt=global_dt):
    size_E, size_I, stim_a = 750, 250, bm.pi / 6
    bg_str = amplitude * 0.1
    st_amp = bp.inputs.section_input(values=[[bg_str]], durations=[duration], dt=global_dt)
    E_inputs = st_amp * bm.ones([1, size_E])
    I_inputs = st_amp * bm.ones([1, size_I])
    return E_inputs, I_inputs, duration


def persistent_input_protocol(amplitude, duration, dt=global_dt):
    size_E, size_I, stim_a = 750, 250, bm.pi/6
    bg_str = amplitude * 0.1
    st_amp = bp.inputs.section_input(values=[[bg_str]], durations=[500.], dt=global_dt)
    ramp_amp = bp.inputs.ramp_input(c_start=0, c_end=1 - bg_str, duration=100., dt=global_dt)
    hold_amp = bp.inputs.section_input(values=[[1 - bg_str]], durations=[2400.], dt=global_dt)
    remove_amp = bp.inputs.section_input(values=[[0.]], durations=[duration-3000.], dt=global_dt)

    E_bump = generate_bump_stimulus(0., size_E, stim_a)
    I_bump = generate_bump_stimulus(0., size_I, stim_a)

    E_inputs = bm.concatenate([
        st_amp * bm.ones([1, size_E]),
        ramp_amp[:, None] * E_bump[None, ] + bg_str*bm.ones([1, size_E]),
        hold_amp * E_bump[None, ] + bg_str * bm.ones([1, size_E]),
        remove_amp*bm.ones([1, size_E]),
    ])

    # shouldn't matter, because Ip has random connection
    I_inputs = bm.concatenate([
        st_amp * bm.ones([1, size_I]),
        ramp_amp[:, None] * I_bump[None, ] + bg_str*bm.ones([1, size_I]),
        hold_amp * I_bump[None, ] + bg_str * bm.ones([1, size_I]),
        remove_amp*bm.ones([1, size_I]),
    ])

    return E_inputs, I_inputs, duration


def noisy_input_protocol(amplitude, duration, dt=global_dt):
    size_E, size_I, stim_a = 750, 250, bm.pi/6
    bg_str = amplitude * 0.1
    st_amp = bp.inputs.section_input(values=[[bg_str]], durations=[500.], dt=global_dt)
    ramp_amp = bp.inputs.ramp_input(c_start=0, c_end=1-bg_str, duration=100., dt=global_dt)
    hold_amp = bp.inputs.section_input(values=[[1 - bg_str]], durations=[duration-600.], dt=global_dt)

    ramp_T, hold_T = ramp_amp.shape[0], hold_amp.shape[0]

    E_bump = generate_bump_stimulus(0., size_E, stim_a)
    I_bump = generate_bump_stimulus(0., size_I, stim_a)

    ramp_noise = bm.sqrt(ramp_amp[:, None]+bg_str) * bm.sqrt(dt) * bm.random.randn(ramp_T, size_E)
    hold_noise = bm.sqrt(hold_amp+bg_str) * bm.sqrt(dt) * bm.random.randn(hold_T, size_E)

    E_inputs = bm.concatenate([
        st_amp * bm.ones([1, size_E]),
        bm.maximum(ramp_amp[:, None] * E_bump[None,] + bg_str * bm.ones([1, size_E]) + ramp_noise, 0),
        bm.maximum(hold_amp * E_bump[None,] + bg_str * bm.ones([1, size_E]) + hold_noise, 0),
    ])

    ramp_noise = bm.sqrt(ramp_amp[:, None] + bg_str) * bm.sqrt(dt) * bm.random.randn(ramp_T, size_I)
    hold_noise = bm.sqrt(hold_amp + bg_str) * bm.sqrt(dt) * bm.random.randn(hold_T, size_I)

    I_inputs = bm.concatenate([
        st_amp * bm.ones([1, size_I]),
        bm.maximum(ramp_amp[:, None] * I_bump[None, ] + bg_str*bm.ones([1, size_I]) + ramp_noise, 0),
        bm.maximum(hold_amp * I_bump[None, ] + bg_str * bm.ones([1, size_I]) + hold_noise, 0),
    ])

    return E_inputs, I_inputs, duration


def global_inhibition_protocol(amplitude, duration, dt=global_dt):
    size_E, size_I, stim_a = 750, 250, bm.pi / 6
    bg_str = amplitude * 0.1
    small_amp = amplitude * 0.4
    large_amp = amplitude * 0.9
    st_amp = bp.inputs.section_input(values=[[bg_str]], durations=[500.], dt=global_dt)
    ramp_small_amp = small_amp * bp.inputs.ramp_input(c_start=0, c_end=1., duration=100., dt=global_dt)
    hold_small_amp = small_amp * bp.inputs.section_input(values=[[1.]], durations=[1000.], dt=global_dt)
    ramp_large_amp = large_amp * bp.inputs.ramp_input(c_start=0, c_end=1., duration=100., dt=global_dt)
    small_rl_amp   = small_amp * bp.inputs.section_input(values=[[1.]], durations=[100.], dt=global_dt)
    hold_large_amp = large_amp * bp.inputs.section_input(values=[[1.]], durations=[1000.], dt=global_dt)
    small_hl_amp   = small_amp * bp.inputs.section_input(values=[[1.]], durations=[1000.], dt=global_dt)
    remove_amp = bp.inputs.section_input(values=[[0]], durations=[duration - 2700.], dt=global_dt)

    E_bump_large = generate_bump_stimulus(+bm.pi/2., size_E, stim_a)
    E_bump_small = generate_bump_stimulus(-bm.pi/2., size_E, stim_a)

    I_bump_large = generate_bump_stimulus(+bm.pi/2., size_I, stim_a)
    I_bump_small = generate_bump_stimulus(-bm.pi/2., size_I, stim_a)

    E_inputs = bm.concatenate([
        # background input
        st_amp*bm.ones([1, size_E]),
        # ramping small bump + bg input
        ramp_small_amp[:, None]*E_bump_small[None,] + bg_str*bm.ones([1,size_E]),
        # holding small bump + bg input
        hold_small_amp*E_bump_small[None,] + bg_str*bm.ones([1,size_E]),
        # holding small bump + ramping large bump + bg input
        ramp_large_amp[:, None]*E_bump_large[None,] + small_rl_amp*E_bump_small[None,] + bg_str*bm.ones([1,size_E]),
        # hold small and large bump + bg input
        hold_large_amp*E_bump_large[None,] + small_hl_amp*E_bump_small[None,] + bg_str*bm.ones([1, size_E]),
        # bg input
        remove_amp * bm.ones([1, size_E]),
    ])

    I_inputs = bm.concatenate([
        # background input
        st_amp * bm.ones([1, size_I]),
        # ramping small bump + bg input
        ramp_small_amp[:, None] * I_bump_small[None,] + bg_str * bm.ones([1, size_I]),
        # holding small bump + bg input
        hold_small_amp * I_bump_small[None,] + bg_str * bm.ones([1, size_I]),
        # holding small bump + ramping large bump + bg input
        ramp_large_amp[:, None] * I_bump_large[None,] + small_rl_amp * I_bump_small[None,] + bg_str * bm.ones(
            [1, size_I]),
        # hold small and large bump + bg input
        hold_large_amp * I_bump_large[None,] + small_hl_amp * I_bump_small[None,] + bg_str * bm.ones([1, size_I]),
        # bg input
        remove_amp * bm.ones([1, size_I]),
    ])

    return E_inputs, I_inputs, duration


def tracking_input_protocol(amplitude, duration, n_period, dt=global_dt):
    size_E, size_I, stim_a = 750, 250, bm.pi / 6
    bg_str = amplitude * 0.1
    bump_str = amplitude * 1.0
    n_step = int(duration / dt)
    pos = bm.linspace(-bm.pi*2/4, n_period*2*bm.pi, n_step)[:, None]
    E_inputs = bump_str * generate_bump_stimulus(pos, size_E, stim_a) + bg_str
    I_inputs = bump_str * generate_bump_stimulus(pos, size_I, stim_a) + bg_str

    return E_inputs, I_inputs, duration


def compare_speed_input_protocol(amplitude, duration, dt=global_dt):
    size_E, size_I, stim_a = 750, 250, bm.pi/6
    bg_str = amplitude * 0.1
    st_amp = bp.inputs.section_input(values=[[bg_str]], durations=[500.], dt=global_dt)
    ramp_amp = bp.inputs.ramp_input(c_start=0, c_end=1 - bg_str, duration=10., dt=global_dt)
    hold_amp = bp.inputs.section_input(values=[[1 - bg_str]], durations=[duration-510.], dt=global_dt)

    E_bump = generate_bump_stimulus(0., size_E, stim_a)
    I_bump = generate_bump_stimulus(0., size_I, stim_a)

    E_inputs = bm.concatenate([
        st_amp * bm.ones([1, size_E]),
        ramp_amp[:, None] * E_bump[None, ] + bg_str*bm.ones([1, size_E]),
        hold_amp * E_bump[None, ] + bg_str * bm.ones([1, size_E]),
    ])

    # shouldn't matter, because Ip has random connection
    I_inputs = bm.concatenate([
        st_amp * bm.ones([1, size_I]),
        ramp_amp[:, None] * I_bump[None, ] + bg_str*bm.ones([1, size_I]),
        hold_amp * I_bump[None, ] + bg_str * bm.ones([1, size_I]),
    ])

    return E_inputs, I_inputs, duration


def compare_current_input_protocol(amplitude, duration, dt=global_dt):
    size_E, size_I, stim_a = 750, 250, bm.pi/6
    bg_str = amplitude * 0.1
    st_amp = bp.inputs.section_input(values=[[bg_str]], durations=[500.], dt=global_dt)
    ramp_amp = bp.inputs.ramp_input(c_start=0, c_end=1 - bg_str, duration=10., dt=global_dt)
    hold_amp = bp.inputs.section_input(values=[[1 - bg_str]], durations=[duration-510.], dt=global_dt)

    E_bump = generate_bump_stimulus(0., size_E, stim_a)
    I_bump = generate_bump_stimulus(0., size_I, stim_a)

    E_inputs = bm.concatenate([
        st_amp * bm.ones([1, size_E]),
        ramp_amp[:, None] * E_bump[None, ] + bg_str*bm.ones([1, size_E]),
        hold_amp * E_bump[None, ] + bg_str * bm.ones([1, size_E]),
    ])

    # shouldn't matter, because Ip has random connection
    I_inputs = bm.concatenate([
        st_amp * bm.ones([1, size_I]),
        ramp_amp[:, None] * I_bump[None, ] + bg_str*bm.ones([1, size_I]),
        hold_amp * I_bump[None, ] + bg_str * bm.ones([1, size_I]),
    ])

    return E_inputs, I_inputs, duration


def compare_noise_sensitivity_input_protocol(signal_amplitude, noise_amplitude, noise_cv, duration, dt):
    size_E, size_I, stim_a = 750, 250, bm.pi / 6
    bg_str = noise_amplitude
    stimulus_amp = bp.inputs.section_input(values=[[signal_amplitude-bg_str]], durations=[duration], dt=global_dt)
    stimulus_T = stimulus_amp.shape[0]

    E_bump = generate_bump_stimulus(0., size_E, stim_a)
    I_bump = generate_bump_stimulus(0., size_I, stim_a)
    E_noise = bm.sqrt(stimulus_amp*E_bump[None,] + bg_str) * bm.sqrt(dt) * bm.random.randn(stimulus_T, size_E) * noise_cv
    I_noise = bm.sqrt(stimulus_amp*I_bump[None,] + bg_str) * bm.sqrt(dt) * bm.random.randn(stimulus_T, size_I) * noise_cv

    E_inputs = bm.maximum(stimulus_amp*E_bump[None,]+bg_str+E_noise, 0)
    I_inputs = bm.maximum(stimulus_amp*I_bump[None,]+bg_str+I_noise, 0)

    return E_inputs, I_inputs, duration


input_setup = {
    "background_input": partial(background_input_protocol, amplitude=1.0, duration=2000., dt=global_dt),
    "persistent_input": partial(persistent_input_protocol, amplitude=1.0, duration=4500., dt=global_dt),
    "noisy_input": partial(noisy_input_protocol, amplitude=1.0, duration=3000., dt=global_dt),
    "global_inhibition": partial(global_inhibition_protocol, amplitude=1.0, duration=4700.),
    "tracking_input": partial(tracking_input_protocol, amplitude=1.0, duration=3000, n_period=10., dt=global_dt),
    "compare_speed_input": partial(compare_speed_input_protocol, amplitude=1.0, duration=3000., dt=global_dt),
    "compare_current_input": partial(compare_current_input_protocol, amplitude=1.0, duration=3000., dt=global_dt),
    "compare_noise_sensitivity_input": partial(compare_noise_sensitivity_input_protocol, signal_amplitude=1.0, noise_amplitude=0.2, noise_cv=1.0, duration=3000., dt=global_dt),
}


