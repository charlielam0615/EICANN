import brainpy as bp
import brainpy.math as bm
from functools import partial


global_dt = 0.01
n_size = 1
size_E, size_I, stim_a = int(800 * n_size), int(100*n_size), 2*(bm.pi/6)**2


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


def background_input_protocol(amplitude, duration, dt=global_dt):
    bg_str = amplitude * 0.1
    st_amp = bp.inputs.section_input(values=[[bg_str]], durations=[duration], dt=global_dt)
    E_inputs = st_amp * bm.ones([1, size_E])
    I_inputs = st_amp * bm.ones([1, size_I])
    return E_inputs, I_inputs, duration


def persistent_input_protocol(amplitude, duration, n_scale=1, dt=global_dt):
    bg_str = amplitude * 0.1
    st_amp = bp.inputs.section_input(values=[[bg_str]], durations=[500.], dt=global_dt)
    ramp_amp = bp.inputs.ramp_input(c_start=0, c_end=1 - bg_str, duration=100., dt=global_dt)
    hold_amp = bp.inputs.section_input(values=[[1 - bg_str]], durations=[1400.], dt=global_dt)
    remove_amp = bp.inputs.section_input(values=[[bg_str]], durations=[duration-2000.], dt=global_dt)

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


def check_balance_input_protocol(amplitude, duration, dt=global_dt):
    bg_str = amplitude * 0.1
    st_amp = bp.inputs.section_input(values=[[bg_str]], durations=[500.], dt=global_dt)
    ramp_amp = bp.inputs.ramp_input(c_start=0, c_end=amplitude-bg_str, duration=10., dt=global_dt)
    hold_amp = bp.inputs.section_input(values=[[amplitude - bg_str]], durations=[duration-510.], dt=global_dt)

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


def check_balance_flat_input_protocol(amplitude, duration, dt=global_dt):
    bg_str = amplitude * 0.1
    st_amp = bp.inputs.section_input(values=[[bg_str]], durations=[500.], dt=global_dt)
    ramp_amp = bp.inputs.ramp_input(c_start=0, c_end=amplitude-bg_str, duration=10., dt=global_dt)
    hold_amp = bp.inputs.section_input(values=[[amplitude - bg_str]], durations=[duration-510.], dt=global_dt)

    ramp_T, hold_T = ramp_amp.shape[0], hold_amp.shape[0]

    ramp_noise = bm.sqrt(ramp_amp[:, None]+bg_str) * bm.sqrt(dt) * bm.random.randn(ramp_T, size_E)
    hold_noise = bm.sqrt(hold_amp+bg_str) * bm.sqrt(dt) * bm.random.randn(hold_T, size_E)

    E_inputs = bm.concatenate([
        st_amp * bm.ones([1, size_E]),
        bm.maximum(ramp_amp[:, None] * bm.ones([1, size_E]) + bg_str * bm.ones([1, size_E]) + ramp_noise, 0),
        bm.maximum(hold_amp * bm.ones([1, size_E]) + bg_str * bm.ones([1, size_E]) + hold_noise, 0),
    ])

    ramp_noise = bm.sqrt(ramp_amp[:, None] + bg_str) * bm.sqrt(dt) * bm.random.randn(ramp_T, size_I)
    hold_noise = bm.sqrt(hold_amp + bg_str) * bm.sqrt(dt) * bm.random.randn(hold_T, size_I)

    I_inputs = bm.concatenate([
        st_amp * bm.ones([1, size_I]),
        bm.maximum(ramp_amp[:, None] * bm.ones([1, size_I]) + bg_str*bm.ones([1, size_I]) + ramp_noise, 0),
        bm.maximum(hold_amp * bm.ones([1, size_I]) + bg_str * bm.ones([1, size_I]) + hold_noise, 0),
    ])

    return E_inputs, I_inputs, duration


def noisy_input_protocol(amplitude, duration, dt=global_dt):
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
    bg_str = amplitude * 0.1
    small_amp = amplitude * 0.4
    large_amp = amplitude * 0.9
    st_amp = bp.inputs.section_input(values=[[bg_str]], durations=[500.], dt=global_dt)
    ramp_small_amp = small_amp * bp.inputs.ramp_input(c_start=0, c_end=1., duration=100., dt=global_dt)
    hold_small_amp = small_amp * bp.inputs.section_input(values=[[1.]], durations=[1000.], dt=global_dt)
    ramp_large_amp = large_amp * bp.inputs.ramp_input(c_start=0, c_end=1., duration=100., dt=global_dt)
    small_rl_amp   = small_amp * bp.inputs.section_input(values=[[1.]], durations=[100.], dt=global_dt)
    hold_large_amp = large_amp * bp.inputs.section_input(values=[[1.]], durations=[2500.], dt=global_dt)
    small_hl_amp   = small_amp * bp.inputs.section_input(values=[[1.]], durations=[2500.], dt=global_dt)
    remove_amp = bp.inputs.section_input(values=[[0]], durations=[duration - 4200.], dt=global_dt)

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
    bg_str = amplitude * 0
    bump_str = amplitude * 1.0 - bg_str
    n_step = int(duration / dt)
    pos = bm.linspace(0, n_period*2*bm.pi, n_step)[:, None]
    E_inputs = bump_str * generate_bump_stimulus(pos, size_E, stim_a) + bg_str
    I_inputs = bump_str * generate_bump_stimulus(pos, size_I, stim_a) + bg_str

    return E_inputs, I_inputs, duration


def compare_speed_input_protocol(amplitude, duration, dt=global_dt):
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
    bg_str = amplitude * 0.1
    st_amp = bp.inputs.section_input(values=[[bg_str]], durations=[999.], dt=global_dt)
    ramp_amp = bp.inputs.ramp_input(c_start=0, c_end=1 - bg_str, duration=1., dt=global_dt)
    hold_amp = bp.inputs.section_input(values=[[1 - bg_str]], durations=[duration-1000.], dt=global_dt)

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
    bg_str = noise_amplitude
    stimulus_amp = bp.inputs.section_input(values=[[signal_amplitude-bg_str]], durations=[duration], dt=dt)
    stimulus_T = stimulus_amp.shape[0]

    E_bump = generate_bump_stimulus(0., size_E, stim_a)
    I_bump = generate_bump_stimulus(0., size_I, stim_a)
    E_noise = bm.sqrt(stimulus_amp*E_bump[None,] + bg_str) * bm.sqrt(dt) * bm.random.randn(stimulus_T, size_E) * noise_cv
    I_noise = bm.sqrt(stimulus_amp*I_bump[None,] + bg_str) * bm.sqrt(dt) * bm.random.randn(stimulus_T, size_I) * noise_cv

    E_inputs = bm.maximum(stimulus_amp*E_bump[None,]+bg_str+E_noise, 0)
    I_inputs = bm.maximum(stimulus_amp*I_bump[None,]+bg_str+I_noise, 0)

    return E_inputs, I_inputs, duration


def sudden_change_stimulus(amplitude, wait_dur, sti_dur, dt):
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

    return E_inputs, I_inputs, duration


def smooth_moving_stimulus(amplitude, duration, n_period, dt):
    bg_str = amplitude * 0.1
    bump_str = amplitude * 1.0
    n_step = int(duration / dt)
    pos = bm.linspace(0, n_period * 2 * bm.pi, n_step)[:, None]
    E_inputs = bump_str * generate_bump_stimulus(pos, size_E, stim_a) + bg_str
    I_inputs = bump_str * generate_bump_stimulus(pos, size_I, stim_a) + bg_str
    return E_inputs, I_inputs, duration


input_setup = {
    "background_input": partial(background_input_protocol, amplitude=1.0, duration=2000., dt=global_dt),
    "persistent_input": partial(persistent_input_protocol, amplitude=1.0, duration=3000., n_scale=1, dt=global_dt),
    "check_balance_input": partial(check_balance_input_protocol, amplitude=3.0, duration=1200., dt=global_dt),
    "check_balance_flat_input": partial(check_balance_flat_input_protocol, amplitude=3.0, duration=1500., dt=global_dt),
    "noisy_input": partial(noisy_input_protocol, amplitude=1.0, duration=3000., dt=global_dt),
    "global_inhibition": partial(global_inhibition_protocol, amplitude=1.0, duration=4700.),
    "tracking_input": partial(tracking_input_protocol, amplitude=1.0, duration=3000, n_period=10, dt=global_dt),
    "compare_speed_input": partial(compare_speed_input_protocol, amplitude=1.0, duration=1500., dt=global_dt),
    "compare_current_input": partial(compare_current_input_protocol, amplitude=1.0, duration=2000., dt=global_dt),
    "compare_noise_sensitivity_input": partial(compare_noise_sensitivity_input_protocol, signal_amplitude=1.0, noise_amplitude=0.1, noise_cv=1.0, duration=2000., dt=global_dt),
    "sudden_change_stimulus_converge": partial(sudden_change_stimulus, amplitude=1.0, wait_dur=300., sti_dur=300., dt=global_dt),
    "smooth_moving_stimulus_lag": partial(smooth_moving_stimulus, amplitude=1.0, duration=3000, n_period=2, dt=global_dt),
}
