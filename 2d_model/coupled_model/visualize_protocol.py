import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
from functools import partial
from utils.animate import animate_2D


global_dt = 0.01


def moving_average(a, n, axis):
    ret = bm.cumsum(a, axis=axis, dtype=bm.float32)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def dist(d):
    d = bm.remainder(d, 2 * bm.pi)
    d = bm.where(d > bm.pi, d - 2 * bm.pi, d)
    return d


def plot_peak(t, size_E, sti_theta_x=None, sti_theta_y=None, inp_theta_x=None, inp_theta_y=None):
    def find_nearest(array, value):
        idx = (bm.abs(array - value)).argmin()
        return idx

    x = bm.linspace(-bm.pi, bm.pi, size_E[1])
    y = bm.linspace(-bm.pi, bm.pi, size_E[0])
    offset = size_E[1]
    if sti_theta_x is not None and sti_theta_y is not None:
        plt.plot(find_nearest(x, sti_theta_x[t]), find_nearest(y, sti_theta_y[t]),
                 markersize=8, marker='.', color='red')
    if inp_theta_x is not None and inp_theta_y is not None:
        plt.plot(find_nearest(x, inp_theta_x[t]) + offset, find_nearest(y, inp_theta_y[t]),
                 markersize=8, marker='.', color='red')


def decode_population_vector(v):
    def adjust_range(cos_value, sin_value):
        # adjust arctan range to (-pi, pi)
        tan_value = cos_value / sin_value
        theta = (tan_value>0).astype(bm.float32) * (bm.arctan(tan_value)-(sin_value<0).astype(bm.float32)*bm.pi) + \
                (tan_value<0).astype(bm.float32) * (bm.arctan(tan_value)+(sin_value<0).astype(bm.float32)*bm.pi)
        return theta

    T, h, w = v.shape
    y, x = bm.linspace(-bm.pi, bm.pi, h), bm.linspace(-bm.pi, bm.pi, w)
    quad_coord = bm.stack([bm.cos(y[None, :, None]) * v,
                           bm.sin(y[None, :, None]) * v,
                           bm.cos(x[None, None, :]) * v,
                           bm.sin(x[None, None, :]) * v], axis=-1)
    pcv = bm.sum(quad_coord, axis=[1, 2])
    theta_y = adjust_range(cos_value=pcv[:, 1], sin_value=pcv[:, 0])
    theta_x = adjust_range(cos_value=pcv[:, 3], sin_value=pcv[:, 2])

    return theta_y, theta_x


def persistent_input_protocol(runner, net, E_inp, duration, input_duration, neuron_index, animation_mode):
    size_E = net.size_E
    if not animation_mode:
        fig, gs = bp.visualize.get_figure(3, 4, 3, 3)
        # plot a summed window of spikes
        t_ = [400., 900., 1000., 1100.]
        window_len = 5000
        for i in range(len(t_)):
            fig.add_subplot(gs[0, i])
            spike_section = bm.sum(runner.mon['E.spike'][int(t_[i]*100):int(t_[i]*100+window_len)], axis=0)
            spike_section = bm.reshape(spike_section, size_E)
            plt.imshow(spike_section)
            plt.title(f'Response t={t_[i]}')

        # # plot stimulus
        # fig.add_subplot(gs[0, 1])
        # input_section = bm.sum(E_inp[int(input_duration[0] * 100 * 1.2):int(input_duration[0] * 100 * 1.6)], axis=0)
        # input_section = bm.reshape(input_section, size_E)
        # plt.imshow(input_section)
        # plt.title('Stimulus')

        # plot currents
        fig.add_subplot(gs[1, :])
        tf = lambda d: d.reshape(-1, *net.size_E)[:, neuron_index[0], neuron_index[1]]
        Ec_inp = tf(runner.mon['E2E_s.g']) + tf(runner.mon['E2E_f.g'])
        Fc_inp = tf(E_inp)
        Ic_inp = tf(runner.mon['I2E_s.g']) + tf(runner.mon['I2E_f.g'])
        shunting_inp = tf(runner.mon['ESI.output_value'])
        leak = net.E.gl * tf(runner.mon['E.V'])
        total_inp = Ec_inp + Ic_inp + Fc_inp + shunting_inp + leak
        T = 1000
        ts = moving_average(runner.mon.ts, n=T, axis=0)
        Ec_inp = moving_average(Ec_inp, n=T, axis=0)
        Fc_inp = moving_average(Fc_inp, n=T, axis=0)
        Ic_inp = moving_average(Ic_inp, n=T, axis=0)
        shunting_inp = moving_average(shunting_inp, n=T, axis=0)
        total_inp = moving_average(total_inp, n=T, axis=0)
        leak = moving_average(leak, n=T, axis=0)
        #  === plot detail current ====
        # bp.visualize.line_plot(ts, Ec_inp, legend='rec_E', alpha=0.5)
        # bp.visualize.line_plot(ts, Fc_inp, legend='F', alpha=0.5)
        # bp.visualize.line_plot(ts, Ic_inp, legend='rec_I', alpha=0.5)
        # bp.visualize.line_plot(ts, shunting_inp, legend='shunting_inp', alpha=0.5)
        # bp.visualize.line_plot(ts, total_inp, legend='Total', alpha=0.5)
        # bp.visualize.line_plot(ts, leak, legend='leak', alpha=0.5)
        # === plot only F, E, I current ===
        bp.visualize.line_plot(ts, Ec_inp+Fc_inp, legend='E', alpha=0.5)
        bp.visualize.line_plot(ts, Ic_inp+shunting_inp+leak, legend='I', alpha=0.5)
        bp.visualize.line_plot(ts, total_inp, legend='total', alpha=0.5)
        # plot firing rates
        # fig.add_subplot(gs[2, :])
        # spike = tf(runner.mon['E.spike'])
        # avg_spike = moving_average(spike, n=T, axis=0)
        # bp.visualize.line_plot(ts, avg_spike, legend='firing rate', alpha=0.5)

    else:
        T = 1000  # average window: 10 ms
        avg_spike = moving_average(runner.mon['E.spike'], n=T, axis=0)
        avg_spike = avg_spike.reshape(avg_spike.shape[0], *size_E)
        avg_spike = avg_spike / bm.maximum(bm.max(avg_spike, axis=[1, 2], keepdims=True), 1e-6)
        avg_input = moving_average(E_inp, n=T, axis=0)
        avg_input = avg_input / bm.maximum(bm.max(avg_input, axis=[1, 2], keepdims=True), 1e-6)
        vis_value = bm.concatenate([avg_input, avg_spike], axis=-1)
        vis_value = vis_value.reshape(vis_value.shape[0], -1)
        animate_2D(values=vis_value,
                   dt=global_dt,
                   figsize=(8, 4),
                   video_fps=30,
                   frame_step=100,
                   # save_path='./persistent_activity.mp4',
                   net_size=(net.size_E[0], 2 * net.size_E[1]),
                   aggregate_func=None)


def persistent_spiking_input_protocol(runner, net, E_inp, duration, input_duration, neuron_index, animation_mode):
    return persistent_input_protocol(runner=runner, net=net, E_inp=E_inp, duration=duration,
                                     input_duration=input_duration, neuron_index=neuron_index,
                                     animation_mode=animation_mode)


def tracking_input_protocol(runner, net, E_inp, input_duration, warm_up_time, animation_mode):
    size_E = net.size_E
    if not animation_mode:
        fig, gs = bp.visualize.get_figure(3, 4, 2, 2)
        duration = 250.
        _t = warm_up_time+bm.array([0.2, 0.4, 0.6, 0.8]) * duration
        window_len = 1000  # average window: 10 ms
        for i in range(len(_t)):
            # plot a summed window of spikes
            fig.add_subplot(gs[0, i])
            spike_section = bm.sum(runner.mon['E.spike'][int(_t[i]*100):int(_t[i]*100+window_len)], axis=0)
            spike_section = bm.reshape(spike_section, size_E)
            spike_theta_y, spike_theta_x = decode_population_vector(spike_section[None,])
            plt.imshow(spike_section)
            plot_peak(t=0, size_E=size_E, sti_theta_x=spike_theta_x[None,], sti_theta_y=spike_theta_y[None,])
            # plot a summed window of stimulus
            fig.add_subplot(gs[1, i])
            input_section = bm.sum(E_inp[int(_t[i]*100):int(_t[i]*100+window_len)], axis=0)
            input_section = bm.reshape(input_section, size_E)
            plt.imshow(input_section)

        # plot decoding results
        fig.add_subplot(gs[2, :])
        T = window_len
        ts = moving_average(runner.mon.ts, n=T, axis=0)
        ma = moving_average(runner.mon['E.spike'], n=T, axis=0)
        ma = ma.reshape(-1, *size_E)
        inp = moving_average(E_inp, n=T, axis=0)
        inp = inp - bm.min(inp, axis=[1,2], keepdims=True)
        inp_theta_y, inp_theta_x = decode_population_vector(inp)
        sti_theta_y, sti_theta_x = decode_population_vector(ma)
        decode_error = bm.sqrt(dist(inp_theta_y-sti_theta_y)**2+dist(inp_theta_x-sti_theta_x)**2)
        plt.plot(ts, decode_error, label="error")
        plt.xlim([warm_up_time, warm_up_time+duration])
        plt.legend()
    else:
        plt.figure(figsize=(8, 4))
        T = 1000  # average window: 10 ms
        avg_spike = moving_average(runner.mon['E.spike'], n=T, axis=0)
        avg_spike = avg_spike.reshape(avg_spike.shape[0], *size_E)
        avg_spike = avg_spike / bm.maximum(bm.max(avg_spike, axis=[1, 2], keepdims=True), 1e-6)
        avg_input = moving_average(E_inp, n=T, axis=0)
        avg_input = avg_input / bm.maximum(bm.max(avg_input, axis=[1, 2], keepdims=True), 1e-6)

        inp_theta_y, inp_theta_x = decode_population_vector(avg_spike)
        sti_theta_y, sti_theta_x = decode_population_vector(avg_input-bm.min(avg_input, axis=[1, 2], keepdims=True))

        vis_value = bm.concatenate([avg_input, avg_spike], axis=-1)
        vis_value = vis_value.reshape(vis_value.shape[0], -1)

        animate_2D(values=vis_value,
                   frame_step=100,
                   net_size=(net.size_E[0], 2*net.size_E[1]),
                   aggregate_func=lambda t: partial(plot_peak,
                                                    size_E=size_E,
                                                    sti_theta_x=sti_theta_x,
                                                    sti_theta_y=sti_theta_y,
                                                    inp_theta_x=inp_theta_x,
                                                    inp_theta_y=inp_theta_y)(t))


def tracking_spiking_input_protocol(runner, net, E_inp, input_duration, animation_mode, warm_up_time):
    return tracking_input_protocol(runner=runner, net=net, E_inp=E_inp,
                                   input_duration=input_duration, animation_mode=animation_mode,
                                   warm_up_time=warm_up_time)


def spiking_camera_input_protocol(runner, net, E_inp, warm_up_time, animation_mode):
    size_E = net.size_E
    if not animation_mode:
        pass
    else:
        T = 100  # average window: 1 ms
        avg_spike = moving_average(runner.mon['E.spike'], n=T, axis=0)
        avg_spike = avg_spike.reshape(avg_spike.shape[0], *size_E)
        avg_spike = avg_spike / bm.maximum(bm.max(avg_spike, axis=[1, 2], keepdims=True), 1e-6)
        avg_input = moving_average(E_inp, n=T, axis=0)
        avg_input = avg_input / bm.maximum(bm.max(avg_input, axis=[1, 2], keepdims=True), 1e-6)
        vis_value = bm.concatenate([avg_input, avg_spike], axis=-1)
        vis_value = vis_value.reshape(vis_value.shape[0], -1)

        inp_theta_y, inp_theta_x = decode_population_vector(avg_spike)
        input_shape = avg_input.shape
        avg_input = bm.reshape(avg_input, [avg_input.shape[0], -1])
        max_ind = bm.argmax(avg_input, axis=-1)
        sti_y, sti_x = bm.unravel_index(max_ind, input_shape[1:])
        y, x = bm.linspace(-bm.pi, bm.pi, size_E[0]), bm.linspace(-bm.pi, bm.pi, size_E[1])
        sti_theta_y, sti_theta_x = y[sti_y], x[sti_x]

        animate_2D(values=vis_value[int((warm_up_time-25)*100):],
                   frame_step=100,
                   net_size=(net.size_E[0], 2 * net.size_E[1]),
                   aggregate_func=lambda t: partial(plot_peak,
                                                    size_E=size_E,
                                                    sti_theta_x=sti_theta_x,
                                                    sti_theta_y=sti_theta_y,
                                                    inp_theta_x=inp_theta_x,
                                                    inp_theta_y=inp_theta_y)(t)
                   )


vis_setup = {
    "persistent_input": partial(persistent_input_protocol, duration=1300., input_duration=(500, 900),
                                neuron_index=(27, 27), animation_mode=True),
    "persistent_spiking_input": partial(persistent_spiking_input_protocol, duration=1300., input_duration=(500, 900),
                                        neuron_index=(30, 30), animation_mode=False),
    "tracking_input": partial(tracking_input_protocol, input_duration=(0, 250), warm_up_time=500.,
                              animation_mode=False),
    "tracking_spiking_input": partial(tracking_spiking_input_protocol, input_duration=(0, 250), warm_up_time=500.,
                                      animation_mode=False),
    "spiking_camera_input": partial(spiking_camera_input_protocol, warm_up_time=500., animation_mode=True),
}
