import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
from functools import partial
from utils.animate import animate_2D


def moving_average(a, n, axis):
    ret = bm.cumsum(a, axis=axis, dtype=bm.float32)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def dist(d):
    d = bm.remainder(d, 2 * bm.pi)
    d = bm.where(d > bm.pi, d - 2 * bm.pi, d)
    return d


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


def plot_peak(t, size_E, sti_theta_x, sti_theta_y, inp_theta_x, inp_theta_y):
    def find_nearest(array, value):
        idx = (bm.abs(array - value)).argmin()
        return idx

    x = bm.linspace(-bm.pi, bm.pi, size_E[1])
    y = bm.linspace(-bm.pi, bm.pi, size_E[0])
    offset = size_E[1]
    plt.plot(find_nearest(x, sti_theta_x[t]), find_nearest(y, sti_theta_y[t]), markersize=8, marker='.', color='red')
    plt.plot(find_nearest(x, inp_theta_x[t]) + offset, find_nearest(y, inp_theta_y[t]), markersize=8, marker='.',
             color='red')


def persistent_input_protocol(runner, net, E_inp, duration, input_duration, neuron_index):
    size_E = net.size_E
    fig, gs = bp.visualize.get_figure(2, 2, 3, 3)
    # plot a summed window of spikes
    fig.add_subplot(gs[0, 0])
    spike_section = bm.sum(runner.mon['E.spike'][int(input_duration[0]*100*1.2):int(input_duration[0]*100*1.6)], axis=0)
    spike_section = bm.reshape(spike_section, size_E)
    plt.imshow(spike_section)
    # plot stimulus
    fig.add_subplot(gs[0, 1])
    input_section = bm.sum(E_inp[int(input_duration[0]*100*1.2):int(input_duration[0]*100*1.6)], axis=0)
    input_section = bm.reshape(input_section, size_E)
    plt.imshow(input_section)
    # plot currents
    fig.add_subplot(gs[1, 0:2])
    Ec_inp = runner.mon['E2E_s.g'].reshape(-1, *net.size_E)[:, neuron_index[0], neuron_index[1]]
    Fc_inp = E_inp.reshape(-1, *net.size_E)[:, neuron_index[0], neuron_index[1]]
    Ic_inp = runner.mon['I2E_s.g'].reshape(-1, *net.size_E)[:, neuron_index[0], neuron_index[1]]
    shunting_inp = net.shunting_k * (Ec_inp + Fc_inp) * Ic_inp
    leak = net.E.gl * runner.mon['E.V'].reshape(-1, *net.size_E)[:, neuron_index[0], neuron_index[1]]
    total_inp = Ec_inp + Ic_inp + Fc_inp + shunting_inp + leak
    bp.visualize.line_plot(runner.mon.ts, Ec_inp, legend='rec_E', alpha=0.5)
    bp.visualize.line_plot(runner.mon.ts, Fc_inp, legend='F', alpha=0.5)
    bp.visualize.line_plot(runner.mon.ts, Ic_inp, legend='I', alpha=0.5)
    bp.visualize.line_plot(runner.mon.ts, shunting_inp, legend='shunting_inp', alpha=0.5)
    bp.visualize.line_plot(runner.mon.ts, leak, legend='leak', alpha=0.5)
    bp.visualize.line_plot(runner.mon.ts, total_inp, legend='Total', alpha=0.5)

    # animation
    # T = 1000
    # avg_spike = moving_average(runner.mon['E.spike'], n=T, axis=0)  # average window: 10 ms
    # bp.visualize.animate_2D(values=avg_spike,
    #                         frame_step=100,
    #                         net_size=net.size_E)



def tracking_input_protocol(runner, net, E_inp, input_duration, animation_mode):
    size_E = net.size_E
    if not animation_mode:
        fig, gs = bp.visualize.get_figure(2, 2, 4, 4)
        start_ind = 120000
        end_ind = 121000
        # plot a summed window of spikes
        fig.add_subplot(gs[0, 0])
        spike_section = bm.sum(runner.mon['E.spike'][start_ind:end_ind], axis=0)
        spike_section = bm.reshape(spike_section, size_E)
        plt.imshow(spike_section)
        # plot a summed window of stimulus
        fig.add_subplot(gs[0, 1])
        input_section = bm.sum(E_inp[start_ind:end_ind], axis=0)
        input_section = bm.reshape(input_section, size_E)
        plt.imshow(input_section)
        # plot decoding results
        fig.add_subplot(gs[1, :2])
        T = 1000  # average window: 10 ms
        ts = moving_average(runner.mon.ts, n=T, axis=0)
        ma = moving_average(runner.mon['E.spike'], n=T, axis=0)
        ma = ma.reshape(-1, *size_E)
        inp = moving_average(E_inp, n=T, axis=0)
        inp = inp - bm.min(inp, axis=[1,2], keepdims=True)
        inp_theta_y, inp_theta_x = decode_population_vector(inp)
        sti_theta_y, sti_theta_x = decode_population_vector(ma)
        decode_error = bm.sqrt(dist(inp_theta_y-sti_theta_y)**2+dist(inp_theta_x-sti_theta_x)**2)
        plt.plot(ts, decode_error, label="error")
        plt.xlim([500, 3000])
        plt.legend()
    else:
        plt.figure(figsize=(6, 4))
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
                                                    inp_theta_y=inp_theta_y)(t)
                   )


def spiking_camera_input_protocol(runner, net, E_inp, animation_mode):
    size_E = net.size_E
    if not animation_mode:
        pass
    else:
        plt.figure(figsize=(6, 4))
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

        animate_2D(values=vis_value,
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
    "persistent_input": partial(persistent_input_protocol, duration=4000., input_duration=(500, 2000), neuron_index=(13, 13)),
    "tracking_input": partial(tracking_input_protocol, input_duration=(0, 3000), animation_mode=False),
    "spiking_camera_input": partial(spiking_camera_input_protocol, animation_mode=True),
}