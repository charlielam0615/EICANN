import brainpy as bp
import brainpy.math as bm
import numpy as np
import matplotlib.pyplot as plt
from functools import partial


def moving_average(a, n, axis):
    ret = bm.cumsum(a, axis=axis, dtype=bm.float32)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def get_pos_from_tan(a, b):
    pos = bm.arctan(a/b)
    offset_mask = b < 0
    pos = pos + offset_mask * bm.sign(a) * bm.pi
    return pos


def background_input_protocol(runner, net, E_inp):
    fig, gs = bp.visualize.get_figure(2, 1, 1.5, 10)
    # raster plot on E
    fig.add_subplot(gs[:1, 0])
    bp.visualize.raster_plot(runner.mon.ts, runner.mon['E.spike'], markersize=1.)
    # raster plot on Ip
    fig.add_subplot(gs[1:2, 0])
    bp.visualize.raster_plot(runner.mon.ts, runner.mon['Ip.spike'], markersize=1.)
    return


def persistent_input_protocol(runner, net, E_inp, duration, input_duration, neuron_indices):
    fig, gs = bp.visualize.get_figure(1, 1, 2, 8)
    # raster plot on E
    fig.add_subplot(gs[:1, 0])
    bp.visualize.raster_plot(runner.mon.ts, runner.mon['E.spike'], xlim=[0, duration], markersize=1., alpha=0.3)
    plt.plot(input_duration, [int(net.size_E/2), int(net.size_E/2)], label='input peak', color='red')

    # raster plot on Ip
    # fig.add_subplot(gs[1:2, 0])
    # bp.visualize.raster_plot(runner.mon.ts, runner.mon['Ip.spike'], markersize=1.)
    # # current plot for center E and peripheral E
    # for i in range(1):
    #     fig.add_subplot(gs[1 + 2 * i:3 + 2 * i, 0])
    #     neuron_index = neuron_indices[i]
    #     Ec_inp = runner.mon['E2E_s.g'][:,neuron_index]
    #     Fc_inp = E_inp[:,neuron_index]
    #     shunting_inp = net.shunting_k*(runner.mon['E2E_s.g'][:,neuron_index]+Fc_inp)*runner.mon['I2E_s.g'][:,neuron_index]
    #     Ic_inp = runner.mon['I2E_s.g'][:,neuron_index]
    #     total_inp = Ec_inp + Ic_inp + Fc_inp + shunting_inp
    #     bp.visualize.line_plot(runner.mon.ts, Ec_inp, legend='rec_E', alpha=0.5)
    #     bp.visualize.line_plot(runner.mon.ts, Fc_inp, legend='F', alpha=0.5)
    #     bp.visualize.line_plot(runner.mon.ts, Ic_inp, legend='I', alpha=0.5)
    #     bp.visualize.line_plot(runner.mon.ts, shunting_inp, legend='shunting_inp', alpha=0.5)
    #     bp.visualize.line_plot(runner.mon.ts, total_inp, legend='Total', alpha=0.5)
    #     plt.legend(loc=4)
    #     plt.grid('on')

    plt.show()


def noisy_input_protocol(runner, net, E_inp, duration, input_duration):
    fig, gs = bp.visualize.get_figure(2, 1, 1.5, 8)
    # raster plot on E
    fig.add_subplot(gs[:1, 0])
    bp.visualize.raster_plot(runner.mon.ts, runner.mon['E.spike'], markersize=1., alpha=0.5)
    plt.plot(input_duration, [int(net.size_E / 2), int(net.size_E / 2)], label='input peak', color='red')
    # # raster plot on Ip
    # fig.add_subplot(gs[1:2, 0])
    # bp.visualize.raster_plot(runner.mon.ts, runner.mon['Ip.spike'], markersize=1.)
    # # imshow input
    # fig.add_subplot(gs[2:3, 0])
    # plt.imshow(E_inp.T, aspect='auto')
    # input section
    fig.add_subplot(gs[1:2, 0])
    bp.visualize.line_plot(bm.arange(net.size_E), E_inp[int(duration*0.8)*100, ], legend='input')
    plt.legend()
    # imshow E spike
    # fig.add_subplot(gs[4:5, 0])
    # plt.imshow(runner.mon['E.spike'].T, aspect='auto')
    plt.show()


def global_inhibition_protocol(runner, net, E_inp, small_bump_duration, large_bump_duration, neuron_indices):
    fig, gs = bp.visualize.get_figure(1, 1, 2, 8)
    # raster plot on E
    fig.add_subplot(gs[:1, 0])
    bp.visualize.raster_plot(runner.mon.ts, runner.mon['E.spike'], markersize=1., alpha=0.5)
    plt.plot(small_bump_duration, [int(net.size_E / 4), int(net.size_E / 4)], label='input peak', color='red',
             linewidth=1.0, alpha=0.8)
    plt.plot(large_bump_duration, [int(net.size_E * 3 / 4), int(net.size_E * 3 / 4)], label='input peak',
             color='red', linewidth=2.0, alpha=0.8)

    # # raster plot on Ip
    # fig.add_subplot(gs[1:2, 0])
    # bp.visualize.raster_plot(runner.mon.ts, runner.mon['Ip.spike'], markersize=1.)

    # # current plot for two Es
    # for i in range(2):
    #     fig.add_subplot(gs[2 + i:3 + i, 0])
    #     neuron_index = neuron_indices[i]
    #     Ec_inp = runner.mon['E2E_s.g']
    #     Fc_inp = E_inp
    #     shunting_inp = net.shunting_k * (runner.mon['E2E_s.g'] + Fc_inp) * runner.mon['I2E_s.g']
    #     Ic_inp = runner.mon['I2E_s.g']
    #     total_inp = Ec_inp + Ic_inp + Fc_inp + shunting_inp
    #     bp.visualize.line_plot(runner.mon.ts, (Ec_inp)[:, neuron_index], legend='rec_E', alpha=0.5)
    #     bp.visualize.line_plot(runner.mon.ts, (Fc_inp)[:, neuron_index], legend='F', alpha=0.5)
    #     bp.visualize.line_plot(runner.mon.ts, Ic_inp[:, neuron_index], legend='I', alpha=0.5)
    #     bp.visualize.line_plot(runner.mon.ts, shunting_inp[:, neuron_index], legend='shunting_inp', alpha=0.5)
    #     bp.visualize.line_plot(runner.mon.ts, total_inp[:, neuron_index], legend='Total', alpha=0.5)
    #     plt.grid('on')

    plt.show()


def tracking_input_protocol(runner, net, E_inp, input_duration):
    fig, gs = bp.visualize.get_figure(2, 1, 1.5, 10)
    # raster E plot
    fig.add_subplot(gs[:2, 0])
    bp.visualize.raster_plot(runner.mon.ts, runner.mon['E.spike'], markersize=1.)
    plt.plot(bm.arange(input_duration[1]), bm.argmax(E_inp[::100], axis=1), label='input peak', color='red',
             alpha=0.35, marker='.', markersize=1.5, linestyle='None')

    # # current plots for debug
    # fig.add_subplot(gs[2:4, 0])
    # neuron_index = 510
    # Ec_inp = runner.mon['E2E_s.g']
    # Fc_inp = E_inp
    # shunting_inp = net.shunting_k * (runner.mon['E2E_s.g'] + Fc_inp) * runner.mon['I2E_s.g']
    # Ic_inp = runner.mon['I2E_s.g']
    # leak = net.E.gl * runner.mon['E.V']
    # total_inp = Ec_inp + Ic_inp + Fc_inp + shunting_inp + leak
    # bp.visualize.line_plot(runner.mon.ts, (Ec_inp)[:, neuron_index], legend='rec_E', alpha=0.5)
    # bp.visualize.line_plot(runner.mon.ts, (Fc_inp)[:, neuron_index], legend='F', alpha=0.5)
    # bp.visualize.line_plot(runner.mon.ts, Ic_inp[:, neuron_index], legend='I', alpha=0.5)
    # bp.visualize.line_plot(runner.mon.ts, shunting_inp[:, neuron_index], legend='shunting_inp', alpha=0.5)
    # bp.visualize.line_plot(runner.mon.ts, leak[:, neuron_index], legend='leak', alpha=0.5)
    # bp.visualize.line_plot(runner.mon.ts, total_inp[:, neuron_index], legend='Total', alpha=0.5)
    # plt.grid('on')


    plt.show()


def compare_speed_input_protocol(runner, net, E_inp, duration, input_duration):
    fig, gs = bp.visualize.get_figure(2, 1, 1.5, 5)
    # raster plot on E
    fig.add_subplot(gs[:1, 0])
    bp.visualize.raster_plot(runner.mon.ts, runner.mon['E.spike'], markersize=1., alpha=0.5)
    plt.plot(input_duration, [int(net.size_E / 2), int(net.size_E / 2)], label='input peak', color='red')
    plt.xlim([0, duration])

    # PPC readout normalized
    fig.add_subplot(gs[1:2, 0])
    T = 100  # 1 ms
    ts = moving_average(runner.mon.ts, n=T, axis=0)
    neuron_index = int(net.size_E/2)
    x = bm.linspace(-bm.pi, bm.pi, net.size_E)
    ma = moving_average(runner.mon['E.spike'], n=T, axis=0)  # average window: 1 ms
    bump_activity = bm.vstack([bm.sum(ma * bm.cos(x[None,]), axis=1), bm.sum(ma * bm.sin(x[None,]), axis=1)])
    readout = bm.array([[1., 0.]]) @ bump_activity
    nm_readout = readout.T / bm.mean(readout[int(input_duration[0]/runner.dt*1.5):int(input_duration[1]/runner.dt)])
    ma_input = moving_average(E_inp[:, neuron_index], n=T, axis=0)
    fitted = moving_average(nm_readout, n=T * 10, axis=0)
    plt.plot(ts, nm_readout, label='projection', alpha=0.5)
    plt.plot(ts[T*10-1:], fitted, marker='.', markersize=2., linestyle='None', label='average')
    plt.plot(ts, ma_input / bm.max(ma_input), color='black', linewidth=2., label='input')
    plt.legend(loc='upper left')
    plt.show()


def compare_current_input_protocol(runner, net, E_inp, duration, input_duration, neuron_index):
    fig, gs = bp.visualize.get_figure(2, 1, 2, 7)
    # raster E plot
    fig.add_subplot(gs[0, 0])
    bp.visualize.raster_plot(runner.mon.ts, runner.mon['E.spike'], markersize=1., alpha=0.2)
    plt.xlim([0, duration])
    # plot current on central E
    fig.add_subplot(gs[1, 0])
    Ec_inp = runner.mon['E2E_s.g']
    Fc_inp = E_inp
    shunting_inp = net.shunting_k * (runner.mon['E2E_s.g'] + Fc_inp) * runner.mon['I2E_s.g']
    leak = net.E.gl * runner.mon['E.V']
    Ic_inp = runner.mon['I2E_s.g']
    total_inp = Ec_inp + Ic_inp + Fc_inp + shunting_inp + leak
    bp.visualize.line_plot(runner.mon.ts, (Fc_inp+Ec_inp)[:, neuron_index], legend='E', alpha=0.5)
    bp.visualize.line_plot(runner.mon.ts, (Ic_inp+shunting_inp+leak)[:, neuron_index], legend='I', alpha=0.5)
    bp.visualize.line_plot(runner.mon.ts, total_inp[:, neuron_index], legend='Total', alpha=1.0)
    # bp.visualize.line_plot(runner.mon.ts, Fc_inp[:, neuron_index], legend='Feedforward', linestyle='--',alpha=0.5)
    plt.ylabel(f"Neuron {neuron_index}")
    plt.legend(loc='upper right')
    plt.grid('on')
    plt.xlim([0, duration])
    plt.tight_layout()
    plt.show()


def compare_noise_sensitivity_input_protocol(runner, net, E_inp, duration):
    fig, gs = bp.visualize.get_figure(2, 2, 2, 4)
    # raster plot on E
    fig.add_subplot(gs[0, 0])
    bp.visualize.raster_plot(runner.mon.ts, runner.mon['E.spike'], markersize=1., alpha=0.2)
    # plot a stimulus section
    fig.add_subplot(gs[0, 1])
    bp.visualize.line_plot(bm.arange(net.size_E), E_inp[int(duration * 0.8) * 100,])
    plt.xlabel('Neuron Index')
    plt.ylabel('Input Strength')
    # calculate mean squared error
    fig.add_subplot(gs[1, 0])
    T = 1000  # average window: 10 ms
    x = bm.linspace(-bm.pi, bm.pi, net.size_E)
    ma = moving_average(runner.mon['E.spike'], n=T, axis=0)
    ts = moving_average(runner.mon.ts, n=T, axis=0)
    bump_activity = bm.vstack([bm.sum(ma * bm.cos(x[None,]), axis=1), bm.sum(ma * bm.sin(x[None,]), axis=1)])
    readout = get_pos_from_tan(bump_activity[1], bump_activity[0])
    plt.plot(ts, readout, marker='.', markersize=2., linestyle='None', alpha=0.5)
    plt.ylim([-bm.pi / 2, bm.pi / 2])
    mse = bm.nanmean((readout[100000:] - 0.) ** 2)
    print("Decoding MSE:", mse)
    plt.xlabel('Time (ms)')
    plt.ylabel('Population Vector Angle')
    # todo: subplot(1,1) for decoding error on different noise strength
    plt.tight_layout()


def sudden_change_stimulus(runner, net, E_inp, input_duration):
    fig, gs = bp.visualize.get_figure(2, 1, 1.5, 8)
    # raster plot on E
    fig.add_subplot(gs[:1, 0])
    bp.visualize.raster_plot(runner.mon.ts, runner.mon['E.spike'], markersize=1., alpha=0.5)
    # calculate mean squared error
    fig.add_subplot(gs[1:2, 0])
    T = 500  # average window: 5 ms
    x = bm.linspace(-bm.pi, bm.pi, net.size_E)
    ts = moving_average(runner.mon.ts, n=T, axis=0)
    ma = moving_average(runner.mon['E.spike'], n=T, axis=0)

    bump_activity = bm.vstack([bm.sum(ma * bm.cos(x[None, ]), axis=1),
                               bm.sum(ma * bm.sin(x[None, ]), axis=1)])
    bump_pos = get_pos_from_tan(bump_activity[1], bump_activity[0])

    ma_inp = moving_average(E_inp, n=T, axis=0)
    input_activity = bm.vstack([bm.sum(ma_inp * bm.cos(x[None, ]), axis=1),
                                bm.sum(ma_inp * bm.sin(x[None, ]), axis=1)])
    input_pos = get_pos_from_tan(input_activity[1], input_activity[0])

    plt.plot(ts, bump_pos, alpha=0.5)
    plt.plot(ts, input_pos, alpha=0.5)

    # np.savez('cann_sudden_change.npz', ts=ts, bump_pos=bump_pos, input_pos=input_pos)

    return


def smooth_moving_stimulus(runner, net, E_inp, input_duration):
    fig, gs = bp.visualize.get_figure(3, 1, 1.5, 10)
    # raster plot on E
    fig.add_subplot(gs[0, 0])
    bp.visualize.raster_plot(runner.mon.ts, runner.mon['E.spike'], markersize=1., alpha=0.5)
    # calculate mean squared error
    fig.add_subplot(gs[1, 0])
    T = 500  # average window: 5 ms
    x = bm.linspace(-bm.pi, bm.pi, net.size_E)
    ts = moving_average(runner.mon.ts, n=T, axis=0)
    ma = moving_average(runner.mon['E.spike'], n=T, axis=0)

    bump_activity = bm.vstack([bm.sum(ma * bm.cos(x[None,]), axis=1),
                               bm.sum(ma * bm.sin(x[None,]), axis=1)])
    bump_pos = get_pos_from_tan(bump_activity[1], bump_activity[0])

    ma_inp = moving_average(E_inp, n=T, axis=0)
    input_activity = bm.vstack([bm.sum(ma_inp * bm.cos(x[None,]), axis=1),
                                bm.sum(ma_inp * bm.sin(x[None,]), axis=1)])
    input_pos = get_pos_from_tan(input_activity[1], input_activity[0])

    plt.plot(ts, bump_pos, alpha=0.5, label='response')
    plt.plot(ts, input_pos, alpha=0.5, label='input')
    plt.legend()

    # calculate lag
    fig.add_subplot(gs[2, 0])
    lag = bm.minimum(bm.abs(input_pos-bump_pos), 2*bm.pi - bm.abs(input_pos-bump_pos))
    plt.plot(ts, lag)

    np.savez('cann_tracking_lag.npz', ts=ts, lag=lag)

    return


vis_setup = {
    "background_input": partial(background_input_protocol,),
    "persistent_input": partial(persistent_input_protocol, duration=4000., input_duration=(500, 2000), neuron_indices=(375, 100)),
    "noisy_input": partial(noisy_input_protocol, duration=3000., input_duration=(500, 3000)),
    "global_inhibition": partial(global_inhibition_protocol, small_bump_duration=(500, 2700), large_bump_duration=(1600, 2700), neuron_indices=[187, 563]),
    "tracking_input": partial(tracking_input_protocol, input_duration=(0, 3000)),
    "compare_speed_input": partial(compare_speed_input_protocol, duration=2500., input_duration=(500, 2500)),
    "compare_current_input": partial(compare_current_input_protocol, duration=3000., input_duration=(1000, 3000), neuron_index=700),
    "compare_noise_sensitivity_input": partial(compare_noise_sensitivity_input_protocol, duration=3000.),
    "sudden_change_stimulus_converge": partial(sudden_change_stimulus, input_duration=(300, 300)),
    "smooth_moving_stimulus_lag": partial(smooth_moving_stimulus, input_duration=(0, 3000)),
}