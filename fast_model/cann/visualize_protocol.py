import brainpy as bp
import brainpy.math as bm
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

def calculate_population_readout(activity, T):
    size = activity.shape[1]
    x = bm.linspace(-bm.pi, bm.pi, size)
    ma = moving_average(activity, n=T, axis=0)  # average window: 1 ms
    bump_activity = bm.vstack([bm.sum(ma * bm.cos(x[None, ]), axis=1), bm.sum(ma * bm.sin(x[None, ]), axis=1)])
    readout = bm.array([[1., 0.]]) @ bump_activity
    return readout


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
    fig, gs = bp.visualize.get_figure(2, 1, 1.5, 6)
    # subplot 1: raster plot on E
    axes = fig.add_subplot(gs[:1, 0])
    bp.visualize.raster_plot(runner.mon.ts, runner.mon['E.spike'], xlim=[0, duration], markersize=1., alpha=0.3)
    # plt.plot(input_duration, [int(net.size_E/2), int(net.size_E/2)], label='input peak', color='red')
    plt.plot((input_duration[0], input_duration[0]), (0, net.size_E), color='gray', linestyle='--', linewidth=2.)
    plt.plot((input_duration[1], input_duration[1]), (0, net.size_E), color='gray', linestyle='--', linewidth=2.)
    plt.xlabel("")
    plt.ylim([0, net.size_E])
    axes.set_xticklabels([])

    # subplot 2: readout plot
    fig.add_subplot(gs[1:2, 0])
    T = 100  # 1 ms
    ts = moving_average(runner.mon.ts, n=T, axis=0)
    readout = calculate_population_readout(activity=runner.mon['E.spike'], T=T)
    fitted = moving_average(readout.T, n=T * 10, axis=0)
    nm_readout = readout.T / bm.max(fitted)
    fitted = fitted / bm.max(fitted)
    plt.plot(ts, nm_readout, label='readout', alpha=0.5, color="gray")
    plt.plot(ts[T * 10 - 1:], fitted, marker='.', markersize=2., linestyle='None', label='average', color="black")
    plt.plot((input_duration[0], input_duration[0]), (bm.min(nm_readout), bm.max(nm_readout)),
             color='gray', linestyle='--', linewidth=2.)
    plt.plot((input_duration[1], input_duration[1]), (bm.min(nm_readout), bm.max(nm_readout)),
             color='gray', linestyle='--', linewidth=2.)
    plt.legend(loc='upper right')
    plt.grid('on')
    plt.xlim([0, duration])
    plt.xlabel("Time (ms)")
    plt.ylabel("Readout")

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
             linewidth=0.5, alpha=0.8)
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
    fig, gs = bp.visualize.get_figure(4, 1, 1.5, 10)
    # raster E plot
    fig.add_subplot(gs[:2, 0])
    bp.visualize.raster_plot(runner.mon.ts, runner.mon['E.spike'], markersize=1.)
    plt.plot(bm.arange(input_duration[1]), bm.argmax(E_inp[::100], axis=1), label='input peak', color='red',
             alpha=0.5, marker='.', markersize=1.5, linestyle='None')

    # current plots for debug
    fig.add_subplot(gs[2:4, 0])
    neuron_index = 510
    Ec_inp = runner.mon['E2E_s.g']
    Fc_inp = E_inp
    shunting_inp = net.shunting_k * (runner.mon['E2E_s.g'] + Fc_inp) * runner.mon['I2E_s.g']
    Ic_inp = runner.mon['I2E_s.g']
    leak = net.E.gl * runner.mon['E.V']
    total_inp = Ec_inp + Ic_inp + Fc_inp + shunting_inp + leak
    bp.visualize.line_plot(runner.mon.ts, (Ec_inp)[:, neuron_index], legend='rec_E', alpha=0.5)
    bp.visualize.line_plot(runner.mon.ts, (Fc_inp)[:, neuron_index], legend='F', alpha=0.5)
    bp.visualize.line_plot(runner.mon.ts, Ic_inp[:, neuron_index], legend='I', alpha=0.5)
    bp.visualize.line_plot(runner.mon.ts, shunting_inp[:, neuron_index], legend='shunting_inp', alpha=0.5)
    bp.visualize.line_plot(runner.mon.ts, leak[:, neuron_index], legend='leak', alpha=0.5)
    bp.visualize.line_plot(runner.mon.ts, total_inp[:, neuron_index], legend='Total', alpha=0.5)
    plt.grid('on')


    plt.show()


def compare_speed_input_protocol(runner, net, E_inp, duration, input_duration):
    fig, gs = bp.visualize.get_figure(2, 1, 2, 8)
    # raster plot on E
    fig.add_subplot(gs[:1, 0])
    bp.visualize.raster_plot(runner.mon.ts, runner.mon['E.spike'], markersize=1., alpha=0.5)
    plt.plot(input_duration, [int(net.size_E / 2), int(net.size_E / 2)], label='input peak', color='red')
    plt.xlim([0, 3000.])

    # PPC readout normalized
    fig.add_subplot(gs[1:2, 0])
    T = 100
    ts = moving_average(runner.mon.ts, n=T, axis=0)
    neuron_index = int(net.size_E/2)
    x = bm.linspace(-bm.pi, bm.pi, net.size_E)
    ma = moving_average(runner.mon['E.spike'], n=T, axis=0)  # average window: 1 ms
    bump_activity = bm.vstack([bm.sum(ma * bm.cos(x[None,]), axis=1), bm.sum(ma * bm.sin(x[None,]), axis=1)])
    readout = bm.array([[1., 0.]]) @ bump_activity
    nm_readout = readout.T / bm.mean(readout[:, int(input_duration[0]/runner.dt):int(input_duration[1]/runner.dt)])
    ma_input = moving_average(E_inp[:, neuron_index], n=T, axis=0)
    fitted = moving_average(nm_readout, n=T * 10, axis=0)
    plt.plot(ts, nm_readout, label='projected PPC (normalized)', alpha=0.5)
    plt.plot(ts[T*10-1:], fitted, marker='.', markersize=2., linestyle='None', label='fitted')
    plt.plot(ts, ma_input / bm.max(ma_input), color='black', linewidth=2., label='input')
    plt.legend()
    plt.show()


def compare_current_input_protocol(runner, net, E_inp, duration, input_duration, neuron_index):
    fig, gs = bp.visualize.get_figure(2, 1, 2, 5)
    # raster E plot
    fig.add_subplot(gs[:1, 0])
    bp.visualize.raster_plot(runner.mon.ts, runner.mon['E.spike'], markersize=1., alpha=0.5)
    # plot current on central E
    fig.add_subplot(gs[1:2, 0])
    Ec_inp = runner.mon['E2E_s.g']
    Fc_inp = E_inp
    shunting_inp = net.shunting_k * (runner.mon['E2E_s.g'] + Fc_inp) * runner.mon['I2E_s.g']
    Ic_inp = runner.mon['I2E_s.g']
    leak = net.E.gl * runner.mon['E.V']
    total_inp = Ec_inp + Ic_inp + Fc_inp + shunting_inp + leak
    bp.visualize.line_plot(runner.mon.ts, (Fc_inp+Ec_inp)[:, neuron_index], legend='E', alpha=0.5)
    bp.visualize.line_plot(runner.mon.ts, (Ic_inp+shunting_inp+leak)[:, neuron_index], legend='I', alpha=0.5)
    bp.visualize.line_plot(runner.mon.ts, (leak)[:, neuron_index], legend='I', alpha=0.5)
    bp.visualize.line_plot(runner.mon.ts, total_inp[:, neuron_index], legend='Total', alpha=0.5)
    plt.ylabel(f"Neuron {neuron_index}")
    plt.grid('on')
    plt.show()


def compare_noise_sensitivity_input_protocol(runner, net, E_inp, duration):
    # fig, gs = bp.visualize.get_figure(3, 1, 2, 8)
    # # raster plot on E
    # fig.add_subplot(gs[:1, 0])
    # bp.visualize.raster_plot(runner.mon.ts, runner.mon['E.spike'], markersize=1., alpha=0.5)
    # # plot a stimulus section
    # fig.add_subplot(gs[1:2, 0])
    # bp.visualize.line_plot(bm.arange(net.size_E), E_inp[int(duration * 0.8) * 100,], legend='input')
    # plt.legend()
    # # calculate mean squared error
    # fig.add_subplot(gs[2:3, 0])
    T = 100  # average window: 10 ms
    x = bm.linspace(-bm.pi, bm.pi, net.size_E)
    ma = moving_average(runner.mon['E.spike'], n=T, axis=0)
    ts = moving_average(runner.mon.ts, n=T, axis=0)
    bump_activity = bm.vstack([bm.sum(ma * bm.cos(x[None,]), axis=1), bm.sum(ma * bm.sin(x[None,]), axis=1)])
    readout = bm.arctan(bump_activity[1] / bump_activity[0])
    # plt.plot(ts, readout, marker='.', markersize=2., linestyle='None', alpha=0.2)
    # plt.ylim([-bm.pi / 2, bm.pi / 2])
    mse = bm.nanmean((readout[100000:] - 0.) ** 2)
    print("Fast CANN decoding MSE:", mse)



vis_setup = {
    "background_input": partial(background_input_protocol,),
    "persistent_input": partial(persistent_input_protocol, duration=2400., input_duration=(400, 1400), neuron_indices=(400, 50)),
    "noisy_input": partial(noisy_input_protocol, duration=3000., input_duration=(500, 3000)),
    "global_inhibition": partial(global_inhibition_protocol, small_bump_duration=(500, 2700), large_bump_duration=(1600, 2700), neuron_indices=[187, 563]),
    "tracking_input": partial(tracking_input_protocol, input_duration=(0, 3000)),
    "compare_speed_input": partial(compare_speed_input_protocol, duration=3000., input_duration=(500, 3000)),
    "compare_current_input": partial(compare_current_input_protocol, duration=3000., input_duration=(500, 3000), neuron_index=375),
    "compare_noise_sensitivity_input": partial(compare_noise_sensitivity_input_protocol, duration=3000.)
}