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
    axes = fig.add_subplot(gs[:1, 0])
    # only visualize a portion of E neurons to avoid heavy drawing
    size_E = net.size_E
    n_skip = bm.maximum(size_E // 750, 1).item()
    E_spike = runner.mon['E.spike'][:, ::n_skip]
    bp.visualize.raster_plot(runner.mon.ts, E_spike, xlim=[0, duration], markersize=1., alpha=0.3)
    plt.plot(input_duration, [int(net.size_E / 2 / n_skip), int(net.size_E / 2 / n_skip)], label='input peak',
             color='red')
    # set ticks with correct scaling
    ytick_list = axes.get_yticks().tolist()
    ytick_label = [int(ytick_value*n_skip) for ytick_value in ytick_list]
    axes.set_yticklabels(ytick_label)

    # raster plot on Ip
    # fig.add_subplot(gs[1:2, 0])
    # bp.visualize.raster_plot(runner.mon.ts, runner.mon['Ip.spike'], markersize=1.)
    # current plot for center E and peripheral E

    # for i in range(1):
    #     fig.add_subplot(gs[1 + 2 * i:3 + 2 * i, 0])
    #     neuron_index = int(neuron_indices[i] * net.size_E)
    #
    #     Ec_inp = runner.mon['E2E_s.g'][:, neuron_index] + runner.mon['E2E_f.g'][:, neuron_index]
    #     Fc_inp = E_inp[:, neuron_index]
    #     # shunting_inp = net.shunting_k * (Ec_inp + Fc_inp) * runner.mon['I2E_s.g'][:, neuron_index]
    #     # Ic_inp = runner.mon['I2E_s.g'][:, neuron_index] + runner.mon['I2E_f.g'][:, neuron_index]
    #     leak = net.E.gl * runner.mon['E.V'][:, neuron_index]
    #     # total_inp = Ec_inp + Ic_inp + Fc_inp + shunting_inp + leak
    #     # bp.visualize.line_plot(runner.mon.ts, Ec_inp, legend='rec_E', alpha=0.5)
    #     # bp.visualize.line_plot(runner.mon.ts, Fc_inp, legend='F', alpha=0.5)
    #     # bp.visualize.line_plot(runner.mon.ts, Ic_inp, legend='rec_I', alpha=0.5)
    #     # bp.visualize.line_plot(runner.mon.ts, shunting_inp, legend='shunting_inp', alpha=0.5)
    #     # bp.visualize.line_plot(runner.mon.ts, total_inp, legend='Total', alpha=0.5)
    #     # bp.visualize.line_plot(runner.mon.ts, leak, legend='leak', alpha=0.5)
    #
    #     EI_inp = runner.mon['E2E_f.g'][:, neuron_index] + runner.mon['I2E_f.g'][:, neuron_index]
    #     shunting_inp = net.shunting_k * (Ec_inp + Fc_inp) * runner.mon['I2E_s.g'][:, neuron_index]
    #     CA_inp = runner.mon['E2E_s.g'][:, neuron_index] + shunting_inp + runner.mon['I2E_s.g'][:, neuron_index]
    #     CA_effective_inp = EI_inp + Fc_inp
    #     total_inp = EI_inp + CA_inp + leak + Fc_inp
    #     bp.visualize.line_plot(runner.mon.ts, EI_inp, legend='EI_inp', alpha=0.5)
    #     bp.visualize.line_plot(runner.mon.ts, CA_inp, legend='CA_inp', alpha=0.5)
    #     bp.visualize.line_plot(runner.mon.ts, CA_effective_inp, legend='CA_effective_inp', alpha=1.0)
    #     bp.visualize.line_plot(runner.mon.ts, total_inp, legend='total_inp', alpha=0.5)
    #
    #     plt.legend(loc=4)
    #     plt.grid('on')

    plt.show()


def check_balance_input_protocol(runner, net, E_inp, duration, input_duration, neuron_indices):
    fig, gs = bp.visualize.get_figure(2, 1, 3, 6)
    # current plot for center Ip
    # fig.add_subplot(gs[:1, 0])
    # bp.visualize.raster_plot(runner.mon.ts, runner.mon['E.spike'], markersize=1.)

    for i in range(2):
        fig.add_subplot(gs[i, 0])
        Ec_inp = runner.mon['E2E_s.g'][-55000:, neuron_indices[i]] + runner.mon['E2E_f.g'][-55000:, neuron_indices[i]]
        Fc_inp = E_inp[-55000:, neuron_indices[i]]
        shunting_inp = net.shunting_k * (Ec_inp + Fc_inp) * runner.mon['I2E_s.g'][-55000:, neuron_indices[i]]
        Ic_inp = runner.mon['I2E_s.g'][-55000:, neuron_indices[i]] + runner.mon['I2E_f.g'][-55000:, neuron_indices[i]] + net.E.gl * runner.mon['E.V'][-55000:, neuron_indices[i]]
        total_inp = Ec_inp + Ic_inp + Fc_inp + shunting_inp

        print(f"{neuron_indices[i]}")
        print(f"Total E {bm.mean(Ec_inp + Fc_inp):.5f}")
        print(f"Total I {bm.mean(Ic_inp + shunting_inp):.5f}")
        print(f"Total input {bm.mean(total_inp):.5f}")

        total_E = runner.mon['E2E_s.g'][:, neuron_indices[i]] + E_inp[:, neuron_indices[i]] + runner.mon['E2E_f.g'][:, neuron_indices[i]]
        part_I = runner.mon['I2E_s.g'][:, neuron_indices[i]] + runner.mon['I2E_f.g'][:, neuron_indices[i]] + net.E.gl * runner.mon['E.V'][:, neuron_indices[i]]
        total_shunting = net.shunting_k * total_E * runner.mon['I2E_s.g'][:, neuron_indices[i]]
        total_I = part_I + total_shunting
        bp.visualize.line_plot(runner.mon.ts, total_E, legend='E', alpha=0.5)
        bp.visualize.line_plot(runner.mon.ts, total_I , legend='I', alpha=0.5)
        bp.visualize.line_plot(runner.mon.ts, total_E+total_I, legend='Total', alpha=0.5)

    plt.show()


def check_balance_flat_input_protocol(runner, net, E_inp, duration, input_duration, neuron_indices):
    fig, gs = bp.visualize.get_figure(3, 1, 1.5, 5)
    # current plot for center Ip
    fig.add_subplot(gs[:1, 0])
    bp.visualize.raster_plot(runner.mon.ts, runner.mon['E.spike'], markersize=1.)
    # plt.plot(input_duration, [int(net.size_E / 2), int(net.size_E / 2)], label='input peak', color='red')

    # current plot for center E
    fig.add_subplot(gs[1:2, 0])
    neuron_index = neuron_indices[0]
    Ec_inp = runner.mon['E2E_s.g'] + runner.mon['E2E_f.g']
    Fc_inp = E_inp
    shunting_inp = net.shunting_k * (Ec_inp + Fc_inp) * runner.mon['I2E_s.g']
    Ic_inp = runner.mon['I2E_s.g'] + runner.mon['I2E_f.g']
    total_inp = Ec_inp + Ic_inp + Fc_inp + shunting_inp
    bp.visualize.line_plot(runner.mon.ts, (Ec_inp+Fc_inp)[:, neuron_index], legend='E', alpha=0.5)
    bp.visualize.line_plot(runner.mon.ts, (Ic_inp+shunting_inp)[:, neuron_index], legend='I', alpha=0.5)
    bp.visualize.line_plot(runner.mon.ts, total_inp[:, neuron_index], legend='Total', alpha=0.5)

    # fig.axes[1].yaxis.set_ticks([0])
    plt.legend(loc=4)
    plt.grid('on')
    plt.ylabel(f'Neuron {neuron_index}')
    print(f"Total input {bm.mean(total_inp[-30000:, neuron_index]):.5f}")

    # current plot for peripheral E
    fig.add_subplot(gs[2:3, 0])
    neuron_index = neuron_indices[1]
    Ec_inp = runner.mon['E2E_s.g'] + runner.mon['E2E_f.g']
    Fc_inp = E_inp
    shunting_inp = net.shunting_k * (Ec_inp + Fc_inp) * runner.mon['I2E_s.g']
    Ic_inp = runner.mon['I2E_s.g'] + runner.mon['I2E_f.g']
    total_inp = Ec_inp + Ic_inp + Fc_inp + shunting_inp
    bp.visualize.line_plot(runner.mon.ts, (Ec_inp+Fc_inp)[:, neuron_index], legend='E', alpha=0.5)
    bp.visualize.line_plot(runner.mon.ts, (Ic_inp+shunting_inp)[:, neuron_index], legend='I', alpha=0.5)
    bp.visualize.line_plot(runner.mon.ts, total_inp[:, neuron_index], legend='Total', alpha=0.5)

    # fig.axes[2].yaxis.set_ticks([0])
    plt.legend(loc=4)
    plt.grid('on')
    plt.ylabel(f'Neuron {neuron_index}')

    plt.show()


def noisy_input_protocol(runner, net, E_inp, duration, input_duration):
    fig, gs = bp.visualize.get_figure(5, 1, 1.5, 10)
    # raster plot on E
    fig.add_subplot(gs[:1, 0])
    bp.visualize.raster_plot(runner.mon.ts, runner.mon['E.spike'], markersize=1.)
    plt.plot(input_duration, [int(net.size_E / 2), int(net.size_E / 2)], label='input peak', color='red')
    # raster plot on Ip
    fig.add_subplot(gs[1:2, 0])
    bp.visualize.raster_plot(runner.mon.ts, runner.mon['Ip.spike'], markersize=1.)
    # imshow input
    fig.add_subplot(gs[2:3, 0])
    plt.imshow(E_inp.T, aspect='auto')
    # input section
    fig.add_subplot(gs[3:4, 0])
    bp.visualize.line_plot(bm.arange(net.size_E), E_inp[int(duration*0.8)*100, ], legend='input')
    plt.legend()
    # imshow E spike
    fig.add_subplot(gs[4:5, 0])
    plt.imshow(runner.mon['E.spike'].T, aspect='auto')
    plt.show()


def global_inhibition_protocol(runner, net, E_inp, small_bump_duration, large_bump_duration, neuron_indices):
    fig, gs = bp.visualize.get_figure(4, 1, 1.5, 10)
    # raster plot on E
    fig.add_subplot(gs[:1, 0])
    bp.visualize.raster_plot(runner.mon.ts, runner.mon['E.spike'], markersize=1.)
    plt.plot(small_bump_duration, [int(net.size_E / 4), int(net.size_E / 4)], label='input peak', color='red', alpha=0.5)
    plt.plot(large_bump_duration, [int(net.size_E * 3 / 4), int(net.size_E * 3 / 4)], label='input peak', color='red', alpha=0.5)

    # raster plot on Ip
    fig.add_subplot(gs[1:2, 0])
    bp.visualize.raster_plot(runner.mon.ts, runner.mon['Ip.spike'], markersize=1.)

    # current plot for two Es
    for i in range(2):
        fig.add_subplot(gs[2 + i:3 + i, 0])
        neuron_index = neuron_indices[i]
        Ec_inp = runner.mon['E2E_s.g']
        Fc_inp = E_inp
        shunting_inp = net.shunting_k * (runner.mon['E2E_s.g'] + Fc_inp) * runner.mon['I2E_s.g']
        Ic_inp = runner.mon['I2E_s.g']
        total_inp = Ec_inp + Ic_inp + Fc_inp
        bp.visualize.line_plot(runner.mon.ts, (Ec_inp)[:, neuron_index], legend='rec_E', alpha=0.5)
        bp.visualize.line_plot(runner.mon.ts, (Fc_inp)[:, neuron_index], legend='F', alpha=0.5)
        bp.visualize.line_plot(runner.mon.ts, Ic_inp[:, neuron_index], legend='I', alpha=0.5)
        bp.visualize.line_plot(runner.mon.ts, shunting_inp[:, neuron_index], legend='shunting_inp', alpha=0.5)
        bp.visualize.line_plot(runner.mon.ts, total_inp[:, neuron_index], legend='Total', alpha=0.5)
        plt.grid('on')

    plt.show()


def tracking_input_protocol(runner, net, E_inp, input_duration):
    fig, gs = bp.visualize.get_figure(2, 1, 1.5, 10)
    # raster E plot
    fig.add_subplot(gs[:2, 0])
    bp.visualize.raster_plot(runner.mon.ts, runner.mon['E.spike'], markersize=1.)
    plt.plot(bm.arange(input_duration[1]), bm.argmax(E_inp[::100], axis=1), label='input peak', color='red',
             alpha=0.5, marker='.', markersize=1.5, linestyle='None')
    plt.show()


def compare_speed_input_protocol(runner, net, E_inp, duration, input_duration):
    fig, gs = bp.visualize.get_figure(2, 1, 1.7, 6)
    # raster plot on E
    ax = fig.add_subplot(gs[:1, 0])
    bp.visualize.raster_plot(runner.mon.ts, runner.mon['E.spike'], markersize=1., alpha=0.5)
    ax.plot(input_duration, [int(net.size_E / 2), int(net.size_E / 2)],
             label='input peak', color='white', linestyle='--', linewidth=1.5)
    ax.set_xlim([0, duration])
    ax.set_ylim([0, net.size_E])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels([])
    ax.set_yticks([0, 200, 400, 600, 800])
    ax.set_yticklabels([])

    # PPC readout normalized
    ax = fig.add_subplot(gs[1:2, 0])
    T = 100  # 1 ms
    ts = moving_average(runner.mon.ts, n=T, axis=0)
    neuron_index = int(net.size_E/2)
    x = bm.linspace(-bm.pi, bm.pi, net.size_E)
    ma = moving_average(runner.mon['E.spike'], n=T, axis=0)  # average window: 1 ms
    bump_activity = bm.vstack([bm.sum(ma * bm.cos(x[None, ]), axis=1), bm.sum(ma * bm.sin(x[None, ]), axis=1)])
    readout = bm.array([[1., 0.]]) @ bump_activity
    nm_readout = readout.T / bm.mean(readout[:, int(input_duration[0]/runner.dt*1.5):int(input_duration[1]/runner.dt)])
    ma_input = moving_average(E_inp[:, neuron_index], n=T, axis=0)
    ma_input = ma_input - bm.min(ma_input)
    fitted = moving_average(nm_readout, n=T * 10, axis=0)
    ax.plot(ts, nm_readout, label='projection', alpha=0.5)
    ax.plot(ts[T*10-1:], fitted, marker='.', markersize=2., linestyle='None', label='average')
    ax.plot(ts, ma_input / bm.max(ma_input), color='black', linewidth=2., label='input')
    ax.set_xlim([0, duration])
    ax.set_xlabel("Time (ms)")
    ax.set_ylim([-0.8, 2])
    ax.set_yticklabels([])
    ax.grid()
    # ax.set_ylabel("Readout")

    plt.legend(loc='upper right', bbox_to_anchor=(1.4, 1.0), fancybox=True, shadow=True)
    plt.tight_layout()
    plt.show()


def compare_current_input_protocol(runner, net, E_inp, duration, input_duration, neuron_index):
    fig, gs = bp.visualize.get_figure(2, 1, 1.5, 5)
    # subplot 1: raster E plot
    ax = fig.add_subplot(gs[0, 0])
    bp.visualize.raster_plot(runner.mon.ts, runner.mon['E.spike'], markersize=1., alpha=0.2)
    ax.set_xlim([500, 2000])
    ax.set_ylim([0, net.size_E])
    ax.set_yticks([0, 400, 800])

    # subplot 2: plot current on central E
    ax1 = fig.add_subplot(gs[1, 0])
    E2E_s = runner.mon['E2E_s.g'][:, neuron_index]
    E2E_f = runner.mon['E2E_f.g'][:, neuron_index]
    I2E_s = runner.mon['I2E_s.g'][:, neuron_index]
    I2E_f = runner.mon['I2E_f.g'][:, neuron_index]

    Ec_inp = E2E_s + E2E_f
    Fc_inp = E_inp[:, neuron_index]
    shunting_inp = net.shunting_k * (E2E_s + E2E_f + Fc_inp) * I2E_s
    leak = net.E.gl * runner.mon['E.V'][:, neuron_index]
    Ic_inp = I2E_s + I2E_f
    total_inp = Ec_inp + Ic_inp + Fc_inp + shunting_inp + leak
    # smooth lines
    T = 5000  # 50 ms
    ts = moving_average(runner.mon.ts, n=T, axis=0)
    total_E = moving_average((Fc_inp + Ec_inp), n=T, axis=0)
    total_I = moving_average((Ic_inp + shunting_inp + leak), n=T, axis=0)
    total = moving_average(total_inp, n=T, axis=0)

    ax1.plot(ts, total_E, label='E', color='blue', linewidth=2, linestyle='--')
    ax1.plot(ts, total_I, label='I', color='gray', linewidth=2, linestyle='--')
    ax1.plot(ts, total, label='Total', color='black', linewidth=2, linestyle='-')
    ax1.set_ylabel(f"Neuron {neuron_index}")

    plt.legend(loc='upper left')
    ax1.grid('on')
    plt.xlim([500, 2000])
    # plt.ylim([-2, 2])

    plt.tight_layout()
    plt.show()


def compare_noise_sensitivity_input_protocol(runner, net, E_inp, duration):
    # fig, gs = bp.visualize.get_figure(3, 1, 1.5, 8)
    # # raster plot on E
    # fig.add_subplot(gs[:1, 0])
    # bp.visualize.raster_plot(runner.mon.ts, runner.mon['E.spike'], markersize=1., alpha=0.5)
    # # plot a section of input
    # fig.add_subplot(gs[1:2, 0])
    # bp.visualize.line_plot(bm.arange(net.size_E), E_inp[int(duration * 0.8) * 100,], legend='input')
    # plt.legend()
    # # calculate mean squared error
    # fig.add_subplot(gs[2:3, 0])
    T = 100  # average window: 1 ms
    x = bm.linspace(-bm.pi, bm.pi, net.size_E)
    ts = moving_average(runner.mon.ts, n=T, axis=0)
    ma = moving_average(runner.mon['E.spike'], n=T, axis=0)
    bump_activity = bm.vstack([bm.sum(ma * bm.cos(x[None,]), axis=1), bm.sum(ma * bm.sin(x[None,]), axis=1)])
    readout = get_pos_from_tan(bump_activity[1], bump_activity[0])
    # plt.plot(ts, readout, marker='.', markersize=2., linestyle='None', alpha=0.5)
    # plt.ylim([-bm.pi/2, bm.pi/2])
    mse = bm.nanmean((readout[100000:] - 0.) ** 2)
    print("Slow CANN + EI balance Decoding MSE:", mse)
    # calculate mean firing rate
    # center_fr = bm.mean(runner.mon['E.spike'][:, int(net.size_E / 2) - 10:int(net.size_E / 2) + 10])
    # peripheral_fr = bm.mean(runner.mon['E.spike'][:, int(net.size_E * 0.1) - 10:int(net.size_E * 0.1) + 10])
    # print(f"center firing rate {center_fr:.6f}")
    # print(f"peripheral firing rate {peripheral_fr:.6f}")


def sudden_change_stimulus(runner, net, E_inp, input_duration):
    # fig, gs = bp.visualize.get_figure(2, 1, 1.5, 8)
    # # raster plot on E
    # fig.add_subplot(gs[:1, 0])
    # bp.visualize.raster_plot(runner.mon.ts, runner.mon['E.spike'], markersize=1., alpha=0.5)
    # # calculate mean squared error
    # fig.add_subplot(gs[1:2, 0])
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

    # plt.plot(ts, bump_pos, alpha=0.5, label='response')
    # plt.plot(ts, input_pos, alpha=0.5, label='input')
    # plt.legend()

    import uuid
    np.savez(f'coupled_sudden_change_{str(uuid.uuid4())[:5]}.npz', ts=ts, bump_pos=bump_pos, input_pos=input_pos)

    return


def smooth_moving_stimulus(runner, net, E_inp, input_duration):
    # fig, gs = bp.visualize.get_figure(3, 1, 1.5, 10)
    # # raster plot on E
    # fig.add_subplot(gs[0, 0])
    # bp.visualize.raster_plot(runner.mon.ts, runner.mon['E.spike'], markersize=1., alpha=0.5)
    # # calculate mean squared error
    # fig.add_subplot(gs[1, 0])
    T = 1000  # average window: 10 ms
    x = bm.linspace(-bm.pi, bm.pi, net.size_E)
    ts = moving_average(runner.mon.ts, n=T, axis=0)
    ma = moving_average(runner.mon['E.spike'], n=T, axis=0)

    bump_activity = bm.vstack([bm.sum(ma * bm.cos(x[None,]), axis=1),
                               bm.sum(ma * bm.sin(x[None,]), axis=1)])
    bump_pos = get_pos_from_tan(a=bump_activity[1], b=bump_activity[0])

    ma_inp = moving_average(E_inp, n=T, axis=0)
    input_activity = bm.vstack([bm.sum(ma_inp * bm.cos(x[None,]), axis=1),
                                bm.sum(ma_inp * bm.sin(x[None,]), axis=1)])
    input_pos = get_pos_from_tan(input_activity[1], input_activity[0])

    # plt.plot(ts, bump_pos, alpha=0.5, label='response')
    # plt.plot(ts, input_pos, alpha=0.5, label='input')
    # plt.legend()

    # calculate lag
    # fig.add_subplot(gs[2, 0])
    lag = input_pos - bump_pos
    lag[lag > bm.pi] = lag[lag > bm.pi] - 2 * bm.pi
    lag[lag < -bm.pi] = lag[lag < -bm.pi] + 2 * bm.pi
    # plt.plot(ts, lag)

    import uuid
    np.savez(f'coupled_tracking_lag_{str(uuid.uuid4())[:5]}.npz', ts=ts, lag=lag)

    return


vis_setup = {
    "background_input": partial(background_input_protocol,),
    "persistent_input": partial(persistent_input_protocol, duration=2800., input_duration=(500, 2000), neuron_indices=(0.5, 0.1)),
    "check_balance_input": partial(check_balance_input_protocol, duration=1500., input_duration=(500, 1500), neuron_indices=[int(400*1), int(100*1)]),
    "check_balance_flat_input": partial(check_balance_flat_input_protocol, duration=1500., input_duration=(500, 1500), neuron_indices=[375, 125]),
    "noisy_input": partial(noisy_input_protocol, duration=3000., input_duration=(500, 3000)),
    "global_inhibition": partial(global_inhibition_protocol, small_bump_duration=(500, 4200), large_bump_duration=(1700, 4200), neuron_indices=[187, 563]),
    "tracking_input": partial(tracking_input_protocol, input_duration=(0, 3000)),
    "compare_speed_input": partial(compare_speed_input_protocol, duration=1500., input_duration=(500, 1500)),
    "compare_current_input": partial(compare_current_input_protocol, duration=3000., input_duration=(500, 3000), neuron_index=700),
    "compare_noise_sensitivity_input": partial(compare_noise_sensitivity_input_protocol, duration=3000.),
    "sudden_change_stimulus_converge": partial(sudden_change_stimulus, input_duration=(300, 300)),
    "smooth_moving_stimulus_lag": partial(smooth_moving_stimulus, input_duration=(0, 3000)),
}