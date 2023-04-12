import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import brainpy as bp
import brainpy.math as bm
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from functools import partial

from utils.vis_utils import (
    calculate_population_readout, 
    moving_average, 
    get_pos_from_tan,
    calculate_spike_center,
    plot_E_currents,
    )


global_dt = 0.01


def persistent_protocol(runner, net, E_inp, duration, input_duration, neuron_indices):
    fig, gs = bp.visualize.get_figure(2, 1, 1.5, 4.)
    # subplot 1: raster plot on E
    ax1 = fig.add_subplot(gs[:1, 0])
    bp.visualize.raster_plot(runner.mon.ts, runner.mon['E.spike'], xlim=[0, duration], markersize=1., alpha=0.3)
    # plot onset and offset of the stimulus
    ax1.plot((input_duration[0], input_duration[0]), (0, net.size_E), color='blue', linestyle='--', linewidth=2.)
    ax1.plot((input_duration[1], input_duration[1]), (0, net.size_E), color='blue', linestyle='--', linewidth=2.)
    ax1.set_xlabel("")
    ax1.set_ylim([0, net.size_E])
    ax1.set_xticklabels([])

    # subplot 2: readout plot
    ax2 = fig.add_subplot(gs[1:2, 0])
    T = 200  # 2 ms
    ts = moving_average(runner.mon.ts, n=T, axis=0)
    readout = calculate_population_readout(activity=runner.mon['E.spike'], T=T)
    fitted = moving_average(readout.T, n=T * 10, axis=0)
    nm_readout = readout.T / bm.max(fitted) * 1.1
    fitted = fitted / bm.max(fitted) * 1.1
    ax2.plot(ts, nm_readout, label='readout', alpha=0.3, color="gray")
    ax2.plot(ts[T * 10 - 1:], fitted, marker='.', markersize=2., linestyle='None', label='average', color="black")
    # plot onset and offset of the stimulus
    ax2.plot((input_duration[0], input_duration[0]), (bm.min(nm_readout), bm.max(nm_readout)),
             color='blue', linestyle='--', linewidth=2.)
    ax2.plot((input_duration[1], input_duration[1]), (bm.min(nm_readout), bm.max(nm_readout)),
             color='blue', linestyle='--', linewidth=2.)
    ax2.legend(loc='upper right')
    ax2.grid('on')
    ax2.set_xlim([0, duration])
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Readout")
    ax2.set_ylim([-0.2, 1.5])

    fig.align_ylabels([ax1, ax2])
    plt.show()


def balance_check_bump_protocol(runner, net, E_inp, duration, input_duration, neuron_indices):
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


def balance_check_flat_protocol(runner, net, E_inp, duration, input_duration, neuron_indices):
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


def irregular_check_flat_protocol(runner, net, E_inp, duration, input_duration):
    # subplot 1: raster plot
    fig, gs = bp.visualize.get_figure(1, 4, 1.5, 1.5)
    ax1 = fig.add_subplot(gs[0, 0:2])
    bp.visualize.raster_plot(runner.mon.ts, runner.mon['E.spike'], xlim=(0, duration), ylim=(0, net.size_E), markersize=1., alpha=0.2)

    # # subplot 2: calculate population average autocorrelation c.f. Vreeswijk and Somoplinsky (1998)
    sample_n = 30  # sample pair number
    spiketr = runner.mon['E.spike'].T.astype(bm.float32)
    size_n, length = spiketr.shape
    index = bm.random.choice(bm.arange(size_n), sample_n, replace=False)
    X = spiketr[index, ] # shape [sample_n, T]
    autocorr = bm.zeros(length)
    for i in range(sample_n):
        autocorr += bm.correlate(X[i], X[i], mode='same')
    autocorr /= sample_n
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(runner.mon.ts-duration/2, autocorr, linewidth=2., alpha=0.8, color='k')
    ax2.set_xlim(-duration/2, duration/2)
    ax2.set_ylabel('Autocorr')
    ax2.set_xlabel('Time (ms)')
    ax2.grid()

    # subplot 3: plot spectrogram of the neural activity
    ax3 = fig.add_subplot(gs[0, 3])
    fs = 1 / global_dt * 1000
    data = np.sum(spiketr, axis=-1)
    f, Pxx_den = signal.welch(data, fs)
    ax3.semilogy(f, Pxx_den, color='black')
    ax3.set_ylim([0.5e-3, 1e-2])
    ax3.set_ylabel('PSD')
    ax3.set_xlabel('Frequency [Hz]')

    # plt.tight_layout()


def tracking_protocol(runner, net, E_inp, duration, input_duration):
    fig, gs = bp.visualize.get_figure(1, 1, 1.5, 2.5)
    # raster E plot
    ax1 = fig.add_subplot(gs[:1, 0])
    bp.visualize.raster_plot(runner.mon.ts, runner.mon['E.spike'], markersize=1., alpha=0.5)
    ax1.plot(bm.arange(input_duration[1]), bm.argmax(E_inp[::100], axis=1), label='input peak', color='blue',
             alpha=1.0, marker='.', markersize=2.5, linestyle='None')
    
    T = 1000
    activity = moving_average(runner.mon['E.spike'], n=T, axis=0)
    spike_center = calculate_spike_center(activity, net.size_E)
    ts = moving_average(runner.mon.ts, n=T, axis=-1)
    ax1.plot(ts, spike_center, color='red', alpha=1.0, linewidth=1.5)
    ax1.set_xlim([245, 455])


def convergence_rate_population_readout_protocol(runner, net, E_inp, duration, input_duration):
    fig, gs = bp.visualize.get_figure(3, 4, 1.7, 2)
    # raster plot on E
    ax = fig.add_subplot(gs[:1, :3])
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
    ax = fig.add_subplot(gs[1:2, :3])
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

    plt.legend(loc='center', bbox_to_anchor=(1.25, 1.0), fancybox=True, shadow=True)
    plt.tight_layout()
    plt.show()


def convergence_rate_current_protocol(runner, net, E_inp, duration, input_duration, neuron_index):
    fig, gs = bp.visualize.get_figure(2, 5, 1.8, 1)
    # subplot 1: plot current on central E
    ax2 = fig.add_subplot(gs[0, 0:4])
    plot_E_currents(runner, net, E_inp, neuron_index, ax2, 
                    plot_items=['total_E', 'total_I', 'total'], 
                    # plot_items=['Ec_s', 'Ec_f', 'Ic_s', 'Ic_f', 'Fc', 'shunting'], 
                    smooth_T=2500)

    ax2.grid('on')
    ax2.set_xlim([500, 2000])
    ax2.set_xlabel("Time (ms)")
    ax2.legend(loc='center', bbox_to_anchor=(1.22, 0.5), fancybox=True, shadow=True)

    plt.tight_layout()
    plt.show()


def noise_sensitivity_protocol(runner, net, E_inp, duration, input_duration):
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

    T = 1000  # average window: 1.5 ms
    x = bm.linspace(-bm.pi, bm.pi, net.size_E)
    ma = moving_average(runner.mon['E.spike'], n=T, axis=0)
    ts = moving_average(runner.mon.ts, n=T, axis=0)
    bump_activity = bm.vstack([bm.sum(ma * bm.cos(x[None,]), axis=1), bm.sum(ma * bm.sin(x[None,]), axis=1)])
    readout = get_pos_from_tan(bump_activity[1], bump_activity[0])
    mse = bm.nanmean((readout[100000:] - 0.) ** 2)
    print("Slow CANN Decoding MSE:", mse)

    plt.plot(ts, readout, marker='.', markersize=2., linestyle='None', alpha=0.5)
    plt.ylim([-bm.pi / 2, bm.pi / 2])
    plt.xlabel('Time (ms)')
    plt.ylabel('Population Vector Angle')
    # todo: subplot(1,1) for decoding error on different noise strength
    plt.tight_layout()


def sudden_change_convergence_protocol(runner, net, E_inp, duration, input_duration):
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


def smooth_moving_lag_protocol(runner, net, E_inp, duration, input_duration):
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


def turn_off_with_exicitation_protocol(runner, net, E_inp, duration, input_duration):
    fig, gs = bp.visualize.get_figure(1, 1, 1.5, 4.)
    ax1 = fig.add_subplot(gs[:1, 0])
    bp.visualize.raster_plot(runner.mon.ts, runner.mon['E.spike'], xlim=[0, duration], markersize=1., alpha=0.3)
    # plot onset and offset of the stimulus
    ax1.plot((input_duration[0], input_duration[0]), (0, net.size_E), color='blue', linestyle='--', linewidth=2.)
    ax1.plot((input_duration[1], input_duration[1]), (0, net.size_E), color='blue', linestyle='--', linewidth=2.)
    ax1.set_xlabel("")
    ax1.set_ylim([0, net.size_E])
    ax1.set_xticklabels([])


def debug_protocol(runner, net, E_inp, duration, input_duration, neuron_index):
    fig, gs = bp.visualize.get_figure(2, 1, 2, 3)
    # subplot 1: plot current using `plot_E_currents`
    ax1 = fig.add_subplot(gs[0, 0])
    currents = plot_E_currents(runner, net, E_inp, neuron_index, ax1,
                               plot_items=['leak', 'Fc', 'total_rec'])
    ax1.grid()
    ax1.legend()
    # subplot 2: plot currents directly using monitored values
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(runner.mon.ts, runner.mon['E._leak'][:, neuron_index], color='blue', 
             linestyle='--', alpha=0.5, linewidth=1.0, label='leak')
    ax2.plot(runner.mon.ts, runner.mon['E._ext'][:, neuron_index], color='red', 
             linestyle='--', alpha=0.5, linewidth=1.0, label='Fc')
    ax2.plot(runner.mon.ts, runner.mon['E._recinp'][:, neuron_index], color='green',
             linestyle='-', alpha=1.0, linewidth=2.0, label='total_rec')
    ax2.grid()
    ax2.legend()

    return

vis_setup = {
    "persistent_input": partial(persistent_protocol, neuron_indices=(400, 50)),
    "balance_check_bump_input": partial(balance_check_bump_protocol, neuron_indices=[int(400*1), int(100*1)]),
    "balance_check_flat_input": partial(balance_check_flat_protocol, neuron_indices=[375, 125]),
    "irregular_check_flat_input": partial(irregular_check_flat_protocol),
    "tracking_input": partial(tracking_protocol),
    "convergence_rate_population_readout_input": partial(convergence_rate_population_readout_protocol),
    "convergence_rate_current_input": partial(convergence_rate_current_protocol, neuron_index=700),
    "noise_sensitivity_input": partial(noise_sensitivity_protocol),
    "sudden_change_convergence_input": partial(sudden_change_convergence_protocol),
    "smooth_moving_lag_input": partial(smooth_moving_lag_protocol),
    "turn_off_with_exicitation_input": partial(turn_off_with_exicitation_protocol),
    "debug_input": partial(debug_protocol, neuron_index=400),
}
