import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import brainpy as bp
import brainpy.math as bm
import numpy as np
import matplotlib.pyplot as plt
from functools import partial

from utils.vis_utils import (
    calculate_population_readout, 
    moving_average, 
    get_pos_from_tan,
    plot_E_currents,
    )

import pdb


def persistent_protocol(runner, net, E_inp, duration, input_duration, neuron_indices):
    fig, gs = bp.visualize.get_figure(2, 1, 1.5, 6)
    # subplot 1: raster plot on E
    ax1 = fig.add_subplot(gs[:1, 0])
    bp.visualize.raster_plot(runner.mon.ts, runner.mon['E.spike'], xlim=[0, duration], markersize=1., alpha=0.3)
    # plt.plot(input_duration, [int(net.size_E/2), int(net.size_E/2)], label='input peak', color='red')
    ax1.plot((input_duration[0], input_duration[0]), (0, net.size_E), color='gray', linestyle='--', linewidth=2.)
    ax1.plot((input_duration[1], input_duration[1]), (0, net.size_E), color='gray', linestyle='--', linewidth=2.)
    ax1.set_xlabel("")
    ax1.set_ylim([0, net.size_E])
    ax1.set_xticklabels([])

    # subplot 2: readout plot
    ax2 = fig.add_subplot(gs[1:2, 0])
    T = 200  # 2 ms
    ts = moving_average(runner.mon.ts, n=T, axis=0)
    readout = calculate_population_readout(activity=runner.mon['E.spike'], T=T)
    fitted = moving_average(readout.T, n=T * 10, axis=0)
    nm_readout = readout.T / bm.max(fitted) * 1.05
    fitted = fitted / bm.max(fitted) * 1.05
    ax2.plot(ts, nm_readout, label='readout', alpha=0.5, color="gray")
    ax2.plot(ts[T * 10 - 1:], fitted, marker='.', markersize=2., linestyle='None', label='average', color="black")
    ax2.plot((input_duration[0], input_duration[0]), (bm.min(nm_readout), bm.max(nm_readout)),
             color='gray', linestyle='--', linewidth=2.)
    ax2.plot((input_duration[1], input_duration[1]), (bm.min(nm_readout), bm.max(nm_readout)),
             color='gray', linestyle='--', linewidth=2.)
    ax2.legend(loc='upper right')
    ax2.grid('on')
    ax2.set_xlim([0, duration])
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Readout")
    ax2.set_ylim([-0.2, 1.5])

    fig.align_ylabels([ax1, ax2])

    plt.show()


def tracking_protocol(runner, net, E_inp, duration, input_duration):
    fig, gs = bp.visualize.get_figure(2, 6, 1.7, 1)
    # subplot 1: raster E plot
    ax1 = fig.add_subplot(gs[0, 0:5])
    ax2 = ax1.twinx()
    ax2.set_ylim([0, net.size_E])
    ax1.set_ylim([-np.pi, np.pi])
    ax1.set_ylabel("Preferred feature")
    ax1.set_yticks([-np.pi, -1.5, 0, 1.5, np.pi])
    ax1.set_yticklabels(["$-\pi$", "-1.5", "0", "1.5", "$\pi$"])
    bp.visualize.raster_plot(runner.mon.ts, runner.mon['E.spike'], markersize=1., alpha=0.3)
    ax2.plot(bm.arange(input_duration[1]), bm.argmax(E_inp[::100], axis=1), label='input peak', color='white',
             alpha=1.0, linestyle='--', linewidth=2.)
    ax1.plot([800, 800], [-np.pi, np.pi], color='blue')
    plt.xlim([0, input_duration[1]])
    ax1.set_xticklabels([])
    plt.xlabel("")

    # subplot 2: distorted bump shape
    ax = fig.add_subplot(gs[0, 5])
    profile = bm.sum(runner.mon['E.spike'][75000:85000, :], axis=0)
    ax.plot(profile, color='blue')
    newx = ax.lines[0].get_ydata()
    newy = ax.lines[0].get_xdata()
    ax.lines[0].set_xdata(newx)
    ax.lines[0].set_ydata(newy)
    ax.set_xlim([0, np.max(newx)])
    ax.set_ylim([0, np.max(newy)])
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_xticks([])

    ax.plot([0, np.max(newx)], [bm.argmax(E_inp[80000, :]), bm.argmax(E_inp[80000, :])],
            linewidth=2, linestyle='--', color='black')
    readout = (bm.sum(bm.arange(0, net.size_E) * profile) / bm.sum(profile)).item()
    ax.plot([0, np.max(newx)], [readout, readout], linewidth=2, linestyle='-', color='black')


    # subplot 3: lag plot
    fig.add_subplot(gs[1, 0:5])
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
    lag = input_pos - bump_pos
    lag[lag > bm.pi] = lag[lag > bm.pi] - 2 * bm.pi
    lag[lag < -bm.pi] = lag[lag < -bm.pi] + 2 * bm.pi
    plt.plot(ts, lag, color='black')
    plt.grid("on")
    plt.xlim([0, input_duration[1]])
    plt.xlabel("Time (ms)")
    plt.ylabel("Difference")

    plt.show()


def convergence_rate_population_readout_protocol(runner, net, E_inp, duration, input_duration):
    fig, gs = bp.visualize.get_figure(2, 1, 1.7, 4)
    # subplot 1: raster plot on E
    ax1 = fig.add_subplot(gs[:1, 0])
    bp.visualize.raster_plot(runner.mon.ts, runner.mon['E.spike'], markersize=1., alpha=0.3)
    ax1.plot(input_duration, [int(net.size_E / 2), int(net.size_E / 2)],
             label='input peak', color='white', linestyle='--', linewidth=1.5)
    ax1.set_xlim([0, duration])
    ax1.set_ylim([0, net.size_E])
    ax1.set_xlabel("")
    ax1.set_xticklabels([])
    ax1.set_yticks([0, 200, 400, 600, 800])

    # subplot 2: PPC readout normalized
    ax2 = fig.add_subplot(gs[1:2, 0])
    T = 100  # 1 ms
    ts = moving_average(runner.mon.ts, n=T, axis=0)
    neuron_index = int(net.size_E/2)
    x = bm.linspace(-bm.pi, bm.pi, net.size_E)
    ma = moving_average(runner.mon['E.spike'], n=T, axis=0)  # average window: 1 ms
    bump_activity = bm.vstack([bm.sum(ma * bm.cos(x[None,]), axis=1), bm.sum(ma * bm.sin(x[None,]), axis=1)])
    readout = bm.array([[1., 0.]]) @ bump_activity
    nm_readout = readout.T / bm.mean(readout[:, int(input_duration[0]/runner.dt*1.5):int(input_duration[1]/runner.dt)])
    ma_input = moving_average(E_inp[:, neuron_index], n=T, axis=0)
    ma_input = ma_input - bm.min(ma_input)
    fitted = moving_average(nm_readout, n=T * 10, axis=0)
    ax2.plot(ts, nm_readout, label='projection', alpha=0.5)
    ax2.plot(ts[T*10-1:], fitted, marker='.', markersize=2., linestyle='None', label='average')
    ax2.plot(ts, ma_input / bm.max(ma_input), color='black', linewidth=2., label='input')
    ax2.set_ylabel("Readout")
    ax2.set_xlim([0, duration])
    ax2.set_ylim([-0.8, 2])
    ax2.set_xlabel("Time (ms)")
    ax2.grid()

    # plt.legend(loc='upper left')
    fig.align_ylabels([ax1, ax2])
    plt.tight_layout()
    plt.show()


def convergence_rate_current_protocol(runner, net, E_inp, duration, input_duration, neuron_index):
    fig, gs = bp.visualize.get_figure(2, 5, 1.8, 1)

    # subplot 1: plot current on central E
    ax2 = fig.add_subplot(gs[0, 0:4])
    plot_E_currents(runner, net, E_inp, neuron_index, ax2, 
                    plot_items=['total_E', 'total_I', 'total'], 
                    # plot_items=['Ec_s', 'Ic_s', 'Fc', 'shunting', 'leak', 'total'],
                    smooth_T=2500)

    ax2.grid('on')
    ax2.set_xlim([500, 2000])
    ax2.set_ylabel("Current")
    ax2.set_xlabel("Time (ms)")

    # plt.legend()

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

    T = 1000  # average window: 10 ms
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

    # plt.plot(ts, bump_pos, alpha=0.5)
    # plt.plot(ts, input_pos, alpha=0.5)

    import uuid
    np.savez(f'cann_sudden_change_{str(uuid.uuid4())[:5]}.npz', ts=ts, bump_pos=bump_pos, input_pos=input_pos)

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
    bump_pos = get_pos_from_tan(bump_activity[1], bump_activity[0])

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
    np.savez(f'cann_tracking_lag_{str(uuid.uuid4())[:5]}.npz', ts=ts, lag=lag)

    return


def turn_off_with_excitation_protocol(runner, net, E_inp, duration, input_duration):
    fig, gs = bp.visualize.get_figure(1, 1, 1.5, 2)
    # subplot 1: raster plot on E
    ax1 = fig.add_subplot(gs[:1, 0])
    bp.visualize.raster_plot(runner.mon.ts, runner.mon['E.spike'], xlim=[0, duration], markersize=1., alpha=0.3)
    # plot onset and offset of the stimulus
    ax1.plot((input_duration[0], input_duration[0]), (0, net.size_E), color='blue', linestyle='--', linewidth=2.)
    ax1.plot((input_duration[1], input_duration[1]), (0, net.size_E), color='blue', linestyle='--', linewidth=2.)
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylim([0, net.size_E])
    ax1.set_xlim([1100., 1500.])


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
    "tracking_input": partial(tracking_protocol),
    "convergence_rate_population_readout_input": partial(convergence_rate_population_readout_protocol),
    "convergence_rate_current_input": partial(convergence_rate_current_protocol, neuron_index=700),
    "noise_sensitivity_input": partial(noise_sensitivity_protocol),
    "sudden_change_convergence_input": partial(sudden_change_convergence_protocol),
    "smooth_moving_lag_input": partial(smooth_moving_lag_protocol),
    "turn_off_with_exicitation_input": partial(turn_off_with_excitation_protocol),
    "debug_input": partial(debug_protocol, neuron_index=400),
}