import os, sys
import os.path as osp
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import brainpy as bp
import brainpy.math as bm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import Divider, Size
from functools import partial

from utils.vis_utils import (
    # calculate_population_readout, 
    # calculate_spike_center,
    decode_population_vector,
    moving_average, 
    get_pos_from_tan,
    plot_E_currents,
    )

import pdb


def persistent_protocol(runner, net, E_inp, duration, input_duration, neuron_indices):
    fig = plt.figure(figsize=(5., 4.5))

    # The first items are for padding and the second items are for the axes.
    # sizes are in inch.
    h = [Size.Fixed(1.0), Size.Fixed(3.)]
    v = [Size.Fixed(0.5), Size.Fixed(1.2), Size.Fixed(0.5), Size.Fixed(1.2)]

    divider = Divider(fig, (0, 0, 1, 1), h, v, aspect=False)
    # The width and height of the rectangle are ignored.

    ax1 = fig.add_axes(divider.get_position(),
                      axes_locator=divider.new_locator(nx=1, ny=3))

    bp.visualize.raster_plot(runner.mon.ts, runner.mon['E.spike'], xlim=[0, duration], markersize=1., alpha=0.3)
    # plot onset and offset of the stimulus
    ax1.plot((input_duration[0], input_duration[0]), (0, net.size_E), color='blue', linestyle='--', linewidth=1.5)
    ax1.plot((input_duration[1], input_duration[1]), (0, net.size_E), color='blue', linestyle='--', linewidth=1.5)
    ax1.set_xlabel("")
    ax1.set_ylim([0, net.size_E])
    ax1.set_xticklabels([])

    # subplot 2: readout plot
    ax2 = fig.add_axes(divider.get_position(),
                      axes_locator=divider.new_locator(nx=1, ny=1))
    T = 200  # 2 ms
    ts = moving_average(runner.mon.ts, n=T, axis=0)
    readout = calculate_population_readout(activity=runner.mon['E.spike'], T=T)
    fitted = moving_average(readout.T, n=T * 10, axis=0)
    nm_readout = readout.T / bm.max(fitted) * 1.1
    fitted = fitted / bm.max(fitted) * 1.1
    ax2.plot(ts, nm_readout, label='_projection', alpha=0.2, color="red")
    ax2.plot(ts[T * 10 - 1:], fitted, linewidth=1., linestyle='-', label='readout', color="red")
    # plot onset and offset of the stimulus
    ax2.plot((input_duration[0], input_duration[0]), (bm.min(nm_readout), bm.max(nm_readout)),
             color='blue', linestyle='--', linewidth=1.5)
    ax2.plot((input_duration[1], input_duration[1]), (bm.min(nm_readout), bm.max(nm_readout)),
             color='blue', linestyle='--', linewidth=1.5)
    ax2.legend(loc='upper right')
    ax2.grid('on')
    ax2.set_xlim([0, duration])
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Readout")
    ax2.set_ylim([-0.2, 1.5])

    fig.align_ylabels([ax1, ax2])
    plt.show()


def scalability_protocol(runner, net, E_inp, duration, input_duration):
    fig, gs = bp.visualize.get_figure(1, 1, 1., 4)
    n_skip = int(n_scale)
    skiped_spike = runner.mon['E.spike'][:, ::n_skip]
    # subplot 1: raster plot on E
    ax1 = fig.add_subplot(gs[:1, 0])
    bp.visualize.raster_plot(runner.mon.ts, skiped_spike, xlim=[0, duration], markersize=1., alpha=0.3, ax=ax1)
    # plot onset and offset of the stimulus
    ax1.plot((input_duration[0], input_duration[0]), (0, net.size_E), color='blue', linestyle='--', linewidth=2.)
    ax1.plot((input_duration[1], input_duration[1]), (0, net.size_E), color='blue', linestyle='--', linewidth=2.)
    # yticks = np.arange(0, skiped_spike.shape[-1]+1, 400)
    yticks = np.array([0, skiped_spike.shape[-1]])
    ax1.set_ylim([0, skiped_spike.shape[-1]])
    ax1.set_ylabel("")
    ax1.set_xlabel("")
    ax1.set_xticks([400, 1000])
    ax1.set_xticklabels([])
    ax1.set_yticks(yticks.tolist())
    ax1.set_yticklabels((yticks * n_skip).tolist())


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
    fig = plt.figure(figsize=(4, 5))

    # The first items are for padding and the second items are for the axes.
    # sizes are in inch.
    h = [Size.Fixed(1.0), Size.Fixed(2.)]
    v = [Size.Fixed(0.25), Size.Fixed(0.8), Size.Fixed(0.25), Size.Fixed(0.8), Size.Fixed(0.25), Size.Fixed(0.8), Size.Fixed(0.25), Size.Fixed(0.2)]

    divider = Divider(fig, (0, 0, 1, 1), h, v, aspect=False)
    # The width and height of the rectangle are ignored.

    ax = fig.add_axes(divider.get_position(),
                      axes_locator=divider.new_locator(nx=1, ny=5))

    # raster plot on E
    bp.visualize.raster_plot(runner.mon.ts, runner.mon['E.spike'], markersize=1., alpha=0.15, ax=ax)
    ax.plot(input_duration, [int(net.size_E / 2), int(net.size_E / 2)],
             label='input peak', color='white', linestyle='--', linewidth=1.5)
    ax.set_xlim([0, duration])
    ax.set_ylim([0, net.size_E])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels([])
    ax.set_yticks([0, 400, 800])
    ax.set_yticklabels([])


    # PPC readout
    ax = fig.add_axes(divider.get_position(),
                      axes_locator=divider.new_locator(nx=1, ny=3))
    T = 50  # 0.5 ms
    ts = moving_average(runner.mon.ts, n=T, axis=0)
    neuron_index = int(net.size_E/2)
    x = bm.linspace(-bm.pi, bm.pi, net.size_E)
    ma = moving_average(runner.mon['E.spike'], n=T, axis=0)
    decoded_pos, norm = decode_population_vector(ma)
    # adjust the color a little bit for better visualization
    norm_ = moving_average(norm, n=3*T, axis=0)
    norm = bm.concatenate([norm_, norm[-3*T+1:]])
    norm = norm - bm.mean(norm[10000:]) * 0.25
    confidence = bm.clip(norm / bm.mean(norm[-10000:]), a_min=0., a_max=1.)
    color = [plt.cm.YlGnBu(conf.item()*0.8 + 0.01) for conf in confidence]
    im = ax.scatter(ts, decoded_pos, marker='.', color=color, edgecolors=None, s=1.0)
    ax.set_xlim([0, duration])
    ax.set_ylim([-np.pi, np.pi])
    ax.set_yticks([-np.pi, 0, np.pi])
    ax.set_xticklabels([])
    ax.set_yticklabels([])


    # PPC readout projection
    ax = fig.add_axes(divider.get_position(),
                      axes_locator=divider.new_locator(nx=1, ny=1))
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
    ax.plot(ts, nm_readout, label='_projection', alpha=0.3, color='royalblue')
    ax.plot(ts[T*10-1:], fitted, linewidth=1., linestyle='-', label='       ', color='royalblue')
    ax.plot(ts, ma_input / bm.max(ma_input), color='black', linewidth=2., label='_input', alpha=0.5, linestyle='-')
    ax.set_xlim([0, duration])
    ax.set_xticklabels([])
    ax.set_ylim([-0.8, 2])
    ax.set_yticklabels([])
    ax.grid()

    plt.legend()

    # colorbar
    ax = fig.add_axes(divider.get_position(),
                      axes_locator=divider.new_locator(nx=1, ny=7))
    cmap = plt.cm.get_cmap('YlGnBu', 512)
    newcmp = ListedColormap(cmap(np.linspace(0., 0.8, 256)))
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    cbar = mpl.colorbar.ColorbarBase(ax, cmap=newcmp, norm=norm, orientation='horizontal')
    # cbar = fig.colorbar(im, cax=ax, orientation='horizontal')
    cbar.set_ticks([])


    plt.savefig("/Users/charlie/Local Documents/Projects/EI Balanced CANN/overleaf_version/AI - formal figs/Fig4/Fig4A1b_new.png", format='png', dpi=900)



def convergence_rate_current_protocol(runner, net, E_inp, duration, input_duration):
    E_size = net.size_E
    peri_index = int(E_size * 0.9)
    cent_index = int(E_size * 0.5)

    fig = plt.figure(figsize=(7, 4.5))

    # The first items are for padding and the second items are for the axes.
    # sizes are in inch.
    h = [Size.Fixed(1.0), Size.Fixed(2.), Size.Fixed(1.0), Size.Fixed(2.)]
    v = [Size.Fixed(0.5), Size.Fixed(1.2), Size.Fixed(0.5), Size.Fixed(1.2)]

    divider = Divider(fig, (0, 0, 1, 1), h, v, aspect=False)
    # The width and height of the rectangle are ignored.

    # subplot (1,1): plot current on peripheral E
    ax = fig.add_axes(divider.get_position(),
                      axes_locator=divider.new_locator(nx=1, ny=3))
    currents = plot_E_currents(
        runner, net, E_inp, peri_index, ax, 
        plot_items=['total_E', 'total_I', 'total'], 
        smooth_T=None)

    ax.grid('on')
    ax.set_xlim([500, 2000])
    ax.set_ylabel("Peripheral")
    ax.set_xlabel("Time (ms)")

    # subplot (1,2): plot E/I current ratio on peripheral E
    ax = fig.add_axes(divider.get_position(),
                      axes_locator=divider.new_locator(nx=3, ny=3))
    ax.plot(currents['ts'], currents['total_E'] / np.abs(currents['total_I']))

    ax.grid('on')
    ax.set_xlim([500, 2000])
    ax.set_ylim([0, 5])
    ax.set_ylabel("Peripheral")
    ax.set_xlabel("Time (ms)")

    # subplot (2,1): plot current on center E
    ax = fig.add_axes(divider.get_position(),
                      axes_locator=divider.new_locator(nx=1, ny=1))
    currents = plot_E_currents(runner, net, E_inp, cent_index, ax, 
                    plot_items=['total_E', 'total_I', 'total'], 
                    smooth_T=None)

    ax.grid('on')
    ax.set_xlim([500, 2000])
    ax.set_ylabel("Center")
    ax.set_xlabel("Time (ms)")

    # subplot (2,2): plot E/I current ratio on center E
    ax = fig.add_axes(divider.get_position(),
                      axes_locator=divider.new_locator(nx=3, ny=1))
    ax.plot(currents['ts'], currents['total_E'] / np.abs(currents['total_I']))

    ax.grid('on')
    ax.set_xlim([500, 2000])
    ax.set_ylim([0, 5])
    ax.set_ylabel("Center")
    ax.set_xlabel("Time (ms)")

    plt.show()


def noise_sensitivity_protocol(runner, net, E_inp, duration, input_duration, save_mse, noise_level_str=None):

    def calculate_mse():
        T = 1000  # average window: 10 ms
        ts = moving_average(runner.mon.ts, n=T, axis=0)
        readout = calculate_spike_center(runner.mon['E.spike'], size=net.size_E, T=T, feature_range=[-bm.pi, bm.pi])
        mse = bm.nanmean((readout[100000:] - 0.) ** 2)
        return mse, ts, readout

    if not save_mse:
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
        mse, ts, readout = calculate_mse()
        print("Slow CANN Decoding MSE:", mse)

        plt.plot(ts, readout, marker='.', markersize=2., linestyle='None', alpha=0.5)
        plt.ylim([-bm.pi / 2, bm.pi / 2])
        plt.xlabel('Time (ms)')
        plt.ylabel('Population Vector Angle')
        # todo: subplot(1,1) for decoding error on different noise strength
        plt.tight_layout()

    if save_mse:
        assert noise_level_str is not None
        mse, _, _ = calculate_mse()
        abs_path = '/Users/charlie/Local Documents/Projects/EI Balanced CANN/1d code/cann/'
        file_name = f'fast_cann_noise_sensitivity_{noise_level_str}.txt'
        with open(osp.join(abs_path, file_name), 'a+') as f:
            f.write(str(mse) + '\t')


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
    fig = plt.figure(figsize=(3.5, 2.5))

    # The first items are for padding and the second items are for the axes.
    # sizes are in inch.
    h = [Size.Fixed(1.0), Size.Fixed(1.5)]
    v = [Size.Fixed(0.5), Size.Fixed(1.2)]

    divider = Divider(fig, (0, 0, 1, 1), h, v, aspect=False)
    # The width and height of the rectangle are ignored.
    ax1 = fig.add_axes(divider.get_position(),
                      axes_locator=divider.new_locator(nx=1, ny=1))
    # subplot 1: raster plot on E
    bp.visualize.raster_plot(runner.mon.ts, runner.mon['E.spike'], xlim=[0, duration], markersize=1., alpha=0.3)
    # plot onset and offset of the stimulus
    ax1.plot((input_duration[0], input_duration[0]), (0, net.size_E), color='blue', linestyle='--', linewidth=2.)
    ax1.plot((input_duration[1], input_duration[1]), (0, net.size_E), color='blue', linestyle='--', linewidth=2.)
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylim([0, net.size_E])
    ax1.set_xlim([1100., 1800.])


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
    "convergence_rate_current_input": partial(convergence_rate_current_protocol),
    "noise_sensitivity_input": partial(noise_sensitivity_protocol, save_mse=True, noise_level_str='0_1'),
    "sudden_change_convergence_input": partial(sudden_change_convergence_protocol),
    "smooth_moving_lag_input": partial(smooth_moving_lag_protocol),
    "turn_off_with_exicitation_input": partial(turn_off_with_excitation_protocol),
    "debug_input": partial(debug_protocol, neuron_index=400),
}