import brainpy as bp
import brainpy.math as bm
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from functools import partial
import os


global_dt = 0.01


def moving_average(a, n, axis):
    ret = bm.cumsum(a, axis=axis, dtype=bm.float32)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def linear_input_protocol(runner, net, inputs, duration, input_duration):
    n_size = runner.mon['E.spike'].shape[1]
    fig, gs = bp.visualize.get_figure(3, 1, 1.2, 5)

    # subplot 1: neuron firing rate vs. input
    ax1 = fig.add_subplot(gs[:1, 0])
    T = 100  # 1 ms
    avg_inputs = moving_average(bm.mean(inputs, axis=1), n=T, axis=0)
    ts = moving_average(runner.mon.ts, n=T, axis=0)
    avg_spike = moving_average(bm.mean(runner.mon['E.spike'].astype(
        bm.float32), axis=1), n=T, axis=0) * 2600 + 0.1
    ax1.plot(ts, avg_spike, alpha=0.8, label='firing rate',
             color='black', linewidth=2., linestyle='--')
    ax1.plot(ts, avg_inputs, alpha=0.8, label='input',
             color='blue', linewidth=2., linestyle='-')
    ax1.grid()
    ax1.set_xlim([25, duration])
    ax1.yaxis.set_ticklabels([])
    ax1.set_ylabel('Activity')
    ax1.xaxis.set_ticklabels([])
    ax1.legend()

    # subplot 2: raster plot
    ax2 = fig.add_subplot(gs[1:2, 0])
    bp.visualize.raster_plot(runner.mon.ts, runner.mon['E.spike'], xlim=(
        25, duration), markersize=1., alpha=0.5)
    ax2.xaxis.set_ticklabels([])
    ax2.set_ylim([0, n_size])
    ax2.set_xlabel("")

    # subplot 3: currents
    ax3 = fig.add_subplot(gs[2:3, 0])
    Fc_inp = bm.mean(inputs, axis=1)
    Ec_inp = bm.mean(runner.mon['E2E_s.g']+runner.mon['E2E_f.g'], axis=1)
    leak = bm.mean(net.gl * runner.mon['E.V'], axis=1)
    shunting = net.shunting_k * \
        bm.mean(
            (inputs+runner.mon['E2E_s.g']+runner.mon['E2E_f.g'])*runner.mon['I2E_s.g'], axis=1)
    Ic_inp = bm.mean(runner.mon['I2E_s.g'] +
                     runner.mon['I2E_f.g'], axis=1) + leak + shunting
    total_inp = Ec_inp + Ic_inp + Fc_inp

    ax3.plot(runner.mon.ts, total_inp, label='Total', alpha=1.0,
             color='black', linestyle='-', linewidth=2.)
    ax3.plot(runner.mon.ts, Ec_inp + Fc_inp, label='E', alpha=1.0,
             color='blue', linestyle='--', linewidth=2.)
    ax3.plot(runner.mon.ts, Ic_inp, label='I', alpha=1.0,
             color='gray', linestyle='--', linewidth=2.)
    ax3.set_xlim([25, duration])
    ax3.yaxis.set_ticks([0])
    ax3.grid('on')
    ax3.legend()
    ax3.set_ylabel('Current')
    ax3.set_xlabel('Time (ms)')
    fig.align_ylabels([ax1, ax2, ax3])
    plt.tight_layout()

    return


def linear_input_save_protocol(runner, net, inputs, duration, input_duration):
    import uuid
    n_size = runner.mon['E.spike'].shape[1]
    T = 10  # 0.1 ms
    avg_inputs = moving_average(bm.mean(inputs, axis=1), n=T, axis=0)
    ts = moving_average(runner.mon.ts, n=T, axis=0)
    avg_spike = moving_average(
        bm.mean(runner.mon['E.spike'].astype(bm.float32), axis=1), n=T, axis=0)

    results_path = "/Users/charlie/Local Documents/Projects/EI Balanced CANN/1d code/" \
        "experiments/coupled_fast_response/balanced_linear_input_data"

    np.savez(os.path.join(results_path, f'balanced_{str(uuid.uuid4())[:5]}.npz'),
             ts=ts,
             avg_spike=avg_spike,
             avg_inputs=avg_inputs,
             )

    return


vis_setup = {
    "linear_input": linear_input_protocol,
    "linear_input_save": linear_input_save_protocol,

}
