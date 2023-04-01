import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

global_dt = 0.01

def moving_average(a, n, axis):
    ret = bm.cumsum(a, axis=axis, dtype=bm.float32)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def sin_input_protocol(runner, duration, inputs, start=0., **kwargs):
    fig, gs = bp.visualize.get_figure(3, 1, 1.5, 5)
    # E raster plots
    fig.add_subplot(gs[:1, 0])
    bp.visualize.raster_plot(runner.mon.ts, runner.mon['E.spike'], xlim=(start, duration), markersize=1.)

    # inputs
    fig.add_subplot(gs[1:2, 0])
    T = 1
    avg_inputs = moving_average(bm.mean(inputs, axis=1), n=T, axis=0)
    ts = moving_average(runner.mon.ts, n=T, axis=0)
    avg_spike = moving_average(bm.mean(runner.mon['E.spike'].astype(bm.float32), axis=1), n=T, axis=0) * 3650+0.06
    bp.visualize.line_plot(ts, avg_inputs, xlim=(start, duration), alpha=0.7, legend='stimulus')
    bp.visualize.line_plot(ts, avg_spike, xlim=(start, duration), alpha=0.7, legend='firing rate')
    fig.axes[1].yaxis.set_ticklabels([])
    fig.axes[1].yaxis.set_ticks([])
    plt.legend(loc='upper right')

    # currents
    fig.add_subplot(gs[2:3, 0])
    Fc_inp = inputs
    Ec_inp = runner.mon['E2E.g']
    Ic_inp = runner.mon['I2E.g']
    total_inp = Ec_inp + Ic_inp + Fc_inp
    bp.visualize.line_plot(runner.mon.ts, bm.mean(total_inp, -1), xlim=(start, duration), legend='Total', alpha=0.7)
    bp.visualize.line_plot(runner.mon.ts, bm.mean(Ec_inp+Fc_inp, -1), xlim=(start, duration), legend='Excitatory', alpha=0.7)
    bp.visualize.line_plot(runner.mon.ts, bm.mean(Ic_inp, -1), xlim=(start, duration), legend='Inhibitory', alpha=0.7)
    plt.legend(loc='upper left')
    fig.axes[2].yaxis.set_ticks([0])
    plt.grid('on')
    plt.tight_layout()
    return


def linear_input_protocol(runner, duration, inputs, **kwargs):
    n_size = runner.mon['E.spike'].shape[1]
    fig, gs = bp.visualize.get_figure(3, 1, 1.2, 5)
    # subplot 1: neuron firing rate vs. input
    ax1 = fig.add_subplot(gs[:1, 0])
    T = 100  # 1 ms
    avg_inputs = moving_average(bm.mean(inputs, axis=1), n=T, axis=0)
    ts = moving_average(runner.mon.ts, n=T, axis=0)
    avg_spike = moving_average(bm.mean(runner.mon['E.spike'].astype(bm.float32), axis=1), n=T, axis=0) * 2600 + 0.1
    ax1.plot(ts, avg_spike, alpha=0.8, label='firing rate', color='black', linewidth=2., linestyle='--')
    ax1.plot(ts, avg_inputs, alpha=0.8, label='input', color='blue', linewidth=2., linestyle='-')
    ax1.grid()
    ax1.set_xlim([25, duration])
    ax1.yaxis.set_ticklabels([])
    ax1.set_ylabel('Activity')
    ax1.xaxis.set_ticklabels([])
    ax1.legend()
    # fig.axes[0].yaxis.set_ticks([])

    # subplot 2: raster plot
    ax2 = fig.add_subplot(gs[1:2, 0])
    bp.visualize.raster_plot(runner.mon.ts, runner.mon['E.spike'], xlim=(25, duration), markersize=1., alpha=0.5)
    ax2.xaxis.set_ticklabels([])
    ax2.set_ylim([0, n_size])
    ax2.set_xlabel("")

    # subplot 3: currents
    ax3 = fig.add_subplot(gs[2:3, 0])
    neuron_index = 375
    Fc_inp = inputs
    Ec_inp = runner.mon['E2E.g']
    Ic_inp = runner.mon['I2E.g']
    total_inp = Ec_inp + Ic_inp + Fc_inp

    ax3.plot(runner.mon.ts, total_inp[:, neuron_index], label='Total', alpha=1.0, color='black', linestyle='-', linewidth=2.)
    ax3.plot(runner.mon.ts, (Ec_inp + Fc_inp)[:, neuron_index], label='E', alpha=1.0, color='blue', linestyle='--', linewidth=2.)
    ax3.plot(runner.mon.ts, Ic_inp[:, neuron_index], label='I', alpha=1.0, color='gray', linestyle='--', linewidth=2.)
    ax3.set_xlim([25, duration])
    ax3.yaxis.set_ticks([0])
    ax3.grid('on')
    ax3.legend()
    ax3.set_ylabel('Current')
    ax3.set_xlabel('Time (ms)')
    fig.align_ylabels([ax1, ax2, ax3])
    plt.tight_layout()
    return


def linear_input_save_protocol(runner, duration, inputs, **kwargs):
    import uuid
    n_size = runner.mon['E.spike'].shape[1]
    T = 10  # 0.1 ms
    avg_inputs = moving_average(bm.mean(inputs, axis=1), n=T, axis=0)
    ts = moving_average(runner.mon.ts, n=T, axis=0)
    avg_spike = moving_average(bm.mean(runner.mon['E.spike'].astype(bm.float32), axis=1), n=T, axis=0)

    np.savez(f'linear_input_data/unbalanced_{str(uuid.uuid4())[:5]}.npz', 
             ts=ts, 
             avg_spike=avg_spike, 
             avg_inputs=avg_inputs,
             )
    return



def constant_input_protocol(runner, duration, inputs, **kwargs):
    # calculate population average autocorrelation c.f. Vreeswijk and Somoplinsky (1998)
    sample_n = 100  # sample pair number
    spiketr = runner.mon['E.spike'].T.astype(bm.float32)
    size_n, length = spiketr.shape
    index = bm.random.choice(bm.arange(size_n), sample_n, replace=False)
    X = spiketr[index, ] # shape [sample_n, T]
    autocorr = bm.zeros(length)
    for i in range(sample_n):
        autocorr += bm.correlate(X[i], X[i], mode='same')
    autocorr /= sample_n

    fig, gs = bp.visualize.get_figure(1, 2, 2, 3)
    ax = fig.add_subplot(gs[0, 0])
    bp.visualize.raster_plot(runner.mon.ts, runner.mon['E.spike'], xlim=(0, duration), ylim=(0, size_n), markersize=1., alpha=0.5)

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(runner.mon.ts-duration/2, autocorr, linewidth=2., alpha=0.8, color='k')
    ax.set_xlim(-duration/2, duration/2)
    ax.set_ylabel('Autocorrelation')
    ax.set_xlabel('Time (ms)')
    ax.grid()
    plt.tight_layout()

    return


vis_setup = {
    "sin_input": sin_input_protocol,
    "linear_input": linear_input_protocol,
    "linear_input_save": linear_input_save_protocol,
    "constant_input": constant_input_protocol,
}
