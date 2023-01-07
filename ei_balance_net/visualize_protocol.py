import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
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
    fig, gs = bp.visualize.get_figure(3, 1, 1.5, 5)
    # neuron firing rate vs. input
    fig.add_subplot(gs[:1, 0])
    T = 1000
    avg_inputs = moving_average(bm.mean(inputs, axis=1), n=T, axis=0)
    ts = moving_average(runner.mon.ts, n=T, axis=0)
    avg_spike = moving_average(bm.mean(runner.mon['E.spike'].astype(bm.float32), axis=1), n=T, axis=0) * 3600 + 0.06
    bp.visualize.line_plot(ts, avg_spike, xlim=(0, duration), alpha=0.8, legend='firing rate')
    bp.visualize.line_plot(ts, avg_inputs, xlim=(0, duration), alpha=0.8, linewidth=2., legend='input')
    fig.axes[0].yaxis.set_ticklabels([])
    fig.axes[0].yaxis.set_ticks([])

    # raster plot
    fig.add_subplot(gs[1:2, 0])
    bp.visualize.raster_plot(runner.mon.ts, runner.mon['E.spike'], xlim=(0, duration), markersize=1.)

    # currents
    fig.add_subplot(gs[2:3, 0])
    neuron_index = 375
    Fc_inp = inputs
    Ec_inp = runner.mon['E2E.g']
    Ic_inp = runner.mon['I2E.g']
    total_inp = Ec_inp + Ic_inp + Fc_inp

    bp.visualize.line_plot(runner.mon.ts, total_inp[:, neuron_index], xlim=(0, duration), legend='Total', alpha=0.5)
    bp.visualize.line_plot(runner.mon.ts, (Ec_inp + Fc_inp)[:, neuron_index], xlim=(0, duration), legend='Excitatory',
                           alpha=0.5)
    bp.visualize.line_plot(runner.mon.ts, Ic_inp[:, neuron_index], xlim=(0, duration), legend='Inhibitory', alpha=0.5)
    fig.axes[2].yaxis.set_ticks([0])
    plt.grid('on')
    plt.tight_layout()
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

    fig, gs = bp.visualize.get_figure(1, 1, 2, 5)
    fig.add_subplot(gs[:1, 0])
    bp.visualize.line_plot(runner.mon.ts-duration/2, autocorr, xlim=(-duration/2, duration/2), linewidth=2., alpha=0.8)
    plt.ylabel('Autocorrelation')
    plt.xlabel('Time (ms)')
    plt.tight_layout()

    return


def localized_input_protocol(runner, duration, inputs, neuron_index=375, **kwargs):
    gl = -0.15
    fig, gs = bp.visualize.get_figure(2, 1, 2, 7)
    # raster E plot
    fig.add_subplot(gs[0, 0])
    bp.visualize.raster_plot(runner.mon.ts, runner.mon['E.spike'], markersize=1., alpha=0.2)
    plt.xlim([500, duration])
    # plot current on central E
    fig.add_subplot(gs[1, 0])
    Ec_inp = runner.mon['E2E.g']
    Fc_inp = inputs
    leak = gl * runner.mon['E.V']
    Ic_inp = runner.mon['I2E.g']
    total_inp = Ec_inp + Ic_inp + Fc_inp + leak
    bp.visualize.line_plot(runner.mon.ts, (Fc_inp+Ec_inp)[:, neuron_index], legend='E', alpha=0.5)
    bp.visualize.line_plot(runner.mon.ts, (Ic_inp+leak)[:, neuron_index], legend='I', alpha=0.5)
    bp.visualize.line_plot(runner.mon.ts, total_inp[:, neuron_index], legend='Total', alpha=0.5)
    bp.visualize.line_plot(runner.mon.ts, Fc_inp[:, neuron_index], legend='Feedforward', linestyle='--', alpha=0.5)
    plt.ylabel(f"Neuron {neuron_index}")
    plt.legend(loc='upper right')
    plt.grid('on')
    plt.xlim([500, duration])
    plt.tight_layout()


def staircase_input_protocol(runner, duration, inputs, **kwargs):
    ts = runner.mon.ts[1000:-1000]
    frE = bp.measure.firing_rate(runner.mon['E.spike'], width=10., dt=global_dt)[1000:-1000]
    frI = bp.measure.firing_rate(runner.mon['I.spike'], width=10., dt=global_dt)[1000:-1000]
    fig = plt.figure()
    ax1 = fig.add_subplot()
    lineE, = ax1.plot(ts, frE, label='E rate')
    lineI, = ax1.plot(ts, frI, label='I rate')
    ax1.set_ylabel('firing rates', fontsize=15)
    ax1.set_xlabel('Time(ms)')
    input = inputs[1000:-1000]
    ax2 = ax1.twinx()
    lineInput, = ax2.plot(ts, input, label='input', color='red', linestyle='--')
    ax2.set_ylabel('input', fontsize=15)
    ax2.legend(handles=[lineE, lineI, lineInput], )


def staircase_powerspec_input_protocol(runner, duration, inputs, **kwargs):
    fig, gs = bp.visualize.get_figure(5, 3, 2, 2)
    stair_dur = 1500.
    stair_num = 8
    # frE = bp.measure.firing_rate(runner.mon['E.spike'], width=0.5, dt=global_dt)
    frE_ = bm.sum(runner.mon['E.spike'], axis=-1)
    # E raster plots
    fig.add_subplot(gs[0, :3])
    bp.visualize.raster_plot(runner.mon.ts, runner.mon['E.spike'][:, :300], markersize=1., alpha=0.5)
    # E firing rates
    fig.add_subplot(gs[1, :3])
    bp.visualize.line_plot(runner.mon.ts, frE_, xlim=(0., 1000.))
    # spectral analysis
    for i in range(stair_num):
        ax = fig.add_subplot(5, 3, i+7)
        start = int(stair_dur * i / global_dt)
        end = int(stair_dur * (i+1) / global_dt)-1
        sig = frE_[start+2000:end-2000]
        sig = sig - bm.mean(sig)
        # f, Pxx_den = signal.welch(sig, fs=1/global_dt*1000, nperseg=25*1024)
        f, Pxx_den = signal.periodogram(sig, fs=1/global_dt*1000)
        ax.semilogy(f, Pxx_den)
        ax.set_xlabel(f'{i+1}')
        ax.set_xlim([-10, 500])

    plt.tight_layout()


vis_setup = {
    "sin_input": sin_input_protocol,
    "linear_input": linear_input_protocol,
    "constant_input": constant_input_protocol,
    "localized_input": localized_input_protocol,
    "staircase_input": staircase_input_protocol,
    "staircase_powerspec_input": staircase_powerspec_input_protocol,
}
