import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt


def moving_average(a, n, axis):
    ret = bm.cumsum(a, axis=axis, dtype=bm.float32)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def sin_input_protocol(runner, duration, inputs, start=300., **kwargs):
    fig, gs = bp.visualize.get_figure(3, 1, 1.5, 5)

    fig.add_subplot(gs[0, 0])
    bp.visualize.raster_plot(runner.mon.ts, runner.mon['E.spike'], xlim=(start, duration), markersize=1.)

    # inputs
    fig.add_subplot(gs[1, 0])
    T = 1000
    avg_inputs = moving_average(bm.mean(inputs, axis=1), n=T, axis=0)
    ts = moving_average(runner.mon.ts, n=T, axis=0)
    avg_spike = moving_average(bm.mean(runner.mon['E.spike'].astype(bm.float32), axis=1), n=T, axis=0) * 3650+0.06
    bp.visualize.line_plot(ts, avg_inputs, xlim=(start, duration), alpha=0.7, legend='stimulus')
    bp.visualize.line_plot(ts, avg_spike, xlim=(start, duration), alpha=0.7, legend='firing rate (normalized)')
    fig.axes[0].yaxis.set_ticklabels([])
    fig.axes[0].yaxis.set_ticks([])
    plt.legend(loc='lower left')
    plt.ylim([0, 3])

    # currents
    fig.add_subplot(gs[2, 0])
    neuron_index = 375
    Fc_inp = inputs
    Ec_inp = runner.mon['E2E.g']
    Ic_inp = runner.mon['I2E.g']
    total_inp = Ec_inp + Ic_inp + Fc_inp

    bp.visualize.line_plot(runner.mon.ts, total_inp[:,neuron_index], xlim=(start, duration), legend='Total', alpha=0.7)
    bp.visualize.line_plot(runner.mon.ts, (Ec_inp+Fc_inp)[:,neuron_index], xlim=(start, duration), legend='Excitatory', alpha=0.7)
    bp.visualize.line_plot(runner.mon.ts, (Ic_inp)[:,neuron_index], xlim=(start, duration), legend='Inhibitory', alpha=0.7)
    # fig.axes[2].yaxis.set_ticks([0])
    plt.grid('on')
    return


def linear_input_protocol(runner, duration, inputs, **kwargs):
    fig, gs = bp.visualize.get_figure(3, 1, 1.5, 5)
    # neuron firing rate vs. input
    fig.add_subplot(gs[:1, 0])
    T = 1000
    avg_inputs = moving_average(bm.mean(inputs, axis=1), n=T, axis=0)
    ts = moving_average(runner.mon.ts, n=T, axis=0)
    avg_spike = moving_average(bm.mean(runner.mon['E.spike'].astype(bm.float32), axis=1), n=T, axis=0) * 3600 + 0.06
    bp.visualize.line_plot(ts, avg_spike, xlim=(0, duration), alpha=0.8, legend='firing rate (normalized)')
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

    fig, gs = bp.visualize.get_figure(2, 1, 1.5, 5)
    fig.add_subplot(gs[:2, 0])
    bp.visualize.line_plot(runner.mon.ts-duration/2, autocorr, xlim=(-duration/2, duration/2), linewidth=2., alpha=0.8)

    return


def check_current_input_protocol(runner, duration, inputs, **kwargs):
    size_E = (27, 27)
    neuron_index = (13, 13)
    gl = -0.15

    fig, gs = bp.visualize.get_figure(1, 1, 3, 6)
    # plot currents
    fig.add_subplot(gs[0, 0])
    tf = lambda d: d.reshape(-1, *size_E)[:, neuron_index[0], neuron_index[1]]
    Ec_inp = tf(runner.mon['E2E.g'])
    Fc_inp = inputs[:,0]
    Ic_inp = tf(runner.mon['I2E.g'])
    leak = gl * tf(runner.mon['E.V'])
    total_inp = Ec_inp + Ic_inp + Fc_inp + leak
    bp.visualize.line_plot(runner.mon.ts, Ec_inp, legend='rec_E', alpha=0.5)
    bp.visualize.line_plot(runner.mon.ts, Fc_inp, legend='F', alpha=0.5)
    bp.visualize.line_plot(runner.mon.ts, Ic_inp, legend='rec_I', alpha=0.5)
    bp.visualize.line_plot(runner.mon.ts, total_inp, legend='Total', alpha=0.5)
    bp.visualize.line_plot(runner.mon.ts, leak, legend='leak', alpha=0.5)


vis_setup = {
    "sin_input": sin_input_protocol,
    "linear_input": linear_input_protocol,
    "constant_input": constant_input_protocol,
    "check_current_input": check_current_input_protocol,
}
