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


def different_input_position_protocol(runner, net, E_inp):
    # PPC readout normalized
    T = 30000  # 300 ms
    x = bm.linspace(-bm.pi, bm.pi, net.size_E)
    avg_sp = bm.mean(runner.mon['E.spike'][-T:], axis=0)
    bump_activity = bm.vstack([bm.sum(avg_sp * bm.cos(x[None, ]), axis=1),
                               bm.sum(avg_sp * bm.sin(x[None, ]), axis=1)])
    bump_pos = get_pos_from_tan(bump_activity[1], bump_activity[0])

    return bump_pos.to_numpy().item()


def different_input_position_sanity_check_protocol(runner, net, E_inp):
    duration = 1300.
    fig, gs = bp.visualize.get_figure(3, 1, 2, 8)
    fig.add_subplot(gs[:1, 0])
    center_index = bm.argmax(E_inp, axis=1)[0]
    bp.visualize.raster_plot(runner.mon.ts, runner.mon['E.spike'], xlim=[0, duration], markersize=1., alpha=0.5)
    plt.plot([0, duration], [center_index, center_index], label='input peak', color='red')

    neuron_indices = (center_index, bm.mod(int(net.size_E/2+center_index), net.size_E))
    for i in range(2):
        fig.add_subplot(gs[1+i:2+i, 0])
        neuron_index = neuron_indices[i]
        Ec_inp = runner.mon['E2E_s.g'][:, neuron_index]
        Fc_inp = E_inp[:, neuron_index]
        bp.visualize.line_plot(runner.mon.ts, Ec_inp, legend='rec_E', alpha=0.5)
        bp.visualize.line_plot(runner.mon.ts, Fc_inp, legend='F', alpha=0.5)
        plt.legend(loc=4)
        plt.grid('on')
    plt.show()
    return


vis_setup = {
    "different_input_position_protocol": partial(different_input_position_protocol),
    "sanity_check": partial(different_input_position_sanity_check_protocol),
}