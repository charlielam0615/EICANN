import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.gridspec import GridSpec
from brainpy import math
import brainpy.math as bm
from warnings import warn



def moving_average(a, n, axis):
    ret = bm.cumsum(a, axis=axis, dtype=bm.float32)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def get_pos_from_tan(a, b):
    warn('`get_pos_from_tan` is deprecated', DeprecationWarning, stacklevel=2)
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


def calculate_spike_center(activity, size, T=None, feature_range=None):
    if T is not None:
        activity = moving_average(activity, n=T, axis=0)

    if feature_range is None:
        x = bm.arange(size)
        spike_center = bm.sum(activity * x[None, ], axis=1) / bm.sum(activity, axis=1)
    else:
        assert len(feature_range) == 2, "feature_range must be a tuple of two numbers."
        features = bm.linspace(feature_range[0], feature_range[1], size)
        spike_center = bm.sum(activity * features[None, ], axis=1) / bm.sum(activity, axis=1)
    return spike_center


def plot_E_currents(runner, net, E_inp, neuron_index, ax, plot_items=("total_E", "total_I", "total"), smooth_T=None):
    currents = {}
    leak = net.E.gl * runner.mon['E.V'][:, neuron_index]
    Fc_inp = E_inp[:, neuron_index]
    currents.update({"Fc": Fc_inp, "leak": leak})

    Ec_inp = runner.mon['E2E_s.g'][:, neuron_index]
    Ic_inp = runner.mon['I2E_s.g'][:, neuron_index]
    currents.update({"Ec_s": runner.mon['E2E_s.g'][:, neuron_index], 
                     "Ic_s": runner.mon['I2E_s.g'][:, neuron_index]})

    if net.name.startswith("EICANN"):
       Ec_inp = Ec_inp + runner.mon['E2E_f.g'][:, neuron_index]
       Ic_inp = Ic_inp + runner.mon['I2E_f.g'][:, neuron_index]
       currents.update({"Ec_f": runner.mon['E2E_f.g'][:, neuron_index],
                        "Ic_f": runner.mon['I2E_f.g'][:, neuron_index]})
    
    shunting_inp = net.shunting_k * (Ec_inp + Fc_inp) * runner.mon['I2E_s.g'][:, neuron_index]
    currents.update({"shunting": shunting_inp, "Ec": Ec_inp, "Ic": Ic_inp})

    total_E = Ec_inp + Fc_inp
    total_I = Ic_inp + shunting_inp + leak
    total_rec = Ec_inp + Ic_inp + shunting_inp
    total = total_E + total_I

    currents.update({"total_E": total_E, "total_I": total_I, "total_rec": total_rec, "total": total})
    
    color_palette = ['blue', 'red', 'green', 'orange', 'purple', 'gray', 'pink', 
                     'brown', 'yellow', 'cyan', 'magenta', 'olive', 'lime']

    ts = runner.mon.ts if smooth_T is None else moving_average(runner.mon.ts, n=smooth_T, axis=0)

    for i, item in enumerate(plot_items):
        if smooth_T is not None:
            currents[item] = moving_average(currents[item], n=smooth_T, axis=0)

        # plot arguments
        if item.startswith("total"):
            alpha, linewidth, linestyle = 1.0, 2., '-'
        else:
            alpha, linewidth, linestyle = 0.5, 1., '--'

        if item!= 'total':
            ax.plot(ts, currents[item], linestyle=linestyle,
                    label=item, alpha=alpha, linewidth=linewidth, color=color_palette[i])
        else:
            ax.plot(ts, currents['total'], linestyle=linestyle,
                    label=item, alpha=alpha, linewidth=linewidth, color='black')

    return {k:v for (k,v) in currents.items() if k in plot_items}


def get_current(current, neuron_index, slice_indices=None):
    if slice_indices is not None:
        return current[slice_indices[0]:slice_indices[1], neuron_index]
    else:
        return current[:, neuron_index]


def animate_2D(values,
               net_size,
               dt=None,
               val_min=None,
               val_max=None,
               cmap=None,
               frame_delay=10,
               frame_step=1,
               title_size=10,
               figsize=None,
               gif_dpi=None,
               video_fps=None,
               save_path=None,
               aggregate_func=None,
               show=True):

    return _animate_2D(values, net_size, dt=dt, val_min=val_min, val_max=val_max, cmap=cmap,
                       frame_delay=frame_delay, frame_step=frame_step, title_size=title_size,
                       figsize=figsize, gif_dpi=gif_dpi, video_fps=video_fps, save_path=save_path,
                       aggregate_func=aggregate_func, show=show)


def _animate_2D(values,
               net_size,
               dt=None,
               val_min=None,
               val_max=None,
               cmap=None,
               frame_delay=10,
               frame_step=1,
               title_size=10,
               figsize=None,
               gif_dpi=None,
               video_fps=None,
               save_path=None,
               aggregate_func=None,
               show=True):
  """Animate the potentials of the neuron group.

  Parameters
  ----------
  values : np.ndarray
      The membrane potentials of the neuron group.
  net_size : tuple
      The size of the neuron group.
  dt : float
      The time duration of each step.
  val_min : float, int
      The minimum of the potential.
  val_max : float, int
      The maximum of the potential.
  cmap : str
      The colormap.
  frame_delay : int, float
      The delay to show each frame.
  frame_step : int
      The step to show the potential. If `frame_step=3`, then each
      frame shows one of the every three steps.
  title_size : int
      The size of the title.
  figsize : None, tuple
      The size of the figure.
  gif_dpi : int
      Controls the dots per inch for the movie frames. This combined with
      the figure's size in inches controls the size of the movie. If
      ``None``, use defaults in matplotlib.
  video_fps : int
      Frames per second in the movie. Defaults to ``None``, which will use
      the animation's specified interval to set the frames per second.
  save_path : None, str
      The save path of the animation.
  aggregate_func : None, function handle
      an additional function to plot data on top of `values`.
  show : bool
      Whether show the animation.

  Returns
  -------
  anim : animation.FuncAnimation
      The created animation function.
  """
  dt = math.get_dt() if dt is None else dt
  num_step, num_neuron = values.shape
  height, width = net_size

  values = np.asarray(values)
  val_min = values.min() if val_min is None else val_min
  val_max = values.max() if val_max is None else val_max

  figsize = figsize or (6, 6)

  fig = plt.figure(figsize=(figsize[0], figsize[1]), constrained_layout=True)
  gs = GridSpec(1, 1, figure=fig)
  fig.add_subplot(gs[0, 0])

  def frame(t):
    img = values[t]
    fig.clf()
    plt.pcolor(img, cmap=cmap, vmin=val_min, vmax=val_max)
    plt.colorbar()
    # plt.axis('off')
    fig.suptitle(t="Time: {:.2f} ms".format((t + 1) * dt),
                 fontsize=title_size,
                 fontweight='bold')
    if aggregate_func is not None:
        aggregate_func(t)

    return [fig.gca()]

  values = values.reshape((num_step, height, width))
  anim = animation.FuncAnimation(fig=fig,
                                 func=frame,
                                 frames=list(range(1, num_step, frame_step)),
                                 init_func=None,
                                 interval=frame_delay,
                                 repeat_delay=3000)
  if save_path is None:
    if show:
      plt.show()
  else:
    print(f'Saving the animation into {save_path} ...')
    if save_path[-3:] == 'gif':
      anim.save(save_path, dpi=gif_dpi, writer='imagemagick')
    elif save_path[-3:] == 'mp4':
      anim.save(save_path, writer='ffmpeg', fps=video_fps, bitrate=3000)
    else:
      anim.save(save_path + '.mp4', writer='ffmpeg', fps=video_fps, bitrate=3000)

  # if show:
  #   plt.show()
  return anim