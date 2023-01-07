import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.gridspec import GridSpec
from brainpy import math


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