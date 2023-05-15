import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import Divider, Size
import numpy as np
import os
import glob


def merge_data(prefix, keys):
    file_list = glob.glob(f'/Users/charlie/Local Documents/Projects/EI Balanced CANN/1d code/experiments/compare_sudden_change/data/{prefix}*.npz')
    data = {key: [] for key in keys}
    for file_name in file_list:
        single_data = np.load(file_name)
        for key in keys:
            data[key].append(single_data[key])

    for key in keys:
        data[key] = np.stack(data[key])

    return data


def compare_sudden_change():
    cann_data = merge_data(prefix="cann_sudden_change", keys=['ts', 'bump_pos', 'input_pos'])
    coupled_data = merge_data(prefix="coupled_sudden_change", keys=['ts', 'bump_pos', 'input_pos'])

    cann_ts = cann_data['ts'][0, :]
    coupled_ts = coupled_data['ts'][0, :]
    mean_cann_bump = np.mean(cann_data['bump_pos'], axis=0)
    std_cann_bump = np.std(cann_data['bump_pos'], axis=0)
    cann_sti = np.mean(cann_data['input_pos'], axis=0)
    mean_coupled_bump = np.mean(coupled_data['bump_pos'], axis=0)
    std_coupled_bump = np.std(coupled_data['bump_pos'], axis=0)

    figure = plt.figure(figsize=(4, 2.5))
    # The first items are for padding and the second items are for the axes.
    # sizes are in inch.
    h = [Size.Fixed(1.0), Size.Fixed(2)]
    v = [Size.Fixed(0.5), Size.Fixed(1.2)]

    divider = Divider(figure, (0, 0, 1, 1), h, v, aspect=False)
    # The width and height of the rectangle are ignored.

    ax = figure.add_axes(divider.get_position(),
                    axes_locator=divider.new_locator(nx=1, ny=1))

    # plot stimulus
    ax.plot(cann_ts, cann_sti, linewidth=1.5, label="_Input", alpha=1.0, color='black')
    # plot cann response
    ax.plot(cann_ts, mean_cann_bump, linewidth=1.5, label="CANN", alpha=1.0, color='blue')
    ax.fill_between(cann_ts, mean_cann_bump+std_cann_bump, mean_cann_bump-std_cann_bump,
                     color='blue', alpha=0.2)
    # plot ei balanced cann resposne
    ax.plot(coupled_ts, mean_coupled_bump, linewidth=1.5, label="Our Model", alpha=1.0, color='red')
    ax.fill_between(coupled_ts, mean_coupled_bump+std_coupled_bump, mean_coupled_bump-std_coupled_bump,
                     color='red', alpha=0.2)

    ax.set_xlim([200., 600.])
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Position (rad)")
    ax.grid()
    ax.legend()
    # ax.tight_layout()
    plt.show()


if __name__ == "__main__":
    compare_sudden_change()
