import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import Divider, Size
import numpy as np
import glob
import seaborn as sns


def moving_average(a, n, axis):
    ret = np.cumsum(a, axis=axis, dtype=np.float32)
    if axis == 0:
        ret[n:] = ret[n:] - ret[:-n]
        ret = ret[n - 1:] / n
    elif axis == 1:
        ret[:, n:] = ret[:, n:] - ret[:, :-n]
        ret = ret[:, n - 1:] / n
    else:
        raise Exception
    return ret


def merge_data(prefix, keys):
    file_list = glob.glob(f'/Users/charlie/Local Documents/Projects/EI Balanced CANN/1d code/experiments/compare_tracking_lag/data/{prefix}*.npz')
    data = {key: [] for key in keys}
    for file_name in file_list:
        single_data = np.load(file_name)
        for key in keys:
            data[key].append(single_data[key])

    for key in keys:
        data[key] = np.stack(data[key])

    return data


def compare_tracking_lag():
    figure = plt.figure(figsize=(4, 2.5))
    # The first items are for padding and the second items are for the axes.
    # sizes are in inch.
    h = [Size.Fixed(1.0), Size.Fixed(2)]
    v = [Size.Fixed(0.5), Size.Fixed(1.2)]

    divider = Divider(figure, (0, 0, 1, 1), h, v, aspect=False)
    # The width and height of the rectangle are ignored.

    ax = figure.add_axes(divider.get_position(),
                    axes_locator=divider.new_locator(nx=1, ny=1))
    
    cann_data = merge_data(prefix="cann_tracking_lag", keys=['ts', 'lag'])
    coupled_data = merge_data(prefix="coupled_tracking_lag", keys=['ts', 'lag'])

    cann_lag_data = cann_data['lag'][:, 300000:100:-1].reshape(-1)
    coupled_lag_data = coupled_data['lag'][:, 300000:100:-1].reshape(-1)
    sns.kdeplot(data=cann_lag_data, label='CANN', fill=True, linewidth=2, bw_method=0.1,
                color='blue', ax=ax)
    sns.kdeplot(data=coupled_lag_data, label='Our Model', fill=True, linewidth=2, bw_method=0.1,
                color='red', ax=ax)
    # sns.histplot(data=cann_lag_data, color='gray', label='Unbalanced CANN', alpha=0.3, ax=ax)
    # sns.histplot(data=coupled_lag_data, color='blue', label='Balanced CANN', alpha=0.3, ax=ax)

    ax.set_xlim([-0.1, 0.7])
    ax.set_xlabel('Lag (rad)')
    ax.grid()
    ax.legend()
    plt.show()


if __name__ == "__main__":
    compare_tracking_lag()