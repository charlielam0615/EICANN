import matplotlib.pyplot as plt
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
    file_list = glob.glob(f'compare_tracking_lag/{prefix}*.npz')
    data = {key: [] for key in keys}
    for file_name in file_list:
        single_data = np.load(file_name)
        for key in keys:
            data[key].append(single_data[key])

    for key in keys:
        data[key] = np.stack(data[key])

    return data


def compare_tracking_lag():
    fig = plt.figure(figsize=(5, 2.2))
    ax = fig.add_subplot(1, 1, 1)
    cann_data = merge_data(prefix="cann_tracking_lag", keys=['ts', 'lag'])
    coupled_data = merge_data(prefix="coupled_tracking_lag", keys=['ts', 'lag'])

    cann_lag_data = cann_data['lag'][:, 200000:10:-1].reshape(-1)
    coupled_lag_data = coupled_data['lag'][:, 200000:10:-1].reshape(-1)
    sns.kdeplot(data=cann_lag_data, label='Unbalanced CANN', fill=True, linewidth=2, bw_method=0.1,
                color='gray')
    sns.kdeplot(data=coupled_lag_data, label='Balanced CANN', fill=True, linewidth=2, bw_method=0.1,
                color='blue')
    # sns.histplot(data=cann_lag_data, color='gray', label='Unbalanced CANN', alpha=0.3, ax=ax)
    # sns.histplot(data=coupled_lag_data, color='blue', label='Balanced CANN', alpha=0.3, ax=ax)

    plt.xlabel('Lag (rad)')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    compare_tracking_lag()