import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import brainpy as bp
import numpy as np
import os
import glob
import warnings

# warnings.filterwarnings('error')


# Parameters
n_scale = 0.125
size_E, size_Id, size_ff = int(
    800*n_scale), int(100*n_scale), int(1000*n_scale)
num = size_E + size_Id
num_ff = num
qe = size_E / num
qi = size_Id / num
prob = 0.25
dt = 0.01
tau_scale = 10
tau_E = 2 * tau_scale
tau_I = 2 * tau_scale
V_gap = 1.0
fE = 0.1
fI = 0.0
mu = 1.0
ei_scale = 1.5
tau_Ef = 1.0 * tau_scale
tau_If = 0.6 * tau_scale
jie = -4.8 * ei_scale
jii = -3.8 * ei_scale
jee = 2.5 * ei_scale
jei = 5.0 * ei_scale
JIE = jie / np.sqrt(size_Id*prob)
JII = jii / np.sqrt(size_Id*prob)
JEE = jee / np.sqrt(size_E*prob)
JEI = jei / np.sqrt(size_E*prob)
gl = 0.

processed_data_dir = "/Users/charlie/Local Documents/Projects/EI Balanced CANN/1d code/experiments/coupled_fast_response"

# fit sigmoid curve using scipy.optimize.curve_fit


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def smooth_data(ts, balanced_spike, unbalanced_spike, inputs, window_size):
    ts = moving_average(ts, window_size)
    balanced_spike = moving_average(balanced_spike, window_size)
    unbalanced_spike = moving_average(unbalanced_spike, window_size)
    inputs = moving_average(inputs, window_size)
    return ts, balanced_spike, unbalanced_spike, inputs


def sigmoid(x, a, b, c, d, cosmetic_value=0.):
    exponent = np.minimum(-(b+cosmetic_value) * (x - c), 20.)
    return a / (1 + np.exp(exponent)) + d


def fit_response_curve(x, y, start_t, end_t, skip_n=1):
    ind_s, ind_e = int(start_t/dt), int(end_t/dt)
    data_x = x[ind_s:ind_e:skip_n]
    data_y = y[ind_s:ind_e:skip_n]
    p0 = [max(data_y), 1, np.median(data_x), min(data_y)]
    popt, pcov = curve_fit(sigmoid, data_x, data_y, p0, method='dogbox')
    # a small cosmetic_value is added to account for the curve fitting artifact
    fitted_curve = sigmoid(x, *popt, cosmetic_value=-0.05)
    return fitted_curve


def merge_data(path, prefix, keys):
    file_list = glob.glob(f'{path}/{prefix}*.npz')
    data = {key: [] for key in keys}
    for file_name in file_list:
        single_data = np.load(file_name)
        for key in keys:
            data[key].append(single_data[key])

    for key in keys:
        data[key] = np.stack(data[key])

    return data


def load_raw_data():
    balance_data = merge_data(path='/Users/charlie/Local Documents/Projects/EI Balanced CANN/1d code/experiments/coupled_fast_response/balanced_linear_input_data',
                              prefix="balanced",
                              keys=['ts', 'avg_spike', 'avg_inputs'])
    unbalance_data = merge_data(path="/Users/charlie/Local Documents/Projects/EI Balanced CANN/1d code/experiments/coupled_fast_response/unbalanced_linear_input_data",
                                prefix="unbalanced",
                                keys=['ts', 'avg_spike', 'avg_inputs'])

    return balance_data, unbalance_data


def save_data_stats(balance_data, unbalance_data, smooth_window):
    balanced_spike = np.mean(balance_data['avg_spike'], axis=0)
    unbalanced_spike = np.mean(unbalance_data['avg_spike'], axis=0)
    inputs = np.mean(balance_data['avg_inputs'], axis=0)
    ts = balance_data['ts'][0, :]

    ts, balanced_spike, unbalanced_spike, inputs = smooth_data(
        ts, balanced_spike, unbalanced_spike, inputs, smooth_window)

    balanced_spike_std = np.std(balanced_spike, axis=0)
    unbalanced_spike_std = np.std(unbalanced_spike, axis=0)

    np.savez(os.path.join(processed_data_dir, "processed_data.npz"),
             balanced_spike=balanced_spike, unbalanced_spike=unbalanced_spike,
             balanced_spike_std=balanced_spike_std, unbalanced_spike_std=unbalanced_spike_std,
             ts=ts, inputs=inputs
             )
    return


def load_data_stats():
    data = np.load(os.path.join(processed_data_dir, "processed_data.npz"))
    return (data['balanced_spike'],
            data['unbalanced_spike'],
            data['balanced_spike_std'],
            data['unbalanced_spike_std'],
            data['ts'],
            data['inputs'])


def calculate_offset(data, offset_start_idx, offset_end_idx):
    return np.mean(data[offset_start_idx:offset_end_idx])


def calculate_scale(data, offset, scale_start_idx, scale_end_idx):
    return 1 / (np.mean(data[scale_start_idx:scale_end_idx]) - offset)


def compare_response():
    shift_value = 3.0

    fit_start_time = 170.
    fit_end_time = 210.

    plot_error_shade = False
    plot_raw = False
    plot_fit = True

    normalize_flag = True

    # load data stats if exists else compute satats and save
    if not os.path.exists(os.path.join(processed_data_dir, "processed_data.npz")):
        balance_data, unbalance_data = load_raw_data()
        save_data_stats(balance_data, unbalance_data, smooth_window=100)

    balanced_spike, unbalanced_spike, balanced_spike_std, unbalanced_spike_std, ts, inputs = load_data_stats()

    # change rate units from per dt to s^-1
    balanced_spike = balanced_spike * 1000 / dt
    unbalanced_spike = unbalanced_spike * 1000 / dt
    balanced_spike_std = balanced_spike_std * 1000 / dt
    unbalanced_spike_std = unbalanced_spike_std * 1000 / dt

    # plot stimulus
    if normalize_flag:
        # inputs = (inputs - np.min(inputs)) / (np.max(inputs) - np.min(inputs))
        calculate_t_window = 5.
        offset_start_idx = int(fit_start_time / dt - calculate_t_window / dt)
        offset_end_idx = int(fit_start_time / dt + calculate_t_window / dt)
        scale_start_idx = int(fit_end_time / dt - calculate_t_window / dt)
        scale_end_idx = int(fit_end_time / dt + calculate_t_window / dt)

        balance_offset = calculate_offset(
            balanced_spike, offset_start_idx, offset_end_idx)
        balance_scale = calculate_scale(
            balanced_spike, balance_offset, scale_start_idx, scale_end_idx)

        unbalance_offset = calculate_offset(
            unbalanced_spike, offset_start_idx, offset_end_idx)
        unbalance_scale = calculate_scale(
            unbalanced_spike, unbalance_offset, scale_start_idx, scale_end_idx)

        input_offset = calculate_offset(
            inputs, offset_start_idx, offset_end_idx)
        input_scale = calculate_scale(
            inputs, input_offset, scale_start_idx, scale_end_idx)
    else:
        balance_scale = 1.
        balance_offset = 0.
        unbalance_scale = 1.
        unbalance_offset = 0.
        input_scale = 1
        input_offset = 0.

    # scale data for plotting
    balanced_spike = (balanced_spike - balance_offset) * balance_scale
    unbalanced_spike = (unbalanced_spike - unbalance_offset) * unbalance_scale
    balanced_spike_std = balanced_spike_std * balance_scale
    unbalanced_spike_std = unbalanced_spike_std * unbalance_scale
    inputs = (inputs - input_offset) * input_scale

    # Plotting

    fig, gs = bp.visualize.get_figure(1, 1, 1.5, 3)
    ax = fig.add_subplot(gs[0, 0])

    ax.plot(ts, inputs, linewidth=4.5,
             label="Input", alpha=1.0, color='gray', linestyle='-')

    if plot_raw:
        # plot balance response
        ax.plot(ts, balanced_spike, linewidth=1.,
                 label="Balance", alpha=0.2, color='blue')

        # plot unbalance response
        ax.plot(ts, unbalanced_spike, linewidth=1.,
                 label="Unbalance", alpha=0.2, color='crimson')

    if plot_fit:
        fit_balance = fit_response_curve(
            x=ts, y=balanced_spike, start_t=fit_start_time, end_t=fit_end_time, skip_n=100)
        fit_unbalance = fit_response_curve(
            x=ts, y=unbalanced_spike, start_t=fit_start_time, end_t=fit_end_time, skip_n=100)

        if normalize_flag:
            # normalize fit curve to [0, 1]
            fit_balance = (fit_balance - np.min(fit_balance)) / \
                (np.max(fit_balance) - np.min(fit_balance))
            fit_unbalance = (fit_unbalance - np.min(fit_unbalance)) / \
                (np.max(fit_unbalance) - np.min(fit_unbalance))

        # balance response curve fit
        # shift value is added to account for the curve fitting artifact
        ax.plot(ts+shift_value, fit_balance, linewidth=1.0,
                 label="Balanced", alpha=1.0, color='black', linestyle='-')

        # unbalance response curve fit
        ax.plot(ts+shift_value, fit_unbalance, linewidth=1.0,
                 label="Unbalanced", alpha=1.0, color='black', linestyle='--')

    if plot_error_shade:
        ax.fill_between(ts, balanced_spike+balanced_spike_std, balanced_spike-balanced_spike_std,
                         color='blue', alpha=0.1)
        ax.fill_between(ts, unbalanced_spike+unbalanced_spike_std, unbalanced_spike-unbalanced_spike_std,
                         color='crimson', alpha=0.1)

    ax.set_xlim([fit_start_time, fit_end_time])
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Activity")
    ax.set_xticks(np.arange(fit_start_time, fit_end_time, 10).tolist())
    ax.set_xticklabels(np.arange(0, fit_end_time-fit_start_time, 10).astype(np.int32).tolist())
    ax.grid()

    # ax.legend(loc='center', bbox_to_anchor=(1.4, 0.5),
    #            fancybox=True, shadow=True, fontsize=10.)

    ax.legend(fontsize=7)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    compare_response()
