import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import os
import glob


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
ei_scale = 1.2
jie = -4.8 * ei_scale
jii = -3.8 * ei_scale
jee = 2.5 * ei_scale
jei = 5.0 * ei_scale
JIE = jie / np.sqrt(size_Id*prob)
JII = jii / np.sqrt(size_Id*prob)
JEE = jee / np.sqrt(size_E*prob)
JEI = jei / np.sqrt(size_E*prob)
gl = 0.

# fit sigmoid curve using scipy.optimize.curve_fit
def sigmoid(x, a, b, c, d, cosmetic_value=0.):
    return a / (1 + np.exp(-(b+cosmetic_value) * (x - c))) + d

def fit_response_curve(x, y, start_t, end_t, skip_n=1):
    ind_s, ind_e = int(start_t/dt), int(end_t/dt)
    data_x = x[ind_s:ind_e:skip_n]
    data_y = y[ind_s:ind_e:skip_n]
    p0 = [max(data_y), 1, np.median(data_x), min(data_y)]
    popt, pcov = curve_fit(sigmoid, data_x, data_y, p0, method='trf')
    # a small cosmetic_value is added to account for the curve fitting artifact
    fitted_curve = sigmoid(x, *popt, cosmetic_value=-0.07)
    return fitted_curve


def theoretical_response(inputs, scale, offset):
    def te(x): return np.sqrt(qe*prob) * x
    def ti(x): return np.sqrt(qi*prob) * x
    Iext = inputs / (np.sqrt(num_ff)*fE)
    rate_numerator = (fE*ti(jii)-fI*ti(jie)-1 /
                      np.sqrt(num)*fE*V_gap*tau_I) * Iext
    rate_denominator = ti(jie)*te(jei)-te(jee)*ti(jii) + 1/np.sqrt(num) * \
        V_gap*(te(jee)*tau_I+ti(jii)*tau_E) - 1/num * V_gap**2 * tau_E*tau_I
    rate = rate_numerator / rate_denominator
    # change unit from ms to s
    rate = rate * 1000.
    rate = rate * scale + offset

    return rate


def merge_data(path, prefix, keys):
    file_list = glob.glob(f'{path}/linear_input_data/{prefix}*.npz')
    data = {key: [] for key in keys}
    for file_name in file_list:
        single_data = np.load(file_name)
        for key in keys:
            data[key].append(single_data[key])

    for key in keys:
        data[key] = np.stack(data[key])

    return data


def load_raw_data():
    balance_data = merge_data(path='/Users/charlie/Local Documents/Projects/EI Balanced CANN/code/ei_balance_net',
                              prefix="balanced",
                              keys=['ts', 'avg_spike', 'avg_inputs'])
    unbalance_data = merge_data(path="/Users/charlie/Local Documents/Projects/EI Balanced CANN/code/ei_balance_net/unbalanced_net",
                                prefix="unbalanced",
                                keys=['ts', 'avg_spike', 'avg_inputs'])

    return balance_data, unbalance_data


def save_data_stats(balance_data, unbalance_data):
    balanced_spike = np.mean(balance_data['avg_spike'], axis=0)
    unbalanced_spike = np.mean(unbalance_data['avg_spike'], axis=0)
    balanced_spike_std = np.std(balance_data['avg_spike'], axis=0)
    unbalanced_spike_std = np.std(unbalance_data['avg_spike'], axis=0)
    ts = balance_data['ts'][0, :]
    inputs = np.mean(balance_data['avg_inputs'], axis=0)

    np.savez("/Users/charlie/Local Documents/Projects/EI Balanced CANN/code/ei_balance_net/balance_unbalance_data.npz",
             balanced_spike=balanced_spike, unbalanced_spike=unbalanced_spike,
             balanced_spike_std=balanced_spike_std, unbalanced_spike_std=unbalanced_spike_std,
             ts=ts, inputs=inputs
             )
    return


def load_data_stats():
    data = np.load(
        "/Users/charlie/Local Documents/Projects/EI Balanced CANN/code/ei_balance_net/balance_unbalance_data.npz")
    return (data['balanced_spike'],
            data['unbalanced_spike'],
            data['balanced_spike_std'],
            data['unbalanced_spike_std'],
            data['ts'],
            data['inputs'])


def compare_response():
    balance_scale = 1.
    balance_offset = -10.
    unbalance_scale = 22.
    unbalance_offset = -75.
    input_scale = 65.
    input_offset = -12.
    shift_value = 2.0

    plot_error_shade = False
    plot_prediction = False
    plot_raw = False
    plot_fit = True

    normalize_flag = True
    
    # load data stats if exists else compute satats and save
    if not os.path.exists("/Users/charlie/Local Documents/Projects/EI Balanced CANN/code/ei_balance_net/balance_unbalance_data.npz"):
        balance_data, unbalance_data = load_raw_data()
        save_data_stats(balance_data, unbalance_data)

    balanced_spike, unbalanced_spike, balanced_spike_std, unbalanced_spike_std, ts, inputs = load_data_stats()

    # change rate units from per dt to s^-1
    balanced_spike = balanced_spike * 1000 / dt
    unbalanced_spike = unbalanced_spike * 1000 / dt
    balanced_spike_std = balanced_spike_std * 1000 / dt
    unbalanced_spike_std = unbalanced_spike_std * 1000 / dt

    # scale data for plotting
    balanced_spike = balanced_spike * balance_scale + balance_offset
    unbalanced_spike = unbalanced_spike * unbalance_scale + unbalance_offset
    balanced_spike_std = balanced_spike_std * balance_scale
    unbalanced_spike_std = unbalanced_spike_std * unbalance_scale
    prediction = theoretical_response(inputs, balance_scale, balance_offset)
    inputs = inputs * input_scale + input_offset

    figure = plt.figure(figsize=(6, 2))

    # plot stimulus
    if normalize_flag:
        # normalize inputs to [0, 1]
        inputs = (inputs - np.min(inputs)) / (np.max(inputs) - np.min(inputs))
    plt.plot(ts, inputs, linewidth=4.5,
             label="Input", alpha=1.0, color='gray', linestyle='-')
    
    if plot_raw:
        # plot balance response
        plt.plot(ts, balanced_spike, linewidth=1.,
                label="Balance", alpha=0.2, color='blue')

        # plot unbalance response
        plt.plot(ts, unbalanced_spike, linewidth=1.,
                label="Unbalance", alpha=0.2, color='crimson')
    
    if plot_fit:
        fit_balance = fit_response_curve(x=ts, y=balanced_spike, start_t=60., end_t=140., skip_n=100)
        fit_unbalance = fit_response_curve(x=ts, y=unbalanced_spike, start_t=60., end_t=140., skip_n=100)

        if normalize_flag:
            # normalize fit curve to [0, 1]
            fit_balance = (fit_balance - np.min(fit_balance)) / (np.max(fit_balance) - np.min(fit_balance))
            fit_unbalance = (fit_unbalance - np.min(fit_unbalance)) / (np.max(fit_unbalance) - np.min(fit_unbalance))

        # balance response curve fit
        # shift value is added to account for the curve fitting artifact
        plt.plot(ts+shift_value, fit_balance, linewidth=1.0,
                label="Balanced", alpha=1.0, color='black', linestyle='-')
        
        # unbalance response curve fit
        plt.plot(ts+shift_value, fit_unbalance, linewidth=1.0,
                label="Unbalanced", alpha=1.0, color='black', linestyle='--')

    if plot_error_shade:
        plt.fill_between(ts, balanced_spike+balanced_spike_std, balanced_spike-balanced_spike_std,
                         color='blue', alpha=0.1)
        plt.fill_between(ts, unbalanced_spike+unbalanced_spike_std, unbalanced_spike-unbalanced_spike_std,
                         color='crimson', alpha=0.1)

    if plot_prediction:
        if normalize_flag:
            # normalize prediction to [0, 1]
            prediction = (prediction - np.min(prediction)) / (np.max(prediction) - np.min(prediction))

        plt.plot(ts, prediction, linewidth=2.5,
                label="Prediction", alpha=1.0, color='green')
    
    plt.xlim([70., 140.])
    plt.xlabel("Time (ms)")
    plt.ylabel("Normalized Activity")
    plt.grid()
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 0.8), fancybox=True, shadow=True, fontsize=10.)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    compare_response()
