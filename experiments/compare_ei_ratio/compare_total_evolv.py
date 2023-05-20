import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import os.path as osp

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import Divider, Size
from utils.vis_utils import plot_and_fill_between, moving_average
import glob


def merge_data(prefix, keys):
    path = f'/Users/charlie/Local Documents/Projects/EI Balanced CANN/1d code/experiments/compare_ei_ratio/data_1000/{prefix}*.npz'
    # path = f'./data_1000/{prefix}*.npz'
    file_list = glob.glob(path)
    data = {key: [] for key in keys}
    for file_name in file_list:
        single_data = np.load(file_name)
        for key in keys:
            data[key].append(single_data[key])

    for key in keys:
        data[key] = np.stack(data[key])

    return data

smooth_T = 1000
dt = 0.01
sti_onset_idx = int(1000 / dt + smooth_T)
available_keys = ['center_total_E', 'center_total_I', 'center_total', 'center_ts',
                  'peripheral_total_E', 'peripheral_total_I', 'peripheral_total', 'peripheral_ts']

balanced_data = merge_data(prefix="balanced", 
                           keys=available_keys)
unbalanced_data = merge_data(prefix="unbalanced", 
                             keys=available_keys)


# smooth data
for key in available_keys:
    balanced_data[key+'_sm'] = moving_average(balanced_data[key], smooth_T, axis=1)
    unbalanced_data[key+'_sm'] = moving_average(unbalanced_data[key], smooth_T, axis=1)

bpo = balanced_data['peripheral_total_sm'][:, sti_onset_idx:sti_onset_idx+100].mean(axis=1, keepdims=True)
upo = unbalanced_data['peripheral_total_sm'][:, sti_onset_idx:sti_onset_idx+100].mean(axis=1, keepdims=True)
bpr = balanced_data['peripheral_total_sm'][:, sti_onset_idx:] / bpo
upr = unbalanced_data['peripheral_total_sm'][:, sti_onset_idx:] / upo

bpei = np.clip(balanced_data['peripheral_total_E_sm'] / np.abs(balanced_data['peripheral_total_I_sm']), a_min=0, a_max=5)
upei = np.clip(unbalanced_data['peripheral_total_E_sm'] / np.abs(unbalanced_data['peripheral_total_I_sm']), a_min=0, a_max=5)
bcei = np.clip(balanced_data['center_total_E_sm'] / np.abs(balanced_data['center_total_I_sm']), a_min=0, a_max=5)
ucei = np.clip(unbalanced_data['center_total_E_sm'] / np.abs(unbalanced_data['center_total_I_sm']), a_min=0, a_max=5)

# compute mean and std
for key in available_keys:
    bpr_mean = bpr.mean(axis=0)
    upr_mean = upr.mean(axis=0)
    bpr_std = bpr.std(axis=0)
    upr_std = upr.std(axis=0)

    bpei_mean = bpei.mean(axis=0)
    upei_mean = upei.mean(axis=0)
    bpei_std = bpei.std(axis=0)
    upei_std = upei.std(axis=0)

    bcei_mean = bcei.mean(axis=0)
    ucei_mean = ucei.mean(axis=0)
    bcei_std = bcei.std(axis=0)
    ucei_std = ucei.std(axis=0)


plot1_dict = {
    'balanced_peripheral': {
        'mean': bpr_mean,
        'std': bpr_std,
        'color': 'red',
    },
    'unbalanced_peripheral': {
        'mean': upr_mean,
        'std': upr_std,
        'color': 'blue',
    },
}

plot2_dict = {
    # balanced
    'center':{
        'mean': bcei_mean,
        'std': bcei_std,
        'color': 'green',
    },
    'peripheral':{
        'mean': bpei_mean,
        'std': bpei_std,
        'color': 'orange',
    },
}

plot3_dict = {
    # unbalanced
    'center':{
        'mean': ucei_mean,
        'std': ucei_std,
        'color': 'green',
    },
    'peripheral':{
        'mean': upei_mean,
        'std': upei_std,
        'color': 'orange',
    },
}

fig = plt.figure(figsize=(9, 1.5))
ax = fig.add_subplot(1, 3, 1)
ts = moving_average(balanced_data['center_ts'][0, sti_onset_idx:], smooth_T)
for key, value in plot1_dict.items():
    plot_and_fill_between(ax, x=ts, y_mean=value['mean'], y_std=value['std'], color=value['color'], label="           ", shade_alpha=0.3)
ax.set_ylim([-0.5, 3])
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.legend(frameon=False)

ts = moving_average(balanced_data['center_ts'][0], smooth_T)
ax = fig.add_subplot(1, 3, 2)
for key, value in plot2_dict.items():
    plot_and_fill_between(ax, x=ts, y_mean=value['mean'], y_std=value['std'], color=value['color'], label="            ", shade_alpha=0.3)
ax.set_xlim([500, 2000])
ax.set_ylim([0, 6])
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.legend(frameon=False, loc='upper right')

ax = fig.add_subplot(1, 3, 3)
for key, value in plot3_dict.items():
    plot_and_fill_between(ax, x=ts, y_mean=value['mean'], y_std=value['std'], color=value['color'], label="            ", shade_alpha=0.3)
plt.xlim([500, 2000])
plt.ylim([0, 6])
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.legend(frameon=False, loc='upper right')

plt.savefig("/Users/charlie/Local Documents/Projects/EI Balanced CANN/overleaf_version/AI - formal figs/Fig5/new_fig_2.png", dpi=600)

plt.show()

