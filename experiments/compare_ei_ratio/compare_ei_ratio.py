import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import os.path as osp

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import Divider, Size
from utils.vis_utils import plot_and_fill_between, moving_average
import glob


def merge_data(prefix, keys):
    file_list = glob.glob(f'/Users/charlie/Local Documents/Projects/EI Balanced CANN/1d code/experiments/compare_ei_ratio/data_1000/{prefix}*.npz')
    data = {key: [] for key in keys}
    for file_name in file_list:
        single_data = np.load(file_name)
        for key in keys:
            data[key].append(single_data[key])

    for key in keys:
        data[key] = np.stack(data[key])

    return data

smooth_T = 2000
available_keys = ['center_total_E', 'center_total_I', 'center_total', 'center_ts',
                  'peripheral_total_E', 'peripheral_total_I', 'peripheral_total', 'peripheral_ts']

balanced_data = merge_data(prefix="balanced", 
                           keys=available_keys)
unbalanced_data = merge_data(prefix="unbalanced", 
                             keys=available_keys)


# compute mean and std
for key in available_keys:
    balanced_data[key+'_mean'] = np.mean(balanced_data[key], axis=0)
    balanced_data[key+'_std'] = np.std(balanced_data[key], axis=0)
    unbalanced_data[key+'_mean'] = np.mean(unbalanced_data[key], axis=0)
    unbalanced_data[key+'_std'] = np.std(unbalanced_data[key], axis=0)



fig = plt.figure(figsize=(6, 4.5))

# The first items are for padding and the second items are for the axes.
# sizes are in inch.
h = [Size.Fixed(1.0), Size.Fixed(1.5), Size.Fixed(1.0), Size.Fixed(1.5)]
v = [Size.Fixed(0.5), Size.Fixed(1.2), 
     Size.Fixed(0.5), Size.Fixed(0.6),
     Size.Fixed(0.5), Size.Fixed(0.6),]

divider = Divider(fig, (0, 0, 1, 1), h, v, aspect=False)
# The width and height of the rectangle are ignored.

# ===============================================
# subplot (1,1): balanced model - plot current on center E
# subplot (1,2): balanced model - plot current on peripheral E
# subplot (1,3): unbalanced model - plot current on center E
# subplot (1,4): unbalanced model - plot current on peripheral E
# subplot (2, x): corresponding E/I ratio
# ===============================================

layout_args = {
    'balanced': {
        'center_current': {
            'nx': 1,
            'ny': 5,
            'legend': False,
            },
        'peripheral_current': {
            'nx': 1,
            'ny': 3,
            'legend': False,
            },
        'ratio': {
            'nx': 1,
            'ny': 1,
            'legend': False,
            },
        },
    'unbalanced': {
        'center_current': {
            'nx': 3,
            'ny': 5,
            'legend': True,
            },
        'peripheral_current': {
            'nx': 3,
            'ny': 3,
            'legend': False,
            },
        'ratio': {
            'nx': 3,
            'ny': 1,
            'legend': True,
            },
        }
    }

plot_args = {
    'total_E': {
        'color': 'blue',
        'label': 'total_E'
    },
    'total_I': {
        'color': 'red',
        'label': 'total_I',
    },
    'total': {
        'color': 'black',
        'label': 'total',
    }
}


data_dict = {
    'balanced': balanced_data,
    'unbalanced': unbalanced_data,
}

for mode in data_dict.keys():
    for loc in ['center', 'peripheral']:
        # plot currents
        ax = fig.add_axes(
            divider.get_position(),
            axes_locator=divider.new_locator(nx=layout_args[mode][f"{loc}_current"]['nx'], 
                                             ny=layout_args[mode][f"{loc}_current"]['ny'])
            )
        for current_name in plot_args.keys():
            ts = data_dict[mode][f"{loc}_ts"][0]
            mean_data = data_dict[mode][f"{loc}_{current_name}_mean"]
            std_data = data_dict[mode][f"{loc}_{current_name}_std"]
            ts = moving_average(ts, n=smooth_T, axis=0)
            mean_data = moving_average(mean_data, n=smooth_T, axis=0)
            std_data = moving_average(std_data, n=smooth_T, axis=0)
            plot_and_fill_between(ax, ts, mean_data, std_data, **plot_args[current_name])
            ax.set_yticks([])
            ax.set_xticklabels([])
            legend_flag = layout_args[mode][f"{loc}_current"]['legend']

        ax.set_xlim([750, 1500])

        # ax.set_ylabel(loc)
        # if legend_flag:
        #     ax.legend()


    # plot E/I ratio
    ax = fig.add_axes(
        divider.get_position(),
        axes_locator=divider.new_locator(nx=layout_args[mode]["ratio"]['nx'],
                                            ny=layout_args[mode]["ratio"]['ny'])
        )
    ts = data_dict[mode][f"{loc}_ts"][0]
    ts = moving_average(ts, n=smooth_T, axis=0)

    for loc, color in [('center', 'green'), ('peripheral', 'orange')]:
        mean_data = data_dict[mode][f"{loc}_total_E_mean"] / np.abs(data_dict[mode][f"{loc}_total_I_mean"])
        mean_data = moving_average(mean_data, n=smooth_T, axis=0)
        plt.plot(ts, mean_data, color=color, label=loc, linewidth=3, alpha=0.5)
        legend_flag = layout_args[mode][f"ratio"]['legend']

    ax.grid('on')
    ax.set_xlim([750, 1500])
    ax.set_ylim([0, 4])

    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # ax.set_ylabel(loc)
    # ax.set_xlabel("Time (ms)")
    # if legend_flag:
    #     ax.legend()

# plt.savefig("/Users/charlie/Local Documents/Projects/EI Balanced CANN/overleaf_version/AI - formal figs/Fig6/test_no_text.png", dpi=300)
plt.savefig("/Users/charlie/Local Documents/Projects/EI Balanced CANN/overleaf_version/AI - formal figs/Fig6/test.svg", format="svg",transparent=True)
plt.show()