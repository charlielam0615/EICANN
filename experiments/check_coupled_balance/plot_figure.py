import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import Divider, Size
from tqdm import tqdm

network_size = [1000, 2000, 3000, 4000, 5000]
abspath = "/Users/charlie/Local Documents/Projects/EI Balanced CANN/1d code/experiments/check_coupled_balance/data"

model_list = ['balanced', 'unbalanced']
# avaliable data: 
# 'E2E_s', 'E2E_f', 'Fc', 'SI', 'I2E_s', 'I2E_f', 'leak', 
# 'total_E', 'total_I', 'total', 'index', 'stable_idx'
retrieve_item = ['E2E_s', 'E2E_f', 'Fc', 'SI', 'I2E_s', 'I2E_f', 'leak', 'total_E', 'total_I', 'total']
# plot_item = ['total_I', 'I2E_s', 'I2E_f', 'SI']
plot_item = ['total_E', 'total_I', 'total']
data_fig1 = {
    size:{
        item:{
            'balanced':{
                "mean": None,
                "std": None
            }, 
            'unbalanced':{
                "mean": None,
                "std": None
            }
        } for item in retrieve_item
    } for size in network_size
}

data_fig2 = {
    "balanced":{
        "total_E": None,
        "total_I": None,
        "SI": None,
        "I2E_s": None,
        "I2E_f": None,
    },
    "unbalanced":{
        "total_E": None,
        "total_I": None,
        "SI": None,
        "I2E_s": None,
        "I2E_f": None,
    }
}

# iterate over data
for size in network_size:
    for model in model_list:
        _stack = {}
        for file_name in os.listdir(f"{abspath}/{model}/{size}"):
            if file_name.endswith('.npz') and file_name.startswith("center"):
                np_data = np.load(f"{abspath}/{model}/{size}/{file_name}")
                start_idx = np_data['stable_idx']
                for item in retrieve_item:
                    if item not in _stack.keys():
                        _stack[item] = np_data[item][start_idx:]
                    else:
                        _stack[item] = np.concatenate((_stack[item], np_data[item][start_idx:]), axis=0)

        total_E = _stack['E2E_s'] + _stack['Fc'] + _stack['E2E_f']
        SI = 1 * (_stack['E2E_s'] + _stack['Fc'] + _stack['E2E_f']) * _stack['I2E_s']
        total_I = _stack['I2E_s'] + _stack['I2E_f'] + _stack['leak'] + SI
        total = total_E + total_I

        data_fig1[size]['total_E'][model]["mean"] = np.mean(total_E)
        data_fig1[size]['total_E'][model]["std"] = np.std(total_E)
        data_fig1[size]['total_I'][model]["mean"] = np.mean(total_I)
        data_fig1[size]['total_I'][model]["std"] = np.std(total_I)
        data_fig1[size]['total'][model]["mean"] = np.mean(total)
        data_fig1[size]['total'][model]["std"] = np.std(total)

        remaining_item = [item for item in retrieve_item if item not in ['total_E', 'total_I', 'total']]
        for item in remaining_item:
            data_fig1[size][item][model]["mean"] = np.mean(_stack[item])
            data_fig1[size][item][model]["std"] = np.std(_stack[item])


for model in model_list:
    _stack = {}
    for file_name in os.listdir(f"{abspath}/.."):
        if file_name.endswith('.npz') and file_name.startswith(model):
            np_data = np.load(f"{abspath}/../{file_name}")
            start_idx = np_data['stable_idx']
            for item in ['E2E_s', 'E2E_f', 'Fc', 'I2E_s', 'I2E_f', 'leak']:
                    if item not in _stack.keys():
                        _stack[item] = np_data[item][start_idx:]
                    else:
                        _stack[item] = np.concatenate((_stack[item], np_data[item][start_idx:]), axis=1)

    total_E = _stack['E2E_s'] + _stack['Fc'] + _stack['E2E_f']
    SI = 1 * (_stack['E2E_s'] + _stack['Fc'] + _stack['E2E_f']) * _stack['I2E_s']
    total_I = _stack['I2E_s'] + _stack['I2E_f'] + _stack['leak'] + SI
    total = total_E + total_I

    data_fig2[model]['I2E_f'] = np.mean(_stack['I2E_f'], axis=0)
    data_fig2[model]['I2E_s'] = np.mean(_stack['I2E_s'], axis=0)
    data_fig2[model]['SI'] = np.mean(SI, axis=0)
    data_fig2[model]['total_I'] = np.mean(total_I, axis=0)
    data_fig2[model]['total_E'] = np.mean(total_E, axis=0)


fig = plt.figure(figsize=(4., 4.2))

# The first items are for padding and the second items are for the axes.
# sizes are in inch.
h = [Size.Fixed(1.0), Size.Fixed(2.)]
v = [Size.Fixed(0.5), Size.Fixed(1.2), Size.Fixed(0.5), Size.Fixed(1.2),]

divider = Divider(fig, (0, 0, 1, 1), h, v, aspect=False)
# The width and height of the rectangle are ignored.

# plot 1: compare center total current in balanced and unbalanced model
ax1 = fig.add_axes(divider.get_position(),
                    axes_locator=divider.new_locator(nx=1, ny=3))
balanced_mean = [data_fig1[size]['total']['balanced']['mean'] for size in network_size]
balanced_std = [data_fig1[size]['total']['balanced']['std'] for size in network_size]
unbalanced_mean = [data_fig1[size]['total']['unbalanced']['mean'] for size in network_size]
unbalanced_std = [data_fig1[size]['total']['unbalanced']['std'] for size in network_size]
ax1.errorbar(network_size, balanced_mean, yerr=balanced_std, fmt='-o', color='C0', label='          ')
ax1.errorbar(network_size, unbalanced_mean, yerr=unbalanced_std, fmt='-o', color='C1', label='         ')
ax1.legend()
ax1.set_xticks(network_size)
ax1.set_xticklabels(network_size)

# plot 2: plot current vs. index
ax2 = fig.add_axes(divider.get_position(),
                    axes_locator=divider.new_locator(nx=1, ny=1))
prob_index = [100, 200, 300, 400, 500, 600, 700]
df = pd.DataFrame(
    {
        r"SI": [data_fig2['balanced']['SI'][i] for i in prob_index],
        r"$\Gamma^{I_p}$": [data_fig2['balanced']['I2E_s'][i] for i in prob_index],
        r"$\Omega^{I_d}$": [data_fig2['balanced']['I2E_f'][i] for i in prob_index],
    },
    index = prob_index,
)

df.plot(kind='bar', stacked=True, color=['orange', 'green', 'skyblue'], ax=ax2)
ax2.set_ylim([-10, 0])
ax2.set_xticks([0, 3, 6])
ax2.set_yticks([-10, -5, 0])
ax2.legend(ncol=len(df.columns))

plt.show()