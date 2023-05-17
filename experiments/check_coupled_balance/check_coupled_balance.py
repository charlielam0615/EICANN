import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import Divider, Size
from tqdm import tqdm

model = "balanced"
network_size = [1000, 2000, 3000, 4000, 5000]
loc_list = ['center', 'peripheral']
abspath = "/Users/charlie/Local Documents/Projects/EI Balanced CANN/1d code/experiments/check_coupled_balance/data"


# avaliable data: 
# 'E2E_s', 'E2E_f', 'Fc', 'SI', 'I2E_s', 'I2E_f', 'leak', 
# 'total_E', 'total_I', 'total', 'index', 'stable_idx'
retrieve_item = ['E2E_s', 'E2E_f', 'Fc', 'SI', 'I2E_s', 'I2E_f', 'leak', 'total_E', 'total_I', 'total']
# plot_item = ['total_I', 'I2E_s', 'I2E_f', 'SI']
plot_item = ['total_E', 'total_I', 'total']
data = {
    size:{
        item:{
            'center':{
                "mean": None,
                "std": None
            }, 
            'peripheral':{
                "mean": None,
                "std": None
            }
        } for item in retrieve_item
    } for size in network_size
}

# iterate over data
for size in network_size:
    for loc in loc_list:
        _stack = {}
        for file_name in os.listdir(f"{abspath}/{model}/{size}"):
            if file_name.endswith('.npz') and file_name.startswith(loc):
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

        data[size]['total_E'][loc]["mean"] = np.mean(total_E)
        data[size]['total_E'][loc]["std"] = np.std(total_E)
        data[size]['total_I'][loc]["mean"] = np.mean(total_I)
        data[size]['total_I'][loc]["std"] = np.std(total_I)
        data[size]['total'][loc]["mean"] = np.mean(total)
        data[size]['total'][loc]["std"] = np.std(total)

        remaining_item = [item for item in retrieve_item if item not in ['total_E', 'total_I', 'total']]
        for item in remaining_item:
            data[size][item][loc]["mean"] = np.mean(_stack[item])
            data[size][item][loc]["std"] = np.std(_stack[item])

# plot
fig = plt.figure(figsize=(6, 4.5))
ax = fig.add_subplot(1, 2, 1)
for item in plot_item:
    _mean = []
    _std = []
    for size in network_size:
        _mean.append(data[size][item]['center']['mean'])
        _std.append(data[size][item]['center']['std'])
    
    # plot with error bar
    ax.errorbar(network_size, _mean, yerr=_std, capsize=5, label=f"{item} center")
ax.grid()
ax.legend()

ax = fig.add_subplot(1, 2, 2)
for item in plot_item:
    _mean = []
    _std = []
    for size in network_size:
        _mean.append(data[size][item]['peripheral']['mean'])
        _std.append(data[size][item]['peripheral']['std'])
    
    # plot with error bar
    ax.errorbar(network_size, _mean, yerr=_std, capsize=5, label=f"{item} peripheral")
ax.grid()
ax.legend()

plt.show()
                


