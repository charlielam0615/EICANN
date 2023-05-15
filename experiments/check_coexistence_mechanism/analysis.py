import matplotlib.pyplot as plt
import numpy as np
import os

data = {
    1000: {'Id2E':[], 'E2Id':[], 'Ip2E':[], 'E2Ip':[]},
    2000: {'Id2E':[], 'E2Id':[], 'Ip2E':[], 'E2Ip':[]},
    3000: {'Id2E':[], 'E2Id':[], 'Ip2E':[], 'E2Ip':[]},
    4000: {'Id2E':[], 'E2Id':[], 'Ip2E':[], 'E2Ip':[]},
    5000: {'Id2E':[], 'E2Id':[], 'Ip2E':[], 'E2Ip':[]},
}

abspath = '/Users/charlie/Local Documents/Projects/EI Balanced CANN/1d code/experiments/check_coexistence_mechanism/'
for file_name in os.listdir(abspath+'data/'):
    if file_name.endswith('.npz'):
        t = np.load(abspath+'data/' + file_name)
        data[int(t['net_size'])]['Id2E'].append(t['Id2E'])
        data[int(t['net_size'])]['E2Id'].append(t['E2Id'])
        data[int(t['net_size'])]['Ip2E'].append(t['Ip2E'])
        data[int(t['net_size'])]['E2Ip'].append(t['E2Ip'])

# compute mean and std
mean = {}
std = {}
for net_size in data.keys():
    mean[net_size] = {}
    std[net_size] = {}
    for conn in data[net_size].keys():
        mean[net_size][conn] = np.mean(data[net_size][conn]).item()
        std[net_size][conn] = np.std(data[net_size][conn]).item()

blue = '#2680FF'
red = '#C30017'

colors = [red, red, blue, blue]
linestyles = ['--', '-', '--', '-']
linewidth = [2, 2, 1, 1]
label = {'Id2E': 'Id to E', 'E2Id': 'E to Id', 'Ip2E': 'Ip to E', 'E2Ip': 'E to Ip'}
        
# plot
fig, ax = plt.subplots(1, 1, figsize=(2.5, 2))
for i, conn in enumerate(label.keys()):
    x = []
    y = []
    yerr = []
    for net_size in data.keys():
        x.append(net_size)
        y.append(mean[net_size][conn])
        yerr.append(std[net_size][conn])
    ax.errorbar(x, y, yerr=yerr, color=colors[i], linestyle=linestyles[i], 
                linewidth=linewidth[i], label=label[conn])
    
ax.set_xticks([1000, 2000, 3000, 4000, 5000])
ax.spines[['right', 'top']].set_visible(False)
plt.legend()
plt.show()

