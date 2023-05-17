import numpy as np
import matplotlib.pyplot as plt

models = {
    'slow_eicann': {
        'color': 'black',
        'linestyle': '-',
        'linewidth': 1.5,
        'label': 'Slow EICANN',
        'alpha': 0.2,
    },
    'fast_eicann': {
        'color': 'green',
        'linestyle': '--',
        'linewidth': 1.5,
        'label': 'Fast EICANN',
        'alpha': 0.2,
    }, 
    'slow_cann': {
        'color': 'blue',
        'linestyle': '-',
        'linewidth': 1.5,
        'label': 'Slow CANN',
        'alpha': 0.2,
    }, 
    'fast_cann': {
        'color': 'red',
        'linestyle': '--',
        'linewidth': 1.5,
        'label': 'Fast CANN',
        'alpha': 0.2,
    },
    }
# noise_str = ['0_1', '0_2', '0_3', '0_4', '0_5']
noise_str = ['0_2']

decoding_accuracy = {}
abs_path = '/Users/charlie/Local Documents/Projects/EI Balanced CANN/1d code/experiments/compare_decoding_accuracy/'

for model in models.keys():
    for noise in noise_str:
        file_name = abs_path + 'data/' + model + '_noise_sensitivity_' + noise + '.txt'
        with open(file_name, 'r') as f:
            data = f.read()
        data = data.split()
        # convert noise and data to floats
        data = [float(i) for i in data]
        noise_float = float(noise.replace('_', '.')) 
        decoding_accuracy[(model, noise_float)] = data

# compute mean and std
mean = {}
std = {}
noise_x = [float(noise.replace('_', '.')) for noise in noise_str]
for model in models.keys():
    mean[model] = []
    std[model] = []
    for noise_float in noise_x:
        data = decoding_accuracy[(model, noise_float)]
        mean[model].append(np.mean(data).item())
        std[model].append(np.std(data).item())

# plot wit errorbar
fig, ax = plt.subplots()
for model in models.keys():
    ax.errorbar(noise_x, mean[model], yerr=std[model], **models[model])

ax.set_xlabel('Noise level')
ax.set_ylabel('Decoding Error')
ax.legend()
plt.show()

