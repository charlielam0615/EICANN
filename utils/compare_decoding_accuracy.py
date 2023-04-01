"""
Data

Fast CANN
===============
Noise = 0.1
0.018928463, 0.020199658, 0.017535334
---------------
Noise = 0.2
0.06885283, 0.063408084, 0.07103434
---------------
Noise = 0.3
0.19791338, 0.17883591, 0.18705745
---------------
Noise = 0.4
0.3518065, 0.31564468, 0.367718
---------------
Noise = 0.5
0.5072848, 0.5239094, 0.47308773
---------------



Fast CANN + EI balance
===============
Noise = 0.1
0.014067223, 0.014991953, 0.015217036
---------------
Noise = 0.2
0.053741388, 0.053977765, 0.043768976
---------------
Noise = 0.3
0.123121254, 0.13368577, 0.1161965
---------------
Noise = 0.4
0.2479118, 0.23233795, 0.2536413
---------------
Noise = 0.5
0.44252056, 0.36390838, 0.35949895
---------------



Slow CANN
===============
Noise = 0.1
0.010111801, 0.010650488, 0.011178165
---------------
Noise = 0.2
0.021412248, 0.0194644, 0.019985432
---------------
Noise = 0.3
0.043533947, 0.041882724, 0.04501113
---------------
Noise = 0.4
0.085150704, 0.09398377, 0.09624506
---------------
Noise = 0.5
0.19255184, 0.21667916, 0.21936971
---------------



Slow CANN + EI balance
===============
Noise = 0.1
0.009588719, 0.0094349375, 0.00902201
---------------
Noise = 0.2
0.0151795605, 0.013791247, 0.014179827
---------------
Noise = 0.3
0.001417, 0.024519978, 0.025443543
---------------
Noise = 0.4
0.0505553, 0.05302107, 0.04568492
---------------
Noise = 0.5
0.13372952, 0.09256403, 0.11448792
---------------


"""

import numpy as np
import matplotlib.pyplot as plt
import brainpy as bp


def data_process(noise_key, speed_key, balance_key, values):
    if not isinstance(noise_key, str):
        noise_key = str(noise_key)
    data[speed_key][balance_key]["MSE"][noise_key] = np.asarray(values)
    data[speed_key][balance_key]["std"][noise_key] = np.std(data[speed_key][balance_key]["MSE"][noise_key])
    data[speed_key][balance_key]["mean"][noise_key] = np.mean(data[speed_key][balance_key]["MSE"][noise_key])

    return


def data_dict2list(dict_data):
    return np.asarray([dict_data[str(l)] for l in noise_level])


noise_level = [0.1, 0.2, 0.3, 0.4, 0.5]
data = {
    "Fast": {
        "Balanced": {
            "MSE": {},
            "mean": {},
            "std": {},
            "plot_args": {
                'color': 'blue',
                'linewidth': 1.5,
                'linestyle': '-',
            },
        },
        "Unbalanced": {
            "MSE": {},
            "mean": {},
            "std": {},
            "plot_args": {
                'color': 'blue',
                'linewidth': 1.5,
                'linestyle': '--',
            },
        },
    },
    "Slow": {
        "Balanced": {
            "MSE": {},
            "mean": {},
            "std": {},
            "plot_args": {
                'color': 'black',
                'linewidth': 1.5,
                'linestyle': '-',
            },
        },
        "Unbalanced": {
            "MSE": {},
            "mean": {},
            "std": {},
            "plot_args": {
                'color': 'black',
                'linewidth': 1.5,
                'linestyle': '--',
            },
        },
    },
}

# ==========================
# Fast CANN
# ==========================
data_process(
    noise_key=0.1,
    speed_key='Fast',
    balance_key='Unbalanced',
    values=[0.018928463, 0.020199658, 0.017535334],
)

data_process(
    noise_key=0.2,
    speed_key='Fast',
    balance_key='Unbalanced',
    values=[0.06885283, 0.063408084, 0.07103434]
)

data_process(
    noise_key=0.3,
    speed_key='Fast',
    balance_key='Unbalanced',
    values=[0.19791338, 0.17883591, 0.18705745]
)

data_process(
    noise_key=0.4,
    speed_key='Fast',
    balance_key='Unbalanced',
    values=[0.3518065, 0.31564468, 0.367718]
)

data_process(
    noise_key=0.5,
    speed_key='Fast',
    balance_key='Unbalanced',
    values=[0.5072848, 0.5239094, 0.47308773]
)

# ==========================
# Fast CANN + E/I Balance
# ==========================
data_process(
    noise_key=0.1,
    speed_key='Fast',
    balance_key='Balanced',
    values=[0.014067223, 0.014991953, 0.015217036],
)

data_process(
    noise_key=0.2,
    speed_key='Fast',
    balance_key='Balanced',
    values=[0.053741388, 0.053977765, 0.043768976],
)

data_process(
    noise_key=0.3,
    speed_key='Fast',
    balance_key='Balanced',
    values=[0.123121254, 0.13368577, 0.1161965],
)

data_process(
    noise_key=0.4,
    speed_key='Fast',
    balance_key='Balanced',
    values=[0.2479118, 0.23233795, 0.2536413],
)

data_process(
    noise_key=0.5,
    speed_key='Fast',
    balance_key='Balanced',
    values=[0.44252056, 0.36390838, 0.35949895],
)


# ==========================
# Slow CANN
# ==========================
data_process(
    noise_key=0.1,
    speed_key='Slow',
    balance_key='Unbalanced',
    values=[0.010111801, 0.010650488, 0.011178165],
)

data_process(
    noise_key=0.2,
    speed_key='Slow',
    balance_key='Unbalanced',
    values=[0.021412248, 0.0194644, 0.019985432]
)

data_process(
    noise_key=0.3,
    speed_key='Slow',
    balance_key='Unbalanced',
    values=[0.043533947, 0.041882724, 0.04501113]
)

data_process(
    noise_key=0.4,
    speed_key='Slow',
    balance_key='Unbalanced',
    values=[0.085150704, 0.09398377, 0.09624506]
)

data_process(
    noise_key=0.5,
    speed_key='Slow',
    balance_key='Unbalanced',
    values=[0.19255184, 0.21667916, 0.21936971]
)

# ==========================
# Slow CANN + EI balance
# ==========================
data_process(
    noise_key=0.1,
    speed_key='Slow',
    balance_key='Balanced',
    values=[0.009588719, 0.0094349375, 0.00902201],
)

data_process(
    noise_key=0.2,
    speed_key='Slow',
    balance_key='Balanced',
    values=[0.0151795605, 0.013791247, 0.014179827],
)

data_process(
    noise_key=0.3,
    speed_key='Slow',
    balance_key='Balanced',
    values=[0.001417, 0.024519978, 0.025443543],
)

data_process(
    noise_key=0.4,
    speed_key='Slow',
    balance_key='Balanced',
    values=[0.0505553, 0.05302107, 0.04568492],
)

data_process(
    noise_key=0.5,
    speed_key='Slow',
    balance_key='Balanced',
    values=[0.13372952, 0.09256403, 0.11448792],
)

if __name__ == "__main__":
    fig, gs = bp.visualize.get_figure(1, 1, 2, 6)
    # subplot 1: raster E plot
    ax = fig.add_subplot(gs[0, 0])
    speed_keys = ['Fast', 'Slow']
    balance_keys = ['Balanced', 'Unbalanced']

    for speed_key in speed_keys:
        for balance_key in balance_keys:
            ax.errorbar(noise_level,
                        data_dict2list(data[speed_key][balance_key]['mean']),
                        yerr=data_dict2list(data[speed_key][balance_key]['std']),
                        color=data[speed_key][balance_key]['plot_args']['color'],
                        linewidth=data[speed_key][balance_key]['plot_args']['linewidth'],
                        linestyle=data[speed_key][balance_key]['plot_args']['linestyle'],
                        label=' '.join((balance_key, speed_key, 'CANN'))
                        )

    ax.set_xlabel("Noise level")
    ax.set_ylabel("Decoding Error")
    ax.set_xticks(noise_level)
    ax.grid()

    plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), fancybox=True, shadow=True)
    plt.tight_layout()

    plt.show()