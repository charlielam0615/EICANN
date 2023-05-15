"""
Data

N = 1000
===========
400
Total E 14.18116, 13.77297, 13.80864, 13.46646, 13.92584
Total I -6.68270, -7.73446, -6.15748, -6.28955, -5.86351
Total input 7.49846, 6.03850, 7.65116, 7.17692, 8.06233
-----------
100
Total E 3.52094, 3.22952, 3.81789, 3.24236, 3.81216
Total I -3.65362, -3.25850, -3.91178, -3.35445, -3.87836
Total input -0.13268, -0.02897, -0.09390, -0.11209, -0.06621
===========

N = 2000
===========
800
Total E 19.74343, 19.01392, 19.99853, 20.21474, 19.66451
Total I -10.06056, -11.35629, -11.61495, -11.32357, -9.30852
Total input 9.68287, 7.65764, 8.38357, 8.89117, 10.35600
-----------
200
Total E 4.54923, 5.04323, 4.38664, 5.06481, 4.80696
Total I -6.12527, -5.61035, -5.97186, -6.07374, -6.14947
Total input -1.57604, -0.56713, -1.58523, -1.00892, -1.34251
===========

N = 3000
===========
1200
Total E 23.47968, 23.77255, 23.95001, 23.58008, 24.67244
Total I -15.07568, -14.54557, -14.09665, -16.17049, -14.25907
Total input 8.40400, 9.22697, 9.85336, 7.40959, 10.41337
-----------
300
Total E 6.18534, 6.16240, 5.97046, 6.60099, 6.14128
Total I -8.60351, -8.54589, -7.93461, -8.04621, -7.74943
Total input -2.41817, -2.38350, -1.96415, -1.44522, -1.60814
===========


N = 4000
===========
1600
Total E 27.50186, 27.44948, 28.62539, 27.60527, 27.29287
Total I -18.06008, -18.75692, -19.02702, -18.32143, -18.37729
Total input  9.44179, 8.69256, 9.59837, 9.28384, 8.91558
-----------
400
Total E 6.99040, 7.75997, 7.28931, 6.78396, 7.21601
Total I -8.66872, -10.16027, -10.0882, -9.35939, -9.29949
Total input -1.67832, -2.40031, -2.79889, -2.57542, -2.08348
===========


N = 5000
===========
2000
Total E 30.81100, 31.26616, 30.80542, 30.39593, 30.46674
Total I -20.71677, -18.02686, -20.64491, -20.52162, -20.41488
Total input 10.09422, 13.23930, 10.16050, 9.87431, 10.05185
-----------
500
Total E 10.09422, 8.02083, 8.03603, 8.36740, 8.10651
Total I -12.52198, -11.69644, -10.95712, -11.58669, -10.77517
Total input -4.47019, -3.67561, -2.92109, -3.21928, -2.66865
===========

"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import Divider, Size


def data_dict2list(dict_data):
    return np.asarray([dict_data[size] for size in net_size])


def data_process(size_key, E_input, I_input, total_input, neuron_position):
    data[neuron_position]["E"][size_key] = np.asarray(E_input)
    data[neuron_position]["I"][size_key] = np.asarray(I_input)
    data[neuron_position]["total"][size_key] = np.asarray(total_input)
    for part_key in ["E", "I", "total"]:
        data[neuron_position]["mean_" + part_key][size_key] = np.mean(data[neuron_position][part_key][size_key])
        data[neuron_position]["std_" + part_key][size_key] = np.std(data[neuron_position][part_key][size_key])
    return


def data_plot():
    figure = plt.figure(figsize=(7.0, 2.3))

    # The first items are for padding and the second items are for the axes.
    # sizes are in inch.
    h = [Size.Fixed(1.0), Size.Fixed(1.8), Size.Fixed(1.0), Size.Fixed(1.8)]
    v = [Size.Fixed(0.5), Size.Fixed(1.2), Size.Fixed(0.5)]

    divider = Divider(figure, (0, 0, 1, 1), h, v, aspect=False)

    for i, key in enumerate(["center", "peripheral"]):
        # The width and height of the rectangle are ignored.
        ax = figure.add_axes(divider.get_position(),
                      axes_locator=divider.new_locator(nx=1+i*2, ny=1))
        # plot E data
        ax.errorbar(net_size, data_dict2list(data[key]["mean_E"]),
                    yerr=data_dict2list(data[key]["std_E"]),
                    color="blue",
                    linewidth=2.,
                    linestyle="--",
                    label="E")

        # plot I data
        ax.errorbar(net_size, data_dict2list(data[key]["mean_I"]),
                    yerr=data_dict2list(data[key]["std_I"]),
                    color="gray",
                    linewidth=2.,
                    linestyle="--",
                    label="I")

        # plot total data
        ax.errorbar(net_size, data_dict2list(data[key]["mean_total"]),
                    yerr=data_dict2list(data[key]["std_total"]),
                    color="black",
                    linewidth=2.,
                    linestyle="-",
                    label="total")
        ax.set_xlabel("Network size")
        ax.set_ylabel(key.capitalize())
        ax.set_xticks(net_size)
        if key == "center":
            ax.set_ylim([-35, 35])
        else:   
            ax.set_ylim([-15, 15])
        ax.grid()

    plt.legend(loc='center', bbox_to_anchor=(1.4, 0.6), fancybox=True, shadow=True)
    return


net_size = [1000, 2000, 3000, 4000, 5000]

data = {
    "center": {
        "E": {},
        "I": {},
        "total": {},
        "mean_E": {},
        "std_E": {},
        "mean_I": {},
        "std_I": {},
        "mean_total": {},
        "std_total": {},
    },
    "peripheral": {
        "E": {},
        "I": {},
        "total": {},
        "mean_E": {},
        "std_E": {},
        "mean_I": {},
        "std_I": {},
        "mean_total": {},
        "std_total": {},
    },
}

# Network size 1000
data_process(
    size_key=1000,
    E_input=[14.18116, 13.77297, 13.80864, 13.46646, 13.92584],
    I_input=[-6.68270, -7.73446, -6.15748, -6.28955, -5.86351],
    total_input=[7.49846, 6.03850, 7.65116, 7.17692, 8.06233],
    neuron_position="center"
)

data_process(
    size_key=1000,
    E_input=[3.52094, 3.22952, 3.81789, 3.24236, 3.81216],
    I_input=[-3.65362, -3.25850, -3.91178, -3.35445, -3.87836],
    total_input=[-0.13268, -0.02897, -0.09390, -0.11209, -0.06621],
    neuron_position="peripheral"
)

# Network size 2000
data_process(
    size_key=2000,
    E_input=[19.74343, 19.01392, 19.99853, 20.21474, 19.66451],
    I_input=[-10.06056, -11.35629, -11.61495, -11.32357, -9.30852],
    total_input=[9.68287, 7.65764, 8.38357, 8.89117, 10.35600],
    neuron_position="center"
)

data_process(
    size_key=2000,
    E_input=[4.54923, 5.04323, 4.38664, 5.06481, 4.80696],
    I_input=[-6.12527, -5.61035, -5.97186, -6.07374, -6.14947],
    total_input=[-1.57604, -0.56713, -1.58523, -1.00892, -1.34251],
    neuron_position="peripheral"
)

# Network size 3000
data_process(
    size_key=3000,
    E_input=[23.47968, 23.77255, 23.95001, 23.58008, 24.67244],
    I_input=[-15.07568, -14.54557, -14.09665, -16.17049, -14.25907],
    total_input=[8.40400, 9.22697, 9.85336, 7.40959, 10.41337],
    neuron_position="center"
)

data_process(
    size_key=3000,
    E_input=[6.18534, 6.16240, 5.97046, 6.60099, 6.14128],
    I_input=[-8.60351, -8.54589, -7.93461, -8.04621, -7.74943],
    total_input=[-2.41817, -2.38350, -1.96415, -1.44522, -1.60814],
    neuron_position="peripheral"
)

# Network size 4000
data_process(
    size_key=4000,
    E_input=[27.50186, 27.44948, 28.62539, 27.60527, 27.29287],
    I_input=[-18.06008, -18.75692, -19.02702, -18.32143, -18.37729],
    total_input=[9.44179, 8.69256, 9.59837, 9.28384, 8.91558],
    neuron_position="center"
)

data_process(
    size_key=4000,
    E_input=[6.99040, 7.75997, 7.28931, 6.78396, 7.21601],
    I_input=[-8.66872, -10.16027, -10.0882, -9.35939, -9.29949],
    total_input=[-1.67832, -2.40031, -2.79889, -2.57542, -2.08348],
    neuron_position="peripheral"
)

# Network size 5000
data_process(
    size_key=5000,
    E_input=[30.81100, 31.26616, 30.80542, 30.39593, 30.46674, 31.15491, 30.48985, 30.80258],
    I_input=[-20.71677, -18.02686, -20.64491, -20.52162, -20.41488, -20.55890, -20.53783, -18.93875],
    total_input=[10.09422, 13.23930, 10.16050, 9.87431, 10.05185, 10.59602, 9.95202, 11.86383],
    neuron_position="center"
)

data_process(
    size_key=5000,
    E_input=[8.02083, 8.03603, 8.36740, 8.10651, 8.15281, 8.19589, 7.75803],
    I_input=[-11.69644, -10.95712, -11.58669, -10.77517, -10.59796, -11.29882, -9.14917],
    total_input=[-3.67561, -2.92109, -3.21928, -2.66865, -2.44514, -3.10293, -1.39113],
    neuron_position="peripheral"
)

# plot all data
data_plot()
plt.show()

