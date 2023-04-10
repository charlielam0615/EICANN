import matplotlib.pyplot as plt
import numpy as np
import os
import glob


def merge_data(prefix, keys):
    file_list = glob.glob(f'compare_sudden_change/{prefix}*.npz')
    data = {key: [] for key in keys}
    for file_name in file_list:
        single_data = np.load(file_name)
        for key in keys:
            data[key].append(single_data[key])

    for key in keys:
        data[key] = np.stack(data[key])

    return data


def compare_sudden_change():
    cann_data = merge_data(prefix="cann_sudden_change", keys=['ts', 'bump_pos', 'input_pos'])
    coupled_data = merge_data(prefix="coupled_sudden_change", keys=['ts', 'bump_pos', 'input_pos'])

    cann_ts = cann_data['ts'][0, :]
    coupled_ts = coupled_data['ts'][0, :]
    mean_cann_bump = np.mean(cann_data['bump_pos'], axis=0)
    std_cann_bump = np.std(cann_data['bump_pos'], axis=0)
    cann_sti = np.mean(cann_data['input_pos'], axis=0)
    mean_coupled_bump = np.mean(coupled_data['bump_pos'], axis=0)
    std_coupled_bump = np.std(coupled_data['bump_pos'], axis=0)

    figure = plt.figure(figsize=(6, 2))
    # plot stimulus
    plt.plot(cann_ts, cann_sti, linewidth=1.5, label="Input Position", alpha=1.0, color='black')
    # plot cann response
    plt.plot(cann_ts, mean_cann_bump, linewidth=1.5, label="CANN Bump Position", alpha=1.0, color='blue')
    plt.fill_between(cann_ts, mean_cann_bump+std_cann_bump, mean_cann_bump-std_cann_bump,
                     color='blue', alpha=0.2)
    # plot ei balanced cann resposne
    plt.plot(coupled_ts, mean_coupled_bump, linewidth=1.5, label="Balanced Bump Position", alpha=1.0, color='crimson')
    plt.fill_between(coupled_ts, mean_coupled_bump+std_coupled_bump, mean_coupled_bump-std_coupled_bump,
                     color='crimson', alpha=0.2)

    plt.xlabel("Time (ms)")
    plt.ylabel("Position (rad)")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    compare_sudden_change()
