import matplotlib.pyplot as plt
import numpy as np


def compare_sudden_change():
    cann_data = np.load("compare_sudden_change/cann_sudden_change.npz")
    coupled_data = np.load("compare_sudden_change/coupled_sudden_change.npz")

    cann_ts, cann_bump, cann_sti = cann_data['ts'], cann_data['bump_pos'], cann_data['input_pos']
    coupled_ts, coupled_bump, coupled_sti = coupled_data['ts'], coupled_data['bump_pos'], coupled_data['input_pos']

    plt.plot(cann_ts, cann_sti, linewidth=2., label="Input Position", alpha=0.5)
    plt.plot(cann_ts, cann_bump, linewidth=2., label="CANN Bump Position", alpha=0.7)
    plt.plot(coupled_ts, coupled_bump, linewidth=2., label="Balanced Bump Position", alpha=0.7)
    plt.xlabel("Time (ms)")
    plt.ylabel("Position (rad)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def compare_tracking_lag():
    cann_data = np.load("compare_tracking_lag/cann_tracking_lag.npz")
    coupled_data = np.load("compare_tracking_lag/coupled_tracking_lag.npz")

    cann_ts, cann_lag = cann_data['ts'], cann_data['lag']
    coupled_ts, coupled_lag = coupled_data['ts'], coupled_data['lag']

    plt.plot(cann_ts, cann_lag, linewidth=2., label="CANN Lag", alpha=0.5)
    plt.plot(coupled_ts, coupled_lag, linewidth=2., label="Balanced Lag", alpha=0.7)
    plt.xlabel("Time (ms)")
    plt.ylabel("Position (rad)")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plt.style.use('ggplot')
    compare_tracking_lag()
