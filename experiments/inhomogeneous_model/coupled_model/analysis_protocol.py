import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from functools import partial


def different_input_position_protocol(results, resolution):
    bins = np.linspace(-np.pi, np.pi, resolution)

    sns.distplot(results, hist=True, kde=True,
                 bins=bins, color='darkblue',
                 hist_kws={'edgecolor': 'black'},
                 kde_kws={'fill': True, 'linewidth': 3, 'bw_method': 0.05})

    plt.show()
    return


def different_input_position_sanity_check_protocol(results, resolution):
    pass


analy_setup = {
    "different_input_position_protocol": partial(different_input_position_protocol, resolution=100),
    "sanity_check": partial(different_input_position_sanity_check_protocol, resolution=100),
}

if __name__ == "__main__":
    resolution = 300
    bins = np.linspace(-np.pi, np.pi, resolution)
    results = np.load('coupled_result.npy')
    sns.distplot(results, hist=True, kde=True,
                 bins=bins, color='darkblue',
                 hist_kws={'edgecolor': 'black'},
                 kde_kws={'fill': True, 'linewidth': 3, 'bw_method': 0.02})
    plt.show()