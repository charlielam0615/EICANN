import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

resolution = 300
bins = np.linspace(-np.pi, np.pi, resolution)
cann_results = np.load('cann/cann_result_LI_n4500_r(0_9,1_1)_a(0_9,1_1)_T300.npy')
coupled_results = np.load('coupled_model/coupled_result_LI_n4500_r(0_9,1_1)_a(0_9,1_1)_T300.npy')

sns.distplot(cann_results, hist=False, kde=True, label='CANN',
             bins=bins, hist_kws={'edgecolor': 'black'},
             kde_kws={'fill': True, 'linewidth': 2, 'bw_method': 0.05})

sns.distplot(coupled_results, hist=False, kde=True, label='Balanced',
             bins=bins, hist_kws={'edgecolor': 'black'},
             kde_kws={'fill': True, 'linewidth': 2, 'bw_method': 0.05})
plt.legend(loc='lower right')
plt.show()