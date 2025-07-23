import tqdm
import numpy as np
from Trajectory import Trajectory
from DatabaseHandler import DatabaseHandler
from CONSTANTS import *
import matplotlib.pyplot as plt
import matplotlib
from utils import *
from sklearn.mixture import GaussianMixture
from scipy.signal import savgol_filter
import pickle

DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)

# Define simple gaussian
def gauss_function(x, amp, x0, sigma):
    return amp * np.exp(-(x - x0) ** 2. / (2. * sigma ** 2.))


labels = ['CF®680R', 'fPEG-Chol', 'CF®680R(+fPEG-Chol)', 'fPEG-Chol(+CF®680R)']

INDIVIDUAL_DATASETS = [
    'BTX680R',
    #'CK666-BTX680',
    'CholesterolPEGKK114',
    #'CK666-CHOL',
    #'Control',
    #'BTX640-CHOL-50-nM',
    #'BTX640-CHOL-50-nM-LOW-DENSITY',
    #'CDx',
]

new_datasets_list = INDIVIDUAL_DATASETS.copy()

for combined_dataset in [
    'Cholesterol and btx',
    #'CK666-BTX680-CHOL',
    #'BTX680-fPEG-CHOL-50-nM',
    #'BTX680-fPEG-CHOL-100-nM',
]:
    new_datasets_list.append((combined_dataset, BTX_NOMENCLATURE))
    new_datasets_list.append((combined_dataset, CHOL_NOMENCLATURE))

means = []
stds = []

for index, dataset in enumerate(new_datasets_list):
    print(dataset)
    basic_query_dict = {'info.dataset': dataset, 'info.immobile': False} if index < len(INDIVIDUAL_DATASETS) else {'info.dataset': dataset[0], 'info.classified_experimental_condition':dataset[1], 'info.immobile': False}
    
    all_ecos = []
    individual_ecos = []

    for traj_i, traj in tqdm.tqdm(enumerate(Trajectory._get_collection().find(basic_query_dict, {'_id':1, 't':1, 'info.intensity':1}))):
        t = np.array(traj['t'])
        efo = np.array(traj['info']['intensity'])
        eco = np.diff(t) * efo[:-1]

        if len(eco) > 0:
            all_ecos.extend(savgol_filter(eco, 9, 1).tolist())
            individual_ecos.append(eco)

        font = {'size':22}
        matplotlib.rc('font', **font)

        plt.plot(eco)
        plt.xlabel("Step")
        plt.ylabel("ECO")
        plt.tight_layout()
        plt.savefig(f"traces/eco_{traj_i}_{dataset}.svg")
        plt.clf()

    samples = np.array(all_ecos)
    samples = samples[samples < 250]
    binwidth = 3

    gmm = GaussianMixture(n_components=3, covariance_type="full")
    gmm = gmm.fit(X=np.expand_dims(samples, 1))

    gmm_x = np.linspace(0, 750, 750//2)
    gmm_y = np.exp(gmm.score_samples(gmm_x.reshape(-1, 1)))

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[8, 5])

    colors = ["black", "green", "red", "orange"]

    for g_i, (m, c, w) in enumerate(zip(gmm.means_.ravel(), gmm.covariances_.ravel(), gmm.weights_.ravel())):
        gauss = gauss_function(x=gmm_x, amp=1, x0=m, sigma=np.sqrt(c))
        ax.plot(gmm_x, gauss / np.trapz(gauss, gmm_x) * w, color=colors[g_i], lw=5, label=fr'$\sigma={np.round(np.sqrt(c),2)}, \mu={np.round(m,2)}$', linestyle="dashed")

    ax.hist(samples, bins=np.arange(min(all_ecos), max(all_ecos) + binwidth, binwidth), density=True, alpha=1, color="#00A7B1")
    
    font = {'size':22}
    matplotlib.rc('font', **font)

    ax.set_ylabel("Probability density", fontsize=19)
    ax.set_xlabel("ECO", fontsize=19)
    ax.set_xlim([0,250])
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"D:/UCA/Projects/Post MINFLUX/Graphics/Matplotlib/gaussian_mixture_analysis{dataset}.svg")
    plt.clf()

    #plt.hist(all_ecos, bins=np.arange(min(all_ecos), max(all_ecos) + binwidth, binwidth))
    #plt.title(labels[index], fontsize=19)
    #plt.xlabel("ECO", fontsize=19)
    #plt.ylabel("Count", fontsize=19)
    #plt.tight_layout()
    #plt.savefig(f"count_efo_{dataset}.svg")
    #plt.clf()

    with open(f"efo_{dataset}.list", "wb") as file:
        pickle.dump(individual_ecos,file)

    means.append(np.quantile(all_ecos, 0.5))
    stds.append(np.std(all_ecos))

print(np.mean(means), np.mean(stds))

DatabaseHandler.disconnect()