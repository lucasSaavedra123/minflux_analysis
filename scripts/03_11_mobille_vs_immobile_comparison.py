import pandas as pd
import tqdm
from CONSTANTS import *
from DatabaseHandler import DatabaseHandler
from Trajectory import Trajectory
import numpy as np
from scipy.stats import sem
from utils import extract_dataset_file_roi_file, both_trajectories_intersect
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.spatial import ConvexHull, QhullError
import matplotlib
import matplotlib as mpl

mpl.rcParams['axes.linewidth'] = 2
DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)

INDIVIDUAL_DATASETS = [
    'Control',
    'CDx',
    'BTX680R',
    'CholesterolPEGKK114',
    'CK666-BTX680',
    'CK666-CHOL',
    'BTX640-CHOL-50-nM',
    'BTX640-CHOL-50-nM-LOW-DENSITY',
]

new_datasets_list = INDIVIDUAL_DATASETS.copy()

for combined_dataset in [
    'Cholesterol and btx',
    'CK666-BTX680-CHOL',
    'BTX680-fPEG-CHOL-50-nM',
    'BTX680-fPEG-CHOL-100-nM',
]:
    new_datasets_list.append((combined_dataset, BTX_NOMENCLATURE))
    new_datasets_list.append((combined_dataset, CHOL_NOMENCLATURE))


for index, dataset in enumerate(new_datasets_list):
    SEARCH_FIELD = {'info.dataset': dataset} if index < len(INDIVIDUAL_DATASETS) else {'info.dataset': dataset[0], 'info.classified_experimental_condition':dataset[1]}
    trajectories_info = list(Trajectory._get_collection().find(SEARCH_FIELD, {'info.analysis.betha':1,'info.immobile':1, 'x':1, 'y':1}))

    m = []
    im = []

    for info in trajectories_info:
        raw_trajectory = np.zeros((len(info['x']), 2))
        raw_trajectory[:,0] = info['x']
        raw_trajectory[:,1] = info['y']
        try:
            area = ConvexHull(raw_trajectory).volume
            #area = info['info']['analysis']['betha']
            if info['info']['immobile']:
                im.append(area)
            else:
                m.append(area)
        except:
            pass

    m = np.log10(m)
    im = np.log10(im)

    ax = plt.subplot()

    font = {'size'   : 18}

    matplotlib.rc('font', **font)

    ax.hist(m, color='red', bins=np.arange(min(m), max(m) + 0.01, 0.1), alpha=0.5)#, weights=np.zeros_like(m) + 1. / m.size)
    ax.hist(im, color='black',  bins=np.arange(min(im), max(im) + 0.01, 0.1), alpha=0.5)#, weights=np.zeros_like(im) + 1. / im.size)
    plt.tight_layout()
    plt.subplots_adjust(left=0.407, bottom=0.208)
    plt.xticks(fontsize=35)
    plt.yticks(fontsize=35)
    ax.tick_params(which='major', direction='out', length=6, width=2, pad=5)
    ax.tick_params(which='minor', direction='out', length=3, width=2, pad=5)
    plt.savefig(f'areas_{dataset}.jpg', dpi=300)
    #plt.show()
    plt.clf()

DatabaseHandler.disconnect()