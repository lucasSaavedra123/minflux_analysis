import tqdm
import numpy as np
from Trajectory import Trajectory
from DatabaseHandler import DatabaseHandler
from CONSTANTS import *
import matplotlib.pyplot as plt
import matplotlib
from utils import *
from scipy.stats import sem

import matplotlib.pyplot as plt
import matplotlib as mpl

# Cambiar el tamaño de fuente globalmente
# mpl.rcParams.update({'font.size': 35})

DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)

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

for index, dataset in enumerate(new_datasets_list):
    print(dataset)
    basic_query_dict = {'info.dataset': dataset, 'info.immobile': False} if index < len(INDIVIDUAL_DATASETS) else {'info.dataset': dataset[0], 'info.classified_experimental_condition':dataset[1], 'info.immobile': False}

    times_in_states = {
        'file_roi':[],
        'state':[],
        'time': [],
    }

    #eco_cleans = []
    #eco_states = []

    for traj_i, traj_info in tqdm.tqdm(enumerate(Trajectory._get_collection().find(basic_query_dict, {'info.file':1, 'info.roi':1,'t':1,'info.analysis.eco_cleaned':1, 'info.intensity':1, 'info.analysis.eco_states':1}))):
        try:
            #eco_cleans.append([r*275 for r in traj_info['info']['analysis']['eco_cleaned']])
            #eco_states.append(traj_info['info']['analysis']['eco_states'])

            fig, ax0 = plt.subplots(figsize=(8, 8))
            ax1 = ax0.twinx()

            font = {'size':19}
            matplotlib.rc('font', **font)

            eco = np.diff(traj_info['t']) * np.array(traj_info['info']['intensity'])[:-1]

            ax0.plot(eco, linewidth=2, color='blue', label='Noisy ECO', alpha=0.35)
            ax0.plot(np.array(traj_info['info']['analysis']['eco_cleaned'])*275, linewidth=3, color='black', label='Clean ECO')
            ax0.set_ylabel("ECO")
            ax0.set_xlabel("Step")

            yticks = []

            for state in range(5):
                ax1.axhline((state+1)*45.5125, linestyle='--', color='red', linewidth=3)
                yticks.append((state+1)*45.5125)
            ax1.plot((np.array(traj_info['info']['analysis']['eco_states'])+1) * 45.5125, linewidth=3, color='red', label="State")
            ax1.set_ylabel("State")
            ax1.set_xlabel("Step")

            ax1.set_yticks(yticks)
            ax1.set_yticklabels([0,1,2,3,4])

            ax0.set_ylim([0,250])
            ax1.set_ylim([0,250])

            #plt.tight_layout()
            mpl.rcParams.update({
                'font.size': 35,
                'axes.titlesize': 35,
                'axes.labelsize': 35,
                'xtick.labelsize': 30,
                'ytick.labelsize': 30,
                'legend.fontsize': 30,
            })

            plt.savefig(f'D:/UCA/Projects/Post MINFLUX/Graphics/Matplotlib/DGN_ANALYSIS/{dataset}_{traj_i}.svg')
            plt.savefig(f'D:/UCA/Projects/Post MINFLUX/Graphics/Matplotlib/DGN_ANALYSIS/{dataset}_{traj_i}.jpg')
            plt.close(fig)

        except:
            pass

DatabaseHandler.disconnect()