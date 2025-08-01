import tqdm
import numpy as np
from Trajectory import Trajectory
from DatabaseHandler import DatabaseHandler
from CONSTANTS import *
import matplotlib.pyplot as plt
import matplotlib
from utils import *
from scipy.stats import sem


def estimate_hmm_from_multiple_sequences(state_seqs, obs_seqs, n_states):
    A = np.zeros((n_states, n_states))
    pi = np.zeros(n_states)
    all_obs_by_state = {i: [] for i in range(n_states)}

    for states, observations in zip(state_seqs, obs_seqs):
        T = len(states)
        if T < 2:
            continue
        
        pi[states[0]] += 1

        for t in range(T - 1):
            A[states[t], states[t + 1]] += 1
        
        for s, o in zip(states, observations):
            all_obs_by_state[s].append(o)

    pi /= pi.sum()
    A = A / A.sum(axis=1, keepdims=True)

    means = []
    covariances = []

    for i in range(n_states):
        obs_i = np.array(all_obs_by_state[i])
        if obs_i.shape[0] == 0:
            raise ValueError(f"No hay observaciones para el estado {i}")
        mean_i = np.mean(obs_i, axis=0)
        means.append(mean_i)
        covariances.append(obs_i.std())

    return pi, A, means, covariances


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


def segmentar_con_numeros(lista_1, lista_2):
    segmentos = []
    assert len(lista_1) == len(lista_2)

    inicio = 0
    for i in range(1, len(lista_1)):
        if lista_1[i] != lista_1[i-1]:
            segmentos.append((lista_1[inicio], lista_2[inicio:i]))
            inicio = i
    segmentos.append((lista_1[inicio], lista_2[inicio:]))
    return segmentos

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

            segments_and_index = segments_and_index = segmentar_con_numeros(traj_info['info']['analysis']['eco_states'], traj_info['t'][1:])

            fig, ax = plt.subplots(1,2)

            font = {'size':19}
            matplotlib.rc('font', **font)

            eco = np.diff(traj_info['t']) * np.array(traj_info['info']['intensity'])[:-1]

            ax[0].plot(eco, linewidth=2, color='blue', label='Noisy ECO', alpha=0.35)
            ax[0].plot(np.array(traj_info['info']['analysis']['eco_cleaned'])*275, linewidth=2, color='black', label='Clean ECO')
            ax[0].set_ylabel("ECO")
            ax[0].set_xlabel("Step")

            ax[0].set_ylim([0,1000])

            ax[1].axhline(0, linestyle='--', color='black')
            ax[1].axhline(1, linestyle='--', color='black')
            ax[1].axhline(2, linestyle='--', color='black')
            ax[1].axhline(3, linestyle='--', color='black')
            ax[1].axhline(4, linestyle='--', color='black')
            ax[1].plot(traj_info['info']['analysis']['eco_states'], linewidth=2, color='red', label="State")
            ax[1].set_ylabel("State")
            ax[1].set_xlabel("Step")

            ax[0].legend()
            ax[1].legend()

            plt.show()#plt.savefig(f'D:/UCA/Projects/Post MINFLUX/Graphics/Matplotlib/DGN_ANALYSIS/{dataset}_{traj_i}.svg')
            plt.clf()

            for index, segment in segments_and_index:
                if len(segment) > 1:
                    times_in_states['time'].append(segment[-1] - segment[0])
                    times_in_states['state'].append(index)
                    times_in_states['file_roi'].append(str(traj_info['info']['file'])+"_"+str(traj_info['info']['roi']))
        except:
            pass

DatabaseHandler.disconnect()