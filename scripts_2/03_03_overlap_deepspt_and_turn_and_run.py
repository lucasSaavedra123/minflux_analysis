"""
All important results like areas, axis lengths, etc. 
are produced within this file.
"""
import pandas as pd
from scipy.stats import sem
import matplotlib.pyplot as plt
import seaborn as sns

from DatabaseHandler import DatabaseHandler
from CONSTANTS import *
from utils import *


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

TO_SAVE_DATASET = [
    'Control',
    'CDx',
    'BTX680R',
    "('Cholesterol and btx', 'BTX680R')",
    'CholesterolPEGKK114',
    "('Cholesterol and btx', 'fPEG-Chol')",
    'BTX640-CHOL-50-nM',
    "('BTX680-fPEG-CHOL-50-nM', 'BTX680R')",
    "('BTX680-fPEG-CHOL-100-nM', 'BTX680R')",
    "('BTX680-fPEG-CHOL-50-nM', 'fPEG-Chol')",
    "('BTX680-fPEG-CHOL-100-nM', 'fPEG-Chol')",
    'CK666-BTX680',
    "('CK666-BTX680-CHOL', 'BTX680R')",
    'CK666-CHOL',
    "('CK666-BTX680-CHOL', 'fPEG-Chol')",
]

run_and_turns = {}
deepspt_states = {}

for index, dataset in enumerate(new_datasets_list):
    print(dataset,index)

    run_and_turns[index] = []
    deepspt_states[index] = []

    basic_query_dict = {'info.dataset': dataset} if index < len(INDIVIDUAL_DATASETS) else {'info.dataset': dataset[0], 'info.classified_experimental_condition':dataset[1]}

    for immobile in [False, True]:
        immobile_str = 'immobile' if immobile else 'not-immobile'

        filter_query = basic_query_dict.copy()
        filter_query.update({
            'info.immobile': immobile,
            'info.analysis.deepspt_segmenter_result':{'$exists':1},
            'info.analysis.run_and_turn_segmenter_result':{'$exists':1},
        })

        projection = {'info.file':1, 'info.roi':1,'info.analysis.deepspt_segmenter_result':1,f'info.analysis.run_and_turn_segmenter_result':1}

        documents = list(Trajectory._get_collection().find(filter_query, projection))

        state_to_label = {1:'turn', 0:'run'}

        for doc in documents:
            run_and_turns[index].extend(doc['info']['analysis']['run_and_turn_segmenter_result'])
            deepspt_states[index].extend(doc['info']['analysis']['deepspt_segmenter_result'])

run_and_turns_avg = []
deepspt_states_avg = []
avg = {
    1:{0:[],1:[],2:[],3:[]},
    0:{0:[],1:[],2:[],3:[]}
}


for index in run_and_turns:
    states = pd.DataFrame({'run_and_turns':run_and_turns[index], 'deepspt_states':deepspt_states[index]})

    for state_run_and_turn in [0,1]:
        for state_deepspt in [0,1,2,3]:
            run_and_turn_states = states[states['run_and_turns']==state_run_and_turn]
            aux_states = run_and_turn_states[run_and_turn_states['deepspt_states']==state_deepspt]

            avg[state_run_and_turn][state_deepspt].append(len(aux_states)/len(run_and_turn_states))

for state_run_and_turn in [0,1]:
    for state_deepspt in [0,1,2,3]:
        a_mean = np.mean(avg[state_run_and_turn][state_deepspt])
        a_std = np.std(avg[state_run_and_turn][state_deepspt])
        avg[state_run_and_turn][state_deepspt] = (a_mean, a_std)

import json
print(json.dumps(avg,sort_keys=True, indent=3))

DatabaseHandler.disconnect()
