"""
All important results like areas, axis lengths, etc. 
are produced within this file.
"""
import pandas as pd
import matplotlib.pyplot as plt

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

df = []

for index, dataset in enumerate(new_datasets_list):
    print(dataset,index)

    for dgn_state in range(5):
        basic_query_dict = {'info.dataset': dataset} if index < len(INDIVIDUAL_DATASETS) else {'info.dataset': dataset[0], 'info.classified_experimental_condition':dataset[1]}

        filter_query = basic_query_dict.copy()
        filter_query.update({
            'info.immobile': False,
            f'info.analysis.{dgn_state}-deepspt_state':{'$exists':1},
        })

        projection = {'info.file':1, 'info.roi':1,f'info.analysis.{dgn_state}-deepspt_state':1}

        documents = list(Trajectory._get_collection().find(filter_query, projection))

        state_to_label = {1:'turn', 0:'run'}

        for doc in documents:
            for deep_state in doc['info']['analysis'][f'{dgn_state}-deepspt_state']:
                if deep_state is not None:
                    df.append({
                        'dataset': str(dataset),
                        'file': doc['info']['file'],
                        'roi': doc['info']['roi'],
                        'deepspt_state': deep_state,
                        'dgn-state': dgn_state,
                    })

DatabaseHandler.disconnect()

"""
0->Brownian
1->Superdiffusive
2->Confined
3->Subdiffusive
"""

df = pd.DataFrame(df)


for dataset in new_datasets_list:
    dataset_df = df[df['dataset']==str(dataset)]
    groups = dataset_df.groupby(['file', 'roi'])

    crosses = []

    for (file, roi), group in groups:
        cross = pd.crosstab(group['dgn-state'], group['deepspt_state'], normalize='index')
        crosses.append(cross)

    all_states = set()
    for cross in crosses:
        all_states.update(cross.index)
        all_states.update(cross.columns)

    all_states = sorted(all_states)

    aligned_crosses = [
        cross.reindex(index=all_states, columns=all_states, fill_value=0)
        for cross in crosses
    ]

    if len(aligned_crosses) == 0:
        continue

    average_cross = sum(aligned_crosses) / len(aligned_crosses)

    arr = np.stack([df.values for df in aligned_crosses])  # shape: (n, rows, cols)

    mean = arr.mean(axis=0)

    # SEM = std / sqrt(n)
    sem = arr.std(axis=0, ddof=1) / np.sqrt(len(arr))

    average_cross = pd.DataFrame(mean, index=aligned_crosses[0].index, columns=aligned_crosses[0].columns)
    sem_cross = pd.DataFrame(sem, index=aligned_crosses[0].index, columns=aligned_crosses[0].columns)

    mean_values = average_cross.values
    sem_values = sem_cross.values

    formatter = np.vectorize(lambda m, s: f"{m:.3f} ± {s:.3f}")
    combined_values = formatter(mean_values, sem_values)

    combined_df = pd.DataFrame(combined_values, index=average_cross.index, columns=average_cross.columns)

    combined_df.to_csv(f'Results/dgn_vs_deepspt_states_{dataset}.csv')
