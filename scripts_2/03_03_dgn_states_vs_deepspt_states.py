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
    dataset_df = df[df['dataset'] == str(dataset)]
    if dataset_df.empty:
        continue

    all_dgn_states = sorted(dataset_df['dgn-state'].unique())
    all_deep_states = sorted(set(df['deepspt_state']))

    table_data = defaultdict(list)

    for dgn_state in all_dgn_states:
        sub_df = dataset_df[dataset_df['dgn-state'] == dgn_state]
        if sub_df.empty:
            continue

        roi_groups = sub_df.groupby(['file', 'roi'])

        roi_crosses = []
        for (file, roi), group in roi_groups:
            cross = pd.crosstab(index=group['dgn-state'], columns=group['deepspt_state'], normalize='index')
            cross = cross.reindex(columns=all_deep_states, fill_value=np.nan)
            roi_crosses.append(cross.iloc[0].values)

        if not roi_crosses:
            continue

        arr = np.vstack(roi_crosses)
        means = np.nanmean(arr, axis=0)
        sems = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(arr), axis=0))
        counts = np.sum(~np.isnan(arr), axis=0)

        table_data['Group'].append(dgn_state)
        for idx, deep_state in enumerate(all_deep_states):
            table_data[f'{deep_state} - Mean'].append(round(means[idx], 4) if not np.isnan(means[idx]) else np.nan)
            table_data[f'{deep_state} - SEM'].append(round(sems[idx], 4) if not np.isnan(sems[idx]) else np.nan)
            table_data[f'{deep_state} - N'].append(int(counts[idx]) if counts[idx] > 0 else np.nan)

    result_df = pd.DataFrame(table_data)
    result_df = result_df.sort_values(by='Group')

    # Ordenar columnas: Group first, then sorted deepspt_state chunks
    first_col = ['Group']
    rest_cols = []
    for deep_state in sorted(all_deep_states):
        rest_cols += [f'{deep_state} - Mean', f'{deep_state} - SEM', f'{deep_state} - N']

    result_df = result_df[first_col + rest_cols]

    # Guardar
    result_df.to_excel(f'Results/table_dgn_vs_deepspt_wide_{str(dataset)}.xlsx', index=False)
