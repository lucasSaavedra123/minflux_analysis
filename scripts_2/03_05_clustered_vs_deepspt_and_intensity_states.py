import pandas as pd
import matplotlib.pyplot as plt

from DatabaseHandler import DatabaseHandler
from CONSTANTS import *
from utils import *


DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)

INDIVIDUAL_DATASETS = [
    'BTX680R',
]

new_datasets_list = INDIVIDUAL_DATASETS.copy()

for combined_dataset in [
    'Cholesterol and btx',
]:
    new_datasets_list.append((combined_dataset, BTX_NOMENCLATURE))

TO_SAVE_DATASET = [
    'BTX680R',
    "('Cholesterol and btx', 'BTX680R')",
]


for index, dataset in enumerate(new_datasets_list):
    df = []

    print(dataset,index)

    basic_query_dict = {'info.dataset': dataset} if index < len(INDIVIDUAL_DATASETS) else {'info.dataset': dataset[0], 'info.classified_experimental_condition':dataset[1]}
    filter_query = basic_query_dict.copy()
    filter_query.update({
        'info.immobile': False,
    })

    projection = {
        'info.file':1,
        'info.roi':1,
        'info.analysis.deepspt_segmenter_result':1,
        'info.analysis.eco_states':1,
        'info.analysis.storm_based_analysis_cluster':1
    }

    documents = list(Trajectory._get_collection().find(filter_query, projection))

    for doc in documents:
        try:
            df.append({
                'dataset': str(dataset),
                'file': doc['info']['file'],
                'roi': doc['info']['roi'],
                'deepspt': max(doc['info']['analysis']['deepspt_segmenter_result'], key=doc['info']['analysis']['deepspt_segmenter_result'].count),
                'dgn-state': max(doc['info']['analysis']['eco_states'], key=doc['info']['analysis']['eco_states'].count),
                'clustering_state': doc['info']['analysis']['storm_based_analysis_cluster'],
            })
        except KeyError:
            pass

    df = pd.DataFrame(df)

    clustered_dgn_percentages = df.groupby(['file', 'roi', 'dgn-state'])['clustering_state'] \
        .value_counts(normalize=True) \
        .mul(100) \
        .rename('clustered_pct_within_dgn') \
        .reset_index()

    clustered_deepspt_percentages = df.groupby(['file', 'roi', 'deepspt'])['clustering_state'] \
        .value_counts(normalize=True) \
        .mul(100) \
        .rename('clustered_pct_within_deepspt') \
        .reset_index()

    print("\nPorcentajes clustered/unclustered dentro de cada dgn-state:")
    print(clustered_dgn_percentages)

    print("\nPorcentajes clustered/unclustered dentro de cada intensidad (deepspt):")
    print(clustered_deepspt_percentages)

DatabaseHandler.disconnect()
