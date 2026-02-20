import pandas as pd
import matplotlib.pyplot as plt

from DatabaseHandler import DatabaseHandler
from CONSTANTS import *
from utils import *


DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)

INDIVIDUAL_DATASETS = [
    'BTX680R',
    'CDx'
]

new_datasets_list = INDIVIDUAL_DATASETS.copy()

for combined_dataset in [
    'Cholesterol and btx',
]:
    new_datasets_list.append((combined_dataset, BTX_NOMENCLATURE))


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

    def mean_sem(series):
        return pd.Series({
            'mean': series.mean(),
            'sem': series.std(ddof=1) / np.sqrt(len(series))
        })

    avg_dgn = clustered_dgn_percentages.groupby(['dgn-state', 'clustering_state'])['clustered_pct_within_dgn'] \
        .apply(mean_sem).reset_index()

    avg_deepspt = clustered_deepspt_percentages.groupby(['deepspt', 'clustering_state'])['clustered_pct_within_deepspt'] \
        .apply(mean_sem).reset_index()

    print("\nPromedio porcentajes por estado dgn-state (con SEM):")
    print(avg_dgn)

    print("\nPromedio porcentajes por estado deepspt (con SEM):")
    print(avg_deepspt)

    total_clustered_pct = (df['clustering_state'] == 'clustered').mean() * 100
    print(f"\nTotal clustered percentage: {total_clustered_pct:.2f}%")

    clustered_by_roi_file = df.groupby(['file', 'roi'])['clustering_state'] \
        .apply(lambda x: (x == 'clustered').mean() * 100) \
        .reset_index(name='clustered_pct')

    avg_clustered_stats = mean_sem(clustered_by_roi_file['clustered_pct'])

    print(f"Average percentage per ROI and File: {avg_clustered_stats['mean']:.2f}% (SEM={avg_clustered_stats['sem']:.2f}%)")

DatabaseHandler.disconnect()
