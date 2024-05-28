"""
All important results like areas, axis lengths, etc. 
are produced within this file.
"""
import pandas as pd

from DatabaseHandler import DatabaseHandler
from CONSTANTS import *
from utils import *

APPLY_GS_CRITERIA = True

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
    print(dataset,index)

    basic_query_dict = {'info.dataset': dataset} if index < len(INDIVIDUAL_DATASETS) else {'info.dataset': dataset[0], 'info.classified_experimental_condition':dataset[1]}
    with pd.ExcelWriter(f"./Results/{dataset}_{index}_gs_{APPLY_GS_CRITERIA}_basic_information.xlsx") as writer:
        filter_query = basic_query_dict.copy()
        filter_query.update({'info.immobile': False} if APPLY_GS_CRITERIA else {})

        trajectory_analysis_dataframe = get_dataframe_of_trajectory_analysis_data(filter_query)

        trajectory_analysis_dataframe['k'] = 10**remove_outliers_from_set_of_values_of_column(np.log10(trajectory_analysis_dataframe['k']))
        trajectory_analysis_dataframe['residence_time'] = 10**remove_outliers_from_set_of_values_of_column(np.log10(trajectory_analysis_dataframe['residence_time']))
        trajectory_analysis_dataframe['inverse_residence_time'] = 10**remove_outliers_from_set_of_values_of_column(np.log10(trajectory_analysis_dataframe['inverse_residence_time']))

        trajectory_analysis_dataframe['residence_ratios'] = trajectory_analysis_dataframe['residence_time']/(trajectory_analysis_dataframe['residence_time']+trajectory_analysis_dataframe['inverse_residence_time'])
        trajectory_analysis_dataframe[f'{BTX_NOMENCLATURE}_overlap_confinement_portion'] = trajectory_analysis_dataframe[f'number_of_confinement_zones_with_{BTX_NOMENCLATURE}']/trajectory_analysis_dataframe['number_of_confinement_zones']
        trajectory_analysis_dataframe[f'{CHOL_NOMENCLATURE}_overlap_confinement_portion'] = trajectory_analysis_dataframe[f'number_of_confinement_zones_with_{CHOL_NOMENCLATURE}']/trajectory_analysis_dataframe['number_of_confinement_zones']

        trajectory_analysis_dataframe.to_excel(writer, sheet_name=f'trajectory', index=False)
        trajectory_analysis_dataframe.groupby(['file', 'roi']).mean().to_excel(writer, sheet_name=f'trajectory_by_roi', index=False)

        confinement_dataframe, non_confinement_dataframe = get_dataframe_of_portions_analysis_data(filter_query)

        confinement_dataframe['confinement-duration'] = 10**remove_outliers_from_set_of_values_of_column(np.log10(confinement_dataframe['confinement-duration']))
        non_confinement_dataframe['non-confinement-duration'] = 10**remove_outliers_from_set_of_values_of_column(np.log10(non_confinement_dataframe['non-confinement-duration']))

        confinement_dataframe.to_excel(writer, sheet_name=f'confinement', index=False)
        confinement_dataframe.groupby(['file', 'roi']).mean().to_excel(writer, sheet_name=f'confinement_by_roi', index=False)
        non_confinement_dataframe.to_excel(writer, sheet_name=f'non-confinement', index=False)
        non_confinement_dataframe.groupby(['file', 'roi']).mean().to_excel(writer, sheet_name=f'non-confinement_by_roi', index=False)

"""
overlap_non_confinement_portion_info = open('./Results/overlap_non_confinement_portion_info.txt','w')

for combined_dataset in [
    'Cholesterol and btx',
    'CK666-BTX680-CHOL',
    'BTX680-fPEG-CHOL-50-nM',
    'BTX680-fPEG-CHOL-100-nM',
]:
    fractions = []
    for btx_id in Trajectory._get_collection().find({'info.dataset': combined_dataset, 'info.classified_experimental_condition':BTX_NOMENCLATURE, 'info.immobile': False}, {f'id':1}):
        btx_trajectory = Trajectory.objects(id=btx_id['_id'])[0]
        
        if f'{CHOL_NOMENCLATURE}_single_intersections' in btx_trajectory.info:
            try:
                for non_confined_portion in btx_trajectory.sub_trajectories_trajectories_from_confinement_states(v_th=33, use_info=True)[0]:
                    fractions.append(np.sum(non_confined_portion.info[f'{CHOL_NOMENCLATURE}_single_intersections'])/non_confined_portion.length)
            except KeyError:
                pass
    np.savetxt(f'./Results/{combined_dataset}-{BTX_NOMENCLATURE}_non_confinement_overlap_fraction.txt', fractions, fmt='%f')
    overlap_non_confinement_portion_info.write(f"{combined_dataset}-{BTX_NOMENCLATURE} Fraction: {np.mean(fractions)}, S.E.M: {sem(fractions)}s\n")

    fractions = []
    for chol_id in Trajectory._get_collection().find({'info.dataset': combined_dataset, 'info.classified_experimental_condition':CHOL_NOMENCLATURE, 'info.immobile': False}, {f'id':1}):
        chol_trajectory = Trajectory.objects(id=chol_id['_id'])[0]

        if f'{BTX_NOMENCLATURE}_single_intersections' in chol_trajectory.info:
            try:
                for non_confined_portion in chol_trajectory.sub_trajectories_trajectories_from_confinement_states(v_th=33, use_info=True)[0]:
                    fractions.append(np.sum(non_confined_portion.info[f'{BTX_NOMENCLATURE}_single_intersections'])/non_confined_portion.length)
            except KeyError:
                pass
    np.savetxt(f'./Results/{combined_dataset}-{CHOL_NOMENCLATURE}_non_confinement_overlap_fraction.txt', fractions, fmt='%f')
    overlap_non_confinement_portion_info.write(f"{combined_dataset}-{CHOL_NOMENCLATURE} Fraction: {np.mean(fractions)}, S.E.M: {sem(fractions)}s\n")

overlap_non_confinement_portion_info.close()
"""

DatabaseHandler.disconnect()
