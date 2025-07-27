"""
All important results like areas, axis lengths, etc. 
are produced within this file.
"""
import pandas as pd

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

states_dataframe = {'immobile': [], 'not-immobile': []}
states_dataframe_average = {'immobile': [], 'not-immobile': []}


for index, dataset in enumerate(new_datasets_list):
    print(dataset,index)

    basic_query_dict = {'info.dataset': dataset} if index < len(INDIVIDUAL_DATASETS) else {'info.dataset': dataset[0], 'info.classified_experimental_condition':dataset[1]}

    for immobile in [False, True]:
        immobile_str = 'immobile' if immobile else 'not-immobile'
        for state in ['directed','normal','subdifussive','confinement']:
            filter_query = basic_query_dict.copy()
            filter_query.update({'info.immobile': immobile})

            state_dataframe = get_dataframe_of_portions_analysis_data_by_state(filter_query, state)

            state_dataframe[f'{state}-area-deepspt'] *= 1e6
            state_dataframe[f'{state}-a-deepspt'] *= 1e3
            state_dataframe[f'{state}-b-deepspt'] *= 1e3
            state_dataframe[f'{state}-duration-deepspt'] = 10**remove_outliers_from_set_of_values_of_column(np.log10(state_dataframe[f'{state}-duration-deepspt']))
            state_dataframe[f'{state}-a-deepspt'] = 10**remove_outliers_from_set_of_values_of_column(np.log10(state_dataframe[f'{state}-a-deepspt']))
            state_dataframe[f'{state}-b-deepspt'] = 10**remove_outliers_from_set_of_values_of_column(np.log10(state_dataframe[f'{state}-b-deepspt']))
            state_dataframe[f'{state}-area-deepspt'] = 10**remove_outliers_from_set_of_values_of_column(np.log10(state_dataframe[f'{state}-area-deepspt']))
            state_dataframe[f'{state}-steps-deepspt'] = 10**remove_outliers_from_set_of_values_of_column(np.log10(state_dataframe[f'{state}-steps-deepspt']))

            #state_dataframe.to_excel(writer, sheet_name=state, index=False)
            #state_dataframe.groupby(['file', 'roi']).mean().to_excel(writer, sheet_name=f'{state}_by_roi', index=False)
            state_dataframe['dataset'] = str(dataset)
            states_dataframe_average[immobile_str].append(state_dataframe.groupby(['file', 'roi','dataset']).mean().reset_index())
            states_dataframe[immobile_str].append(state_dataframe)

        for dgn_state in [0,1,2,3,4]:
            filter_query = basic_query_dict.copy()
            filter_query.update({'info.immobile': immobile})
            try:
                state_dataframe = get_dataframe_of_portions_analysis_data_by_intensity_state(filter_query, dgn_state)
            except KeyError:
                break
            state_dataframe[f'{dgn_state}-duration-dgn'] = 10**remove_outliers_from_set_of_values_of_column(np.log10(state_dataframe[f'{dgn_state}-duration-dgn']))
            state_dataframe[f'{dgn_state}-steps-dgn'] = 10**remove_outliers_from_set_of_values_of_column(np.log10(state_dataframe[f'{dgn_state}-steps-dgn']))

            state_dataframe['dataset'] = str(dataset)
            states_dataframe_average[immobile_str].append(state_dataframe.groupby(['file', 'roi','dataset']).mean().reset_index())
            states_dataframe[immobile_str].append(state_dataframe)

for immobile in [False, True]:
    immobile_str = 'immobile' if immobile else 'not-immobile'

    df_all = pd.concat(states_dataframe[immobile_str], ignore_index=True)

    for column in df_all.columns:
        with pd.ExcelWriter(f"./Results/{immobile_str}_{column}_gs_deepspt_portions_individual.xlsx") as writer:
            dataset_groups = {
                name: group[column].reset_index(drop=True)
                for name, group in df_all.groupby("dataset")
            }

            df_wide = pd.DataFrame(dataset_groups)
            df_wide = df_wide.dropna(axis=0,how = 'all')
            df_wide = df_wide.apply(lambda col: pd.Series(col.dropna().tolist() + [np.nan] * (len(col) - col.count())), axis=0)
            df_wide[TO_SAVE_DATASET].to_excel(writer, sheet_name=column, index=False)

    df_all = pd.concat(states_dataframe_average[immobile_str], ignore_index=True)

    for column in df_all.columns:
        with pd.ExcelWriter(f"./Results/{immobile_str}_{column}_gs_deepspt_portions_average.xlsx") as writer:
            dataset_groups = {
                name: group[column].reset_index(drop=True)
                for name, group in df_all.groupby("dataset")
            }

            df_wide = pd.DataFrame(dataset_groups)
            df_wide = df_wide.dropna(axis=0,how = 'all')
            df_wide = df_wide.apply(lambda col: pd.Series(col.dropna().tolist() + [np.nan] * (len(col) - col.count())), axis=0)
            df_wide[TO_SAVE_DATASET].to_excel(writer, sheet_name=column, index=False)

DatabaseHandler.disconnect()
