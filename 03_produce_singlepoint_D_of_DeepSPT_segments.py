"""
ALL trajectories are analyzed.
"""
import tqdm
from collections import defaultdict
import numpy as np
import pandas as pd
from DatabaseHandler import DatabaseHandler
from Trajectory import Trajectory
from CONSTANTS import *
from utils import *
import anDiffReg as an


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

    basic_query_dict = {'info.dataset': dataset} if index < len(INDIVIDUAL_DATASETS) else {'info.dataset': dataset[0], 'info.classified_experimental_condition': dataset[1]}

    for immobile in [False, True]:
        immobile_str = 'immobile' if immobile else 'not-immobile'
        print(dataset, index, immobile_str)

        filter_query = basic_query_dict.copy()
        filter_query.update({'info.immobile': immobile})

        uploaded_trajectories_ids = [str(trajectory_result['_id']) for trajectory_result in Trajectory._get_collection().find(filter_query, {'_id': 1})]

        rows_by_state = defaultdict(list)

        for trajectory_id in tqdm.tqdm(uploaded_trajectories_ids):
            trajectory = Trajectory.objects(id=trajectory_id).only(
                'info.analysis.directed-states-deepspt',
                'info.analysis.normal-states-deepspt',
                'info.analysis.subdifussive-states-deepspt',
                'info.analysis.confinement-states-deepspt',
                'info.analysis.diffusion_coefficient_t',
                'info.analysis.hurst_exponent_t',
                'info.roi',
                'info.file',
                'noisy',
                'x',
                'y'
            ).first()

            if trajectory is None:
                continue

            if 'analysis' not in trajectory.info:
                continue

            roi = trajectory.info.get('roi')
            file = trajectory.info.get('file')

            for state in ['directed', 'normal', 'subdifussive', 'confinement']:
                if f'{state}-states-deepspt' not in trajectory.info['analysis']:
                    continue

                states = trajectory.info['analysis'][f'{state}-states-deepspt']

                sub_trajectories_by_state = trajectory.sub_trajectories_trajectories_from_confinement_states(custom_states=states)

                for subtrajectory in sub_trajectories_by_state[1]:
                    if subtrajectory.length >= 30:
                        X = np.array([[s] for s in trajectory.get_noisy_x()])
                        Y = np.array([[s] for s in trajectory.get_noisy_y()])
                        msd = an.tamsd(np.dstack((X,Y)))
                        
                        from IPython import embed
                        embed()

                        rows_by_state[state].append({
                            f'{state}-alpha': alpha,
                            f'{state}-d': d,
                            'roi': roi,
                            'file': file,
                        })

        for state in ['directed', 'normal', 'subdifussive', 'confinement']:
            if len(rows_by_state[state]) == 0:
                continue

            state_dataframe = pd.DataFrame(rows_by_state[state])

            state_dataframe[f'{state}-d'] = 10 ** remove_outliers_from_set_of_values_of_column(state_dataframe[f'{state}-d'])

            state_dataframe['dataset'] = str(dataset)

            states_dataframe_average[immobile_str].append(
                state_dataframe.groupby(['file', 'roi', 'dataset']).mean().reset_index()
            )
            states_dataframe[immobile_str].append(state_dataframe)

for immobile in [False, True]:
    immobile_str = 'immobile' if immobile else 'not-immobile'

    df_all = pd.concat(states_dataframe[immobile_str], ignore_index=True)

    for column in df_all.columns:
        if column in ('roi', 'file', 'dataset'):
            continue
        with pd.ExcelWriter(f"./Results/{immobile_str}_{column}_gs_deepspt_segments_individual.xlsx") as writer:
            dataset_groups = {
                name: group[column].reset_index(drop=True)
                for name, group in df_all.groupby("dataset")
            }

            df_wide = pd.DataFrame(dataset_groups)
            df_wide = df_wide.dropna(axis=0, how='all')
            df_wide = df_wide.apply(lambda col: pd.Series(col.dropna().tolist() + [np.nan] * (len(col) - col.count())), axis=0)
            df_wide[TO_SAVE_DATASET].to_excel(writer, sheet_name=column, index=False)

    df_all = pd.concat(states_dataframe_average[immobile_str], ignore_index=True)

    for column in df_all.columns:
        if column in ('roi', 'file', 'dataset'):
            continue
        with pd.ExcelWriter(f"./Results/{immobile_str}_{column}_gs_deepspt_segments_average.xlsx") as writer:
            dataset_groups = {
                name: group[column].reset_index(drop=True)
                for name, group in df_all.groupby("dataset")
            }

            df_wide = pd.DataFrame(dataset_groups)
            df_wide = df_wide.dropna(axis=0, how='all')
            df_wide = df_wide.apply(lambda col: pd.Series(col.dropna().tolist() + [np.nan] * (len(col) - col.count())), axis=0)
            df_wide[TO_SAVE_DATASET].to_excel(writer, sheet_name=column, index=False)

DatabaseHandler.disconnect()