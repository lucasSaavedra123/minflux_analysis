import os
from collections import defaultdict

import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from DatabaseHandler import DatabaseHandler
from Trajectory import Trajectory
from CONSTANTS import *
from utils import both_trajectories_intersect


"""
DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)

trajectory_ids = [str(trajectory_result['_id']) for trajectory_result in Trajectory._get_collection().find({'info.dataset': 'Cholesterol and btx'}, {'_id':1})]

for trajectory_id in tqdm.tqdm(trajectory_ids):
    trajectories = Trajectory.objects(id=trajectory_id)
    trajectory = trajectories[0]
    both_trajectories_intersect(trajectory, trajectory)

DatabaseHandler.disconnect()
"""

TDCR_THRESHOLD = 0.55

def extract_trajectory_from(track_dataset, dataset_directory, file_name, track_id):
    info_value = {'dataset': dataset_directory, 'file': file_name, 'trajectory_id': track_id}

    if 'dcr' in track_dataset.columns:
        info_value['dcr'] = track_dataset['dcr'].values

    return Trajectory(
        x = track_dataset['x'].values * 1e6,
        y = track_dataset['y'].values * 1e6,
        t = track_dataset['t'].values,
        info=info_value,
        noisy=True
    )

def extract_dataframes_from_file(dataset_directory, a_file):
    print(f"Extracting info from dataset {dataset_directory} and file {a_file}")
    dataset = pd.read_csv(os.path.join(dataset_directory,a_file), sep=' ', header=None)

    if len(dataset.columns) == 5:
        dataset = dataset.rename(columns={index: value for index, value in enumerate(['track_id', 't', 'x', 'y', 'dcr'])})
    elif len(dataset.columns) == 4:
        dataset = dataset.rename(columns={index: value for index, value in enumerate(['track_id', 't', 'x', 'y'])})
    else:
        raise Exception('No valid number of columns')

    current_id = dataset.iloc[0]['track_id']
    initial_row = 0
    row_index = 1
    ids_historial = defaultdict(lambda: 0)

    extraction_result = []

    for row_index in list(range(len(dataset))):
        if dataset.iloc[row_index]['track_id'] != current_id:
            extraction_result.append((
                dataset.iloc[initial_row:row_index].copy().sort_values('t', ascending=True),
                int(current_id),
                int(ids_historial[current_id]),
            ))

            ids_historial[current_id] += 1
            initial_row = row_index
            current_id = dataset.iloc[row_index]['track_id']

    extraction_result.append((
        dataset.iloc[initial_row:row_index].copy().sort_values('t', ascending=True),
        int(current_id),
        int(ids_historial[current_id]),
    ))

    return extraction_result

def get_trajectories_from(dataset_directory, file_index=None):
    trajectories = []

    file_names = [file for file in os.listdir(dataset_directory) if file.endswith('.txt')]

    if file_index is not None:
        file_names = [file_names[file_index]]

    for a_file in file_names:
        extraction_result = extract_dataframes_from_file(dataset_directory, a_file)
        print(f"Reading trajectories from dataset {dataset_directory} and file {a_file}")
        for info_extracted in extraction_result:
            trajectories.append(extract_trajectory_from(
                info_extracted[0],
                dataset_directory,
                a_file,
                f"{info_extracted[1]}_{info_extracted[2]}"
            ))

    return trajectories


number_of_files = len([file for file in os.listdir('Cholesterol and btx') if file.endswith('.txt')])

number_of_intersections = 0

for file_index in tqdm.tqdm(list(range(number_of_files))):
    trajectories = get_trajectories_from('Cholesterol and btx', file_index)
    trajectories_by_condition = defaultdict(lambda: [])

    for trajectory in trajectories:
        trajectories_by_condition['fPEG-Chol' if np.mean(trajectory.info['dcr']) > TDCR_THRESHOLD else 'BTX680R'].append(trajectory)

    for btx_trajectory in trajectories_by_condition['BTX680R']:#tqdm.tqdm(trajectories_by_condition['BTX680R']):
        intersection = []
        for chol_trajectory in trajectories_by_condition['fPEG-Chol']:
            if both_trajectories_intersect(btx_trajectory, chol_trajectory, radius_threshold=0.01):
                intersection.append(chol_trajectory)
        """
        if len(intersection) != 0:
            plt.plot(btx_trajectory.get_noisy_x(), btx_trajectory.get_noisy_y(), color='blue')

            for chol_trajectory in intersection:
                plt.plot(chol_trajectory.get_noisy_x(), chol_trajectory.get_noisy_y(), color='orange')
        
            plt.show()
        """

        number_of_intersections += 1

print(number_of_intersections)