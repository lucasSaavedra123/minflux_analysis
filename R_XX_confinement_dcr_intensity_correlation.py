import os
from collections import defaultdict
import itertools
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tqdm
from scipy.signal import correlate

from Trajectory import Trajectory
from utils import *
from CONSTANTS import *


def get_trajectory_from_track_dataset(track_dataset, dataset_directory, file_name, track_id):
    info_value = {'dataset': dataset_directory, 'file': file_name, 'trajectory_id': track_id}

    info_value['dcr'] = track_dataset['dcr'].values
    info_value['intensity'] = track_dataset['intensity'].values

    return Trajectory(
        x = track_dataset['x'].values * 1e6,
        y = track_dataset['y'].values * 1e6,
        t = track_dataset['t'].values,
        info=info_value,
        noisy=True
    )

def extract_dataframes_from_file(dataset_directory, a_file):
    dataset = pd.read_csv(os.path.join(dataset_directory,a_file), sep=' ', header=None)
    dataset = dataset.rename(columns={index: value for index, value in enumerate(['track_id', 't', 'x', 'y', 'intensity', 'dcr'])})

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

def upload_trajectories_from_file(a_file):
    t = []
    extraction_result = extract_dataframes_from_file('Cholesterol and btx', a_file)
    for info_extracted in extraction_result:
        t.append(get_trajectory_from_track_dataset(
            info_extracted[0],
            'test',
            a_file,
            f"{info_extracted[1]}_{info_extracted[2]}"
        ))
    return t

intensities_on_overlap = []
intensities_without_overlap = []

for file in tqdm.tqdm(list(os.listdir('Cholesterol and btx'))):
    trajectories_from_file = upload_trajectories_from_file(file)

    trajectories_by_condition = defaultdict(lambda: [])
    
    for trajectory in trajectories_from_file:
        trajectories_by_condition[CHOL_NOMENCLATURE if np.mean(trajectory.info['dcr']) > TDCR_THRESHOLD else BTX_NOMENCLATURE].append(trajectory)

    for btx_trajectory in trajectories_by_condition[BTX_NOMENCLATURE]:
        btx_trajectory.info[f'{CHOL_NOMENCLATURE}_intersections'] = []
        for chol_trajectory in trajectories_by_condition[CHOL_NOMENCLATURE]:
            if both_trajectories_intersect(btx_trajectory, chol_trajectory, via='kd-tree', radius_threshold=0.01):
                btx_trajectory.info[f'{CHOL_NOMENCLATURE}_intersections'].append(chol_trajectory)

    for btx_trajectory in trajectories_by_condition[BTX_NOMENCLATURE]:
        btx_confinements = btx_trajectory.sub_trajectories_trajectories_from_confinement_states(v_th=33)[1]

        chol_trajectories = btx_trajectory.info[f'{CHOL_NOMENCLATURE}_intersections']
        chol_confinements = [chol_trajectory.sub_trajectories_trajectories_from_confinement_states(v_th=33)[1] for chol_trajectory in chol_trajectories]
        chol_confinements = list(itertools.chain.from_iterable(chol_confinements))

        for btx_confinement in btx_confinements:
            there_is_overlap = any([both_trajectories_intersect(chol_confinement, btx_confinement, via='hull') for chol_confinement in chol_confinements])

            if there_is_overlap:
                intensities_on_overlap += btx_confinement.info['intensity']
            else:
                intensities_without_overlap += btx_confinement.info['intensity']

    np.savetxt('on_overlap.txt', intensities_on_overlap)
    np.savetxt('without_overlap.txt', intensities_without_overlap)