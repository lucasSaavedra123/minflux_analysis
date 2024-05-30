"""
Upon intersection measure, we measure which
BTX confinement zones overlap with those
of Chol confinement.
"""
import itertools

import tqdm
import ray
import numpy as np
import matplotlib.pyplot as plt
from DatabaseHandler import DatabaseHandler
from Trajectory import Trajectory
from CONSTANTS import *
from utils import measure_overlap, extract_dataset_file_roi_file, get_elliptical_information_of_data_points, ellipse_polyline


CHOL_AND_BTX_DATASETS = [
    'Cholesterol and btx',
    'CK666-BTX680-CHOL',
    'BTX680-fPEG-CHOL-50-nM',
    'BTX680-fPEG-CHOL-100-nM',
]

file_and_rois = [info for info in extract_dataset_file_roi_file() if info[0] in CHOL_AND_BTX_DATASETS]


for dataset, file, roi in file_and_rois:
    DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)
    trajectories = list(Trajectory.objects(info__file=file, info__dataset=dataset, info__roi=roi))
    DatabaseHandler.disconnect()

    trajectories_by_label = {
        CHOL_NOMENCLATURE: [],
        BTX_NOMENCLATURE: []
    }

    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')

    for trajectory in trajectories:
        if 'analysis' in trajectory.info:
            trajectories_by_label[trajectory.info['classified_experimental_condition']].append(trajectory)
            min_x, max_x = min(min_x, np.min(trajectory.get_noisy_x())), max(max_x, np.max(trajectory.get_noisy_x()))
            min_y, max_y = min(min_y, np.min(trajectory.get_noisy_y())), max(max_y, np.max(trajectory.get_noisy_y()))

    real_value = np.mean([t.info[f'number_of_confinement_zones_with_{CHOL_NOMENCLATURE}']/t.info['number_of_confinement_zones'] for t in trajectories_by_label[BTX_NOMENCLATURE] if t.info['number_of_confinement_zones'] != 0])

    chol_confinement_to_chol_trajectory = {}
    chol_confinements = []
    for trajectory in trajectories:
        if 'analysis' in trajectory.info and trajectory.info['classified_experimental_condition'] == CHOL_NOMENCLATURE:
            new_chol_confinements = trajectory.sub_trajectories_trajectories_from_confinement_states(v_th=33, transition_fix_threshold=5, use_info=True)[1]

            for c in new_chol_confinements:
                chol_confinement_to_chol_trajectory[c] = trajectory

            chol_confinements += new_chol_confinements

    simulated_values = []
    print("Real Value:", real_value)
    for _ in tqdm.tqdm(range(99)):
        for t in trajectories:
            t.random_sample([min_x, max_x], [min_y, max_y], in_place=True)

        measure_overlap(trajectories_by_label, chol_confinement_to_chol_trajectory, chol_confinements)

        simulated_values.append(np.mean([t.info[f'number_of_confinement_zones_with_{CHOL_NOMENCLATURE}']/t.info['number_of_confinement_zones'] for t in trajectories_by_label[BTX_NOMENCLATURE] if t.info['number_of_confinement_zones'] != 0]))

        np.savetxt('simulated_values.txt', simulated_values)
    exit()
