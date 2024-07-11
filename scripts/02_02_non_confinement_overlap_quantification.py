"""
Intersection = Overlap

Whenever a BTX trajectory is near 
(distance between points is less than 30nm)
from a Chol trajectory, we consider that 
there is an overlap
"""

from collections import defaultdict

import tqdm
import numpy as np
import ray

from DatabaseHandler import DatabaseHandler
from Trajectory import Trajectory
from CONSTANTS import *
from utils import both_trajectories_intersect, extract_dataset_file_roi_file

RADIUS_THRESHOLD = 0.01

CHOL_AND_BTX_DATASETS = [
    'Cholesterol and btx',
    'CK666-BTX680-CHOL',
    'BTX680-fPEG-CHOL-50-nM',
    'BTX680-fPEG-CHOL-100-nM',
]


ray.init()
@ray.remote
def analyze_dataset(dataset,file,roi):
    DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)
    trajectories = Trajectory.objects(info__dataset=dataset, info__file=file, info__roi=roi)
    trajectories_by_condition = defaultdict(lambda: [])
    non_confined_portions_by_condition = defaultdict(lambda: [])

    for trajectory in trajectories:
        try:
            non_confined_portions_by_condition[trajectory.info['classified_experimental_condition']] += trajectory.sub_trajectories_trajectories_from_confinement_states(v_th=33, transition_fix_threshold=3, use_info=True)[0]
        except KeyError:
            continue
        trajectories_by_condition[trajectory.info['classified_experimental_condition']].append(trajectory)

    for btx_trajectory in trajectories_by_condition[BTX_NOMENCLATURE]:
        try:
            non_confined_portions = btx_trajectory.sub_trajectories_trajectories_from_confinement_states(v_th=33, transition_fix_threshold=3, use_info=True)[0]
        except KeyError:
            continue
        btx_trajectory.info[f'{CHOL_NOMENCLATURE}_intersections'] = 0
        btx_trajectory.info[f'number_of_non_confined_portions'] = len(non_confined_portions)
        for non_btx_confinement_portion in non_confined_portions:
            for non_chol_confinement_portion in non_confined_portions_by_condition[CHOL_NOMENCLATURE]:
                overlap, intersections = both_trajectories_intersect(non_btx_confinement_portion, non_chol_confinement_portion, via='kd-tree', radius_threshold=RADIUS_THRESHOLD, return_kd_tree_intersections=True)
                if overlap:
                    btx_trajectory.info[f'{CHOL_NOMENCLATURE}_intersections'] += 1
                    break

        btx_trajectory.save()

    for chol_trajectory in trajectories_by_condition[CHOL_NOMENCLATURE]:
        try:
            non_confined_portions = chol_trajectory.sub_trajectories_trajectories_from_confinement_states(v_th=33, transition_fix_threshold=3, use_info=True)[0]
        except KeyError:
            continue
        chol_trajectory.info[f'{BTX_NOMENCLATURE}_intersections'] = 0
        chol_trajectory.info[f'number_of_non_confined_portions'] = len(non_confined_portions)
        for non_chol_confinement_portion in non_confined_portions:
            for non_btx_confinement_portion in non_confined_portions_by_condition[BTX_NOMENCLATURE]:
                overlap, intersections = both_trajectories_intersect(non_chol_confinement_portion, non_btx_confinement_portion, via='kd-tree', radius_threshold=RADIUS_THRESHOLD, return_kd_tree_intersections=True)
                if overlap:
                    chol_trajectory.info[f'{BTX_NOMENCLATURE}_intersections'] += 1
                    break

        chol_trajectory.save()

    DatabaseHandler.disconnect()

btx_and_chol_files = [info for info in extract_dataset_file_roi_file() if info[0] in CHOL_AND_BTX_DATASETS]
ray.get([analyze_dataset.remote(dataset,file,roi) for dataset, file, roi in btx_and_chol_files])
