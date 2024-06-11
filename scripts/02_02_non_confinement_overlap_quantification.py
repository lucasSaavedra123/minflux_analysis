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

from DatabaseHandler import DatabaseHandler
from Trajectory import Trajectory
from CONSTANTS import *
from utils import both_trajectories_intersect, extract_dataset_file_roi_file

RADIUS_THRESHOLD = 0.01

DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)

CHOL_AND_BTX_DATASETS = [
    'Cholesterol and btx',
    'CK666-BTX680-CHOL',
    'BTX680-fPEG-CHOL-50-nM',
    'BTX680-fPEG-CHOL-100-nM',
]

btx_and_chol_files = list(set([info[1] for info in extract_dataset_file_roi_file() if info[0] in CHOL_AND_BTX_DATASETS]))

for file in tqdm.tqdm(btx_and_chol_files):
    trajectories = Trajectory.objects(info__file=file)
    trajectories_by_condition = defaultdict(lambda: [])
    non_confined_portions_by_condition = defaultdict(lambda: [])

    for trajectory in trajectories:
        try:
            non_confined_portions_by_condition[trajectory.info['classified_experimental_condition']] += trajectory.sub_trajectories_trajectories_from_confinement_states(v_th=33, transition_fix_threshold=3, use_info=True)[0]
        except KeyError:
            continue
        trajectories_by_condition[trajectory.info['classified_experimental_condition']].append(trajectory)

    for btx_trajectory in trajectories_by_condition[BTX_NOMENCLATURE]:
        btx_trajectory.info[f'non_confinement_portions_lengths'] = []
        btx_trajectory.info[f'{CHOL_NOMENCLATURE}_intersections'] = []
        for non_btx_confinement_portion in btx_trajectory.sub_trajectories_trajectories_from_confinement_states(v_th=33, transition_fix_threshold=3, use_info=True)[0]:
            btx_trajectory.info[f'non_confinement_portions_lengths'].append(non_btx_confinement_portion.length)
            btx_trajectory.info[f'{CHOL_NOMENCLATURE}_intersections'].append(0)
            for non_chol_confinement_portion in non_confined_portions_by_condition[CHOL_NOMENCLATURE]:
                overlap, intersections = both_trajectories_intersect(non_btx_confinement_portion, non_chol_confinement_portion, via='kd-tree', radius_threshold=RADIUS_THRESHOLD, return_kd_tree_intersections=True)
                for intersection in intersections:
                    btx_trajectory.info[f'{CHOL_NOMENCLATURE}_intersections'][-1] += 1 if len(intersection) > 0 else 0
        btx_trajectory.save()

    for chol_trajectory in trajectories_by_condition[CHOL_NOMENCLATURE]:
        chol_trajectory.info[f'non_confinement_portions_lengths'] = []
        chol_trajectory.info[f'{BTX_NOMENCLATURE}_intersections'] = []
        for non_chol_confinement_portion in chol_trajectory.sub_trajectories_trajectories_from_confinement_states(v_th=33, transition_fix_threshold=3, use_info=True)[0]:
            chol_trajectory.info[f'non_confinement_portions_lengths'].append(non_btx_confinement_portion.length)
            chol_trajectory.info[f'{BTX_NOMENCLATURE}_intersections'].append(0)
            for non_btx_confinement_portion in non_confined_portions_by_condition[BTX_NOMENCLATURE]:
                overlap, intersections = both_trajectories_intersect(non_chol_confinement_portion, non_btx_confinement_portion, via='kd-tree', radius_threshold=RADIUS_THRESHOLD, return_kd_tree_intersections=True)
                for intersection in intersections:
                    chol_trajectory.info[f'{BTX_NOMENCLATURE}_intersections'][-1] += 1 if len(intersection) > 0 else 0
        chol_trajectory.save()

DatabaseHandler.disconnect()
