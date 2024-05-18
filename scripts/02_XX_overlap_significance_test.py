"""
Upon intersection measure, we measure which
BTX confinement zones overlap with those
of Chol confinement.
"""
import itertools

import tqdm
import ray
import numpy as np

from DatabaseHandler import DatabaseHandler
from Trajectory import Trajectory
from CONSTANTS import *
from utils import both_trajectories_intersect, extract_dataset_file_roi_file


CHOL_AND_BTX_DATASETS = [
    'Cholesterol and btx',
    'CK666-BTX680-CHOL',
    'BTX680-fPEG-CHOL-50-nM',
    'BTX680-fPEG-CHOL-100-nM',
]

file_and_rois = [info for info in extract_dataset_file_roi_file() if info[0] in CHOL_AND_BTX_DATASETS]

DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)

for dataset, file, roi in file_and_rois:
    trajectories = Trajectory.objects(info__file=file, info__dataset=dataset, info__roi=roi)

    trajectories_by_label = {
        CHOL_NOMENCLATURE: [],
        BTX_NOMENCLATURE: []
    }

    for trajectory in trajectories:
        if 'analysis' in trajectory.info:
            trajectories_by_label[trajectory.info['classified_experimental_condition']].append(trajectory)

    









DatabaseHandler.disconnect()
