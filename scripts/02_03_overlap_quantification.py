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


@ray.remote
def analyze_trajectory(trajectory_id):
    DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)

    trajectory = Trajectory.objects(id=trajectory_id['_id'])[0]

    if 'analysis' not in trajectory.info:
        return None

    confinements = trajectory.sub_trajectories_trajectories_from_confinement_states(v_th=33, use_info=False)[1]

    other_trajectories = Trajectory.objects(id__in=trajectory.info[f'intersections_with_others'])
    other_confinements = [other_trajectory.sub_trajectories_trajectories_from_confinement_states(v_th=33, use_info=False)[1] for other_trajectory in other_trajectories]
    other_confinements = list(itertools.chain.from_iterable(other_confinements))

    trajectory.info['number_of_confinement_zones'] = len(confinements)
    trajectory.info[f'number_of_confinement_zones_with_others'] = 0

    for btx_confinement in confinements:
        there_is_overlap = any([both_trajectories_intersect(chol_confinement, btx_confinement, via='hull') for chol_confinement in other_confinements])
        trajectory.info[f'number_of_confinement_zones_with_others'] += 1 if there_is_overlap else 0
    
    assert trajectory.info[f'number_of_confinement_zones_with_others'] <= trajectory.info['number_of_confinement_zones']
    trajectory.save()

    DatabaseHandler.disconnect()

@ray.remote
def analyze_dataset_and_roi(dataset, file, roi):
    DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)
    trajectories = Trajectory.objects(info__file=file, info__dataset=dataset, info__roi=roi)

    trajectories_by_label = {
        CHOL_NOMENCLATURE: [],
        BTX_NOMENCLATURE: []
    }

    for trajectory in trajectories:
        if 'analysis' in trajectory.info:
            trajectories_by_label[trajectory.info['classified_experimental_condition']].append(trajectory)

    for chol_trajectory in trajectories_by_label[CHOL_NOMENCLATURE]:
        chol_confinements = chol_trajectory.sub_trajectories_trajectories_from_confinement_states(v_th=33, transition_fix_threshold=5, use_info=True)[1]
        btx_confinements = [btx_trajectory.sub_trajectories_trajectories_from_confinement_states(v_th=33, transition_fix_threshold=5, use_info=True)[1] for btx_trajectory in trajectories_by_label[BTX_NOMENCLATURE]]
        btx_confinements = list(itertools.chain.from_iterable(btx_confinements))

        chol_trajectory.info['number_of_confinement_zones'] = len(chol_confinements)
        chol_trajectory.info[f'number_of_confinement_zones_with_{BTX_NOMENCLATURE}'] = 0

        for chol_confinement in chol_confinements:
            for btx_confinement in btx_confinements:
                print(np.linalg.norm(chol_confinement.centroid-btx_confinement.centroid))
                if np.linalg.norm(chol_confinement.centroid-btx_confinement.centroid) < 1:#um
                    there_is_overlap = both_trajectories_intersect(chol_confinement, btx_confinement, via='hull')
                    chol_trajectory.info[f'number_of_confinement_zones_with_{BTX_NOMENCLATURE}'] += 1 if there_is_overlap else 0
                    break
        assert chol_trajectory.info[f'number_of_confinement_zones_with_{BTX_NOMENCLATURE}'] <= chol_trajectory.info['number_of_confinement_zones']
        chol_trajectory.save()

    for btx_trajectory in trajectories_by_label[BTX_NOMENCLATURE]:
        btx_confinements = btx_trajectory.sub_trajectories_trajectories_from_confinement_states(v_th=33, transition_fix_threshold=5, use_info=True)[1]
        chol_confinements = [chol_trajectory.sub_trajectories_trajectories_from_confinement_states(v_th=33, transition_fix_threshold=5, use_info=True)[1] for chol_trajectory in trajectories_by_label[CHOL_NOMENCLATURE]]
        chol_confinements = list(itertools.chain.from_iterable(chol_confinements))

        btx_trajectory.info['number_of_confinement_zones'] = len(btx_confinements)
        btx_trajectory.info[f'number_of_confinement_zones_with_{CHOL_NOMENCLATURE}'] = 0

        for btx_confinement in btx_confinements:
            for chol_confinement in chol_confinements:
                if np.linalg.norm(chol_confinement.centroid-btx_confinement.centroid) < 1:#um
                    there_is_overlap = both_trajectories_intersect(chol_confinement, btx_confinement, via='hull')
                    btx_trajectory.info[f'number_of_confinement_zones_with_{CHOL_NOMENCLATURE}'] += 1 if there_is_overlap else 0
                    break
        assert btx_trajectory.info[f'number_of_confinement_zones_with_{CHOL_NOMENCLATURE}'] <= btx_trajectory.info['number_of_confinement_zones']
        btx_trajectory.save()

    DatabaseHandler.disconnect()

file_and_rois = [info for info in extract_dataset_file_roi_file() if info[0] in CHOL_AND_BTX_DATASETS]

ray.get([analyze_dataset_and_roi.remote(dataset, file, roi) for dataset,file,roi in file_and_rois])

"""
analyzed_ids += chol_trajectories_ids

DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)
all_ids = list(Trajectory._get_collection().find({}, {f'id':1}))
DatabaseHandler.disconnect()

for index, id_batch in tqdm.tqdm(list(enumerate(list(batch(all_ids, n=1000))))):
    new_id_batch = [id_id for id_id in id_batch if id_id not in analyzed_ids]
    ray.get([analyze_trajectory.remote(an_id) for an_id in new_id_batch])
"""