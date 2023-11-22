import itertools

import tqdm

from DatabaseHandler import DatabaseHandler
from Trajectory import Trajectory
from CONSTANTS import *
from utils import both_trajectories_intersect


DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)

btx_trajectories_ids = Trajectory._get_collection().find({'info.classified_experimental_condition':BTX_NOMENCLATURE}, {f'id':1})

for btx_trajectory_id in tqdm.tqdm(list(btx_trajectories_ids)):
    btx_trajectory = Trajectory.objects(id=btx_trajectory_id['_id'])[0]
    btx_confinements = btx_trajectory.sub_trajectories_trajectories_from_confinement_states(v_th=33)[1]

    chol_trajectories = Trajectory.objects(id__in=btx_trajectory.info[f'{CHOL_NOMENCLATURE}_intersections'])
    chol_confinements = [chol_trajectory.sub_trajectories_trajectories_from_confinement_states(v_th=33)[1] for chol_trajectory in chol_trajectories]
    chol_confinements = list(itertools.chain.from_iterable(chol_confinements))

    btx_trajectory.info['number_of_confinement_zones'] = len(btx_confinements)
    btx_trajectory.info[f'number_of_confinement_zones_with_{CHOL_NOMENCLATURE}'] = 0

    for btx_confinement in btx_confinements:
        there_is_overlap = any([both_trajectories_intersect(chol_confinement, btx_confinement, radius_threshold=0.001) for chol_confinement in chol_confinements])
        btx_trajectory.info[f'number_of_confinement_zones_with_{CHOL_NOMENCLATURE}'] += 1 if there_is_overlap else 0

    assert btx_trajectory.info[f'number_of_confinement_zones_with_{CHOL_NOMENCLATURE}'] <= btx_trajectory.info['number_of_confinement_zones']

chol_trajectories_ids = Trajectory._get_collection().find({'info.classified_experimental_condition':CHOL_NOMENCLATURE}, {f'id':1})

for chol_trajectory_id in tqdm.tqdm(list(chol_trajectories_ids)):
    chol_trajectory = Trajectory.objects(id=chol_trajectory_id['_id'])[0]
    chol_confinements = chol_trajectory.sub_trajectories_trajectories_from_confinement_states(v_th=33)[1]

    btx_trajectories = Trajectory.objects(id__in=chol_trajectory.info[f'{BTX_NOMENCLATURE}_intersections'])
    btx_confinements = [btx_trajectory.sub_trajectories_trajectories_from_confinement_states(v_th=33)[1] for btx_trajectory in btx_trajectories]
    btx_confinements = list(itertools.chain.from_iterable(btx_confinements))

    chol_trajectory.info['number_of_confinement_zones'] = len(chol_confinements)
    chol_trajectory.info[f'number_of_confinement_zones_with_{BTX_NOMENCLATURE}'] = 0

    for chol_confinement in chol_confinements:
        there_is_overlap = any([both_trajectories_intersect(chol_confinement, btx_confinement, radius_threshold=0.001) for btx_confinement in btx_confinements])
        chol_trajectory.info[f'number_of_confinement_zones_with_{BTX_NOMENCLATURE}'] += 1 if there_is_overlap else 0

    assert chol_trajectory.info[f'number_of_confinement_zones_with_{BTX_NOMENCLATURE}'] <= chol_trajectory.info['number_of_confinement_zones']

DatabaseHandler.disconnect()
