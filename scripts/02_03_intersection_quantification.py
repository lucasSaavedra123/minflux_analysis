"""
Upon intersection measure, we measure which
BTX confinement zones overlap with those
of Chol confinement.
"""
import itertools

import tqdm
import ray

from DatabaseHandler import DatabaseHandler
from Trajectory import Trajectory
from CONSTANTS import *
from utils import both_trajectories_intersect


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
def analyze_btx_trajectory(btx_trajectory_id):
    DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)

    btx_trajectory = Trajectory.objects(id=btx_trajectory_id['_id'])[0]

    if 'analysis' not in btx_trajectory.info:
        return None

    btx_confinements = btx_trajectory.sub_trajectories_trajectories_from_confinement_states(v_th=33, use_info=False)[1]

    chol_trajectories = Trajectory.objects(id__in=btx_trajectory.info[f'{CHOL_NOMENCLATURE}_intersections'])
    chol_confinements = [chol_trajectory.sub_trajectories_trajectories_from_confinement_states(v_th=33, use_info=False)[1] for chol_trajectory in chol_trajectories]
    chol_confinements = list(itertools.chain.from_iterable(chol_confinements))

    btx_trajectory.info['number_of_confinement_zones'] = len(btx_confinements)
    btx_trajectory.info[f'number_of_confinement_zones_with_{CHOL_NOMENCLATURE}'] = 0

    for btx_confinement in btx_confinements:
        there_is_overlap = any([both_trajectories_intersect(chol_confinement, btx_confinement, via='hull') for chol_confinement in chol_confinements])
        btx_trajectory.info[f'number_of_confinement_zones_with_{CHOL_NOMENCLATURE}'] += 1 if there_is_overlap else 0
    
    assert btx_trajectory.info[f'number_of_confinement_zones_with_{CHOL_NOMENCLATURE}'] <= btx_trajectory.info['number_of_confinement_zones']
    btx_trajectory.save()

    DatabaseHandler.disconnect()

@ray.remote
def analyze_chol_trajectory(chol_trajectory_id):
    DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)

    chol_trajectory = Trajectory.objects(id=chol_trajectory_id['_id'])[0]

    if 'analysis' not in chol_trajectory.info:
        return None

    chol_confinements = chol_trajectory.sub_trajectories_trajectories_from_confinement_states(v_th=33, use_info=False)[1]

    btx_trajectories = Trajectory.objects(id__in=chol_trajectory.info[f'{BTX_NOMENCLATURE}_intersections'])
    btx_confinements = [btx_trajectory.sub_trajectories_trajectories_from_confinement_states(v_th=33, use_info=False)[1] for btx_trajectory in btx_trajectories]
    btx_confinements = list(itertools.chain.from_iterable(btx_confinements))

    chol_trajectory.info['number_of_confinement_zones'] = len(chol_confinements)
    chol_trajectory.info[f'number_of_confinement_zones_with_{BTX_NOMENCLATURE}'] = 0

    for chol_confinement in chol_confinements:
        there_is_overlap = any([both_trajectories_intersect(chol_confinement, btx_confinement, via='hull') for btx_confinement in btx_confinements])
        chol_trajectory.info[f'number_of_confinement_zones_with_{BTX_NOMENCLATURE}'] += 1 if there_is_overlap else 0

    assert chol_trajectory.info[f'number_of_confinement_zones_with_{BTX_NOMENCLATURE}'] <= chol_trajectory.info['number_of_confinement_zones']
    chol_trajectory.save()

    DatabaseHandler.disconnect()

analyzed_ids = []

DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)
btx_trajectories_ids = list(Trajectory._get_collection().find({'info.classified_experimental_condition':BTX_NOMENCLATURE}, {f'id':1}))
DatabaseHandler.disconnect()

for id_batch in tqdm.tqdm(list(batch(btx_trajectories_ids, n=1000))):
    ray.get([analyze_btx_trajectory.remote(an_id) for an_id in id_batch])
analyzed_ids += btx_trajectories_ids

DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)
chol_trajectories_ids = list(Trajectory._get_collection().find({'info.classified_experimental_condition':CHOL_NOMENCLATURE}, {f'id':1}))
DatabaseHandler.disconnect()

for index, id_batch in tqdm.tqdm(list(enumerate(list(batch(chol_trajectories_ids, n=1000))))):
    ray.get([analyze_chol_trajectory.remote(an_id) for an_id in id_batch])
analyzed_ids += chol_trajectories_ids

DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)
all_ids = list(Trajectory._get_collection().find({}, {f'id':1}))
DatabaseHandler.disconnect()

for index, id_batch in tqdm.tqdm(list(enumerate(list(batch(all_ids, n=1000))))):
    new_id_batch = [id_id for id_id in id_batch if id_id not in analyzed_ids]
    ray.get([analyze_trajectory.remote(an_id) for an_id in new_id_batch])
