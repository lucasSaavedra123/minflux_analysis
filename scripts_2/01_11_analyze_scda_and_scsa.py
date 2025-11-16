"""
ALL trajectories are analyzed.
"""

import ray
import tqdm

from DatabaseHandler import DatabaseHandler
from Trajectory import Trajectory
from CONSTANTS import *

ray.init()

@ray.remote
def analyze_trajectory(trajectory_id):
    DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)

    trajectories = Trajectory.objects(id=trajectory_id)
    assert len(trajectories) == 1
    trajectory = trajectories[0]

    if 'analysis' not in trajectory.info:
        return None

    for state in ['directed','normal','subdifussive','confinement']:
        if f'{state}-states-deepspt' not in trajectory.info['analysis']:
            continue

        trajectory.info['analysis'][f'{state}-d-parallel'] = []
        trajectory.info['analysis'][f'{state}-d-perpendicular'] = []
        trajectory.info['analysis'][f'{state}-d-i'] = []

        states = trajectory.info['analysis'][f'{state}-states-deepspt']

        sub_trajectories_by_state = trajectory.sub_trajectories_trajectories_from_confinement_states(custom_states=states)
        for sub_trajectory in sub_trajectories_by_state[1]:
            d_parallel, d_perpendicular = sub_trajectory.scda()
            d_i = sub_trajectory.scsa()

            trajectory.info['analysis'][f'{state}-d-parallel'].append(d_parallel)
            trajectory.info['analysis'][f'{state}-d-perpendicular'].append(d_perpendicular)
            trajectory.info['analysis'][f'{state}-d-i'].append(d_i)

    trajectory.save()

    DatabaseHandler.disconnect()

DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)
uploaded_trajectories_ids = [str(trajectory_result['_id']) for trajectory_result in Trajectory._get_collection().find({}, {'_id':1})]
DatabaseHandler.disconnect()

for id_batch in tqdm.tqdm(list(batch(uploaded_trajectories_ids, n=1000))):
    ray.get([analyze_trajectory.remote(an_id) for an_id in id_batch])
