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

    if trajectory.length == 1:
        return

    trajectory.info['analysis']['angles_by_state'] = {
        '0': {
            'label': 'non-confinement',
            'angles': default_angles()
        },
        '1': {
            'label': 'confinement',
            'angles': default_angles()
        },
    }

    sub_trajectories_by_state = trajectory.sub_trajectories_trajectories_from_confinement_states(v_th=33, use_info=True)
    for state in sub_trajectories_by_state:
        for sub_trajectory in sub_trajectories_by_state[state]:
            for angle in trajectory.info['analysis']['angles_by_state'][str(state)]['angles']:
                trajectory.info['analysis']['angles_by_state'][str(state)]['angles'][angle] += sub_trajectory.turning_angles(steps_lag=int(angle))

    trajectory.save()

    DatabaseHandler.disconnect()

DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)
uploaded_trajectories_ids = [str(trajectory_result['_id']) for trajectory_result in Trajectory._get_collection().find({}, {'_id':1})]
DatabaseHandler.disconnect()

for id_batch in tqdm.tqdm(list(batch(uploaded_trajectories_ids, n=1000))):
    ray.get([analyze_trajectory.remote(an_id) for an_id in id_batch])
