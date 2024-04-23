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

    trajectory.info['analysis']['meanDP'] = trajectory.mean_turning_angle()
    trajectory.info['analysis']['corrDP'] = trajectory.correlated_turning_angle()
    trajectory.info['analysis']['AvgSignD'] = trajectory.directional_persistance()

    trajectory.save()

    DatabaseHandler.disconnect()

DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)
uploaded_trajectories_ids = [str(trajectory_result['_id']) for trajectory_result in Trajectory._get_collection().find({}, {'_id':1})]
DatabaseHandler.disconnect()

for id_batch in tqdm.tqdm(list(batch(uploaded_trajectories_ids, n=1000))):
    ray.get([analyze_trajectory.remote(an_id) for an_id in id_batch])
