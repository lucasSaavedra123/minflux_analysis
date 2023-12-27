"""
ALL trajectories are analyzed.
"""

import tqdm

from DatabaseHandler import DatabaseHandler
from Trajectory import Trajectory
from CONSTANTS import *


DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)

uploaded_trajectories_ids = [str(trajectory_result['_id']) for trajectory_result in Trajectory._get_collection().find({}, {'_id':1})]

for trajectory_id in tqdm.tqdm(uploaded_trajectories_ids):
    trajectories = Trajectory.objects(id=trajectory_id)
    assert len(trajectories) == 1
    trajectory = trajectories[0]

    if 'analysis' in trajectory.info:
        trajectory.info['analysis']['inverse_residence_time'] = trajectory.duration - trajectory.info['analysis']['residence_time']
        trajectory.save()

DatabaseHandler.disconnect()
