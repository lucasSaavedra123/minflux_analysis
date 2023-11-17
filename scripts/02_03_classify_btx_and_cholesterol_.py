import tqdm
import numpy as np

from DatabaseHandler import DatabaseHandler
from Trajectory import Trajectory
from CONSTANTS import *


DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)

trajectory_ids = [str(trajectory_result['_id']) for trajectory_result in Trajectory._get_collection().find({'info.dataset': 'Cholesterol and btx'}, {'_id':1})]

for trajectory_id in tqdm.tqdm(trajectory_ids):
    trajectories = Trajectory.objects(id=trajectory_id)
    trajectory = trajectories[0]
    trajectory_tdcr = np.mean(trajectory.info['dcr'])
    trajectory.info['classified_experimental_condition'] = 'fPEG-Chol' if trajectory_tdcr > 0.55 else 'BTX680R'
    trajectory.save()

DatabaseHandler.disconnect()
