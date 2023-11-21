import tqdm

from DatabaseHandler import DatabaseHandler
from Trajectory import Trajectory
from CONSTANTS import *


DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)

#btx_trajectories = Trajectory.objects(info__classified_experimental_condition=BTX_NOMENCLATURE)
btx_trajectories_ids = Trajectory._get_collection().find({'info.classified_experimental_condition':BTX_NOMENCLATURE}, {f'id':1})

for btx_trajectory_id in tqdm.tqdm(btx_trajectories_ids):
    btx_trajectory = Trajectory.objects(id=btx_trajectory_id['_id'])[0]
    chol_trajectories = Trajectory.objects(id__in=btx_trajectory.info[f'{CHOL_NOMENCLATURE}_intersections'])
    
    


DatabaseHandler.disconnect()
