import tqdm

from DatabaseHandler import DatabaseHandler
from Trajectory import Trajectory
from CONSTANTS import *


DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)

btx_trajectories = Trajectory.objects(info__classified_experimental_condition=BTX_NOMENCLATURE)

for btx_trajectory in tqdm.tqdm(btx_trajectories):
    chol_trajectories = Trajectory.objects(id__in=btx_trajectory.info[f'{CHOL_NOMENCLATURE}_intersections'])
    print(chol_trajectories)

DatabaseHandler.disconnect()
