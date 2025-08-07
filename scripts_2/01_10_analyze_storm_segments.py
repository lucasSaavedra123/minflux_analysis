from matplotlib import pyplot as plt
import tqdm 

from CONSTANTS import IP_ADDRESS, COLLECTION_NAME_STORM
from DatabaseHandler import DatabaseHandler
from Trajectory import Trajectory
import numpy as np

DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME_STORM)

lengths = []

for traj in tqdm.tqdm(Trajectory.objects()):
    if 'analysis' not in traj.info or 'clustered_state' not in traj.info['analysis']:
        continue
    portions = traj.sub_trajectories_trajectories_from_confinement_states(custom_states=traj.info['analysis']['clustered_state'])

    for sub_trajs in portions.values():
        for sub_traj in sub_trajs:
            lengths.append(sub_traj.duration)

print(np.quantile(lengths, 0.5))

#It returned 79ms

DatabaseHandler.disconnect()
