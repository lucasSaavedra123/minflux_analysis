"""
XXXX
"""
import numpy as np
import tqdm
import itertools

from CONSTANTS import *
from DatabaseHandler import DatabaseHandler
from Trajectory import Trajectory
from utils import get_list_of_positions


DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)

values = []

for t in Trajectory.objects():
    print(t)
    new_array = np.vstack((t.get_noisy_x(), t.get_noisy_y()))
    if new_array.shape[1] > 1:
        values.append(new_array)

list_of_trajectories_displacements = values#get_list_of_positions({})
list_of_trajectories_displacements = [np.power(np.diff(X), 2) for X in list_of_trajectories_displacements]
list_of_trajectories_displacements = [np.sqrt(X[0,:] + X[1,:]) for X in list_of_trajectories_displacements]
intervals = list(itertools.chain.from_iterable(list_of_trajectories_displacements))
intervals = [interval for interval in intervals if interval != 0]

print("Localization precision:", np.std(intervals)/np.sqrt(2))

DatabaseHandler.disconnect()
