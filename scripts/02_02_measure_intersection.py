"""
Intersection = Overlap

Whenever a BTX trajectory is near 
(distance between points is less than 30nm)
from a Chol trajectory, we consider that 
there is an overlap
"""

from collections import defaultdict

import tqdm
import numpy as np

from DatabaseHandler import DatabaseHandler
from Trajectory import Trajectory
from CONSTANTS import *
from utils import both_trajectories_intersect


DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)

btx_and_chol_files = Trajectory.objects(info__classified_experimental_condition__exists=True).distinct(field='info.file')

for file in tqdm.tqdm(btx_and_chol_files):
    trajectories = Trajectory.objects(info__file=file)
    trajectories_by_condition = defaultdict(lambda: [])

    for trajectory in trajectories:
        trajectories_by_condition[trajectory.info['classified_experimental_condition']].append(trajectory)

    for btx_trajectory in tqdm.tqdm(trajectories_by_condition[BTX_NOMENCLATURE]):
        btx_trajectory.info[f'{CHOL_NOMENCLATURE}_single_intersections'] = np.zeros(btx_trajectory.length).tolist()
        btx_trajectory.info[f'{CHOL_NOMENCLATURE}_intersections'] = []
        for chol_trajectory in trajectories_by_condition[CHOL_NOMENCLATURE]:
            overlap, intersections = both_trajectories_intersect(btx_trajectory, chol_trajectory, via='kd-tree', radius_threshold=0.03, return_kd_tree_intersections=True)

            if overlap:
                btx_trajectory.info[f'{CHOL_NOMENCLATURE}_intersections'].append(chol_trajectory.id)

                for index, intersection in enumerate(intersections):
                    if len(intersection) != 0:
                        btx_trajectory.info[f'{CHOL_NOMENCLATURE}_single_intersections'][index] = 1

        btx_trajectory.save()

    for chol_trajectory in tqdm.tqdm(trajectories_by_condition[CHOL_NOMENCLATURE]):
        chol_trajectory.info[f'{BTX_NOMENCLATURE}_single_intersections'] = np.zeros(chol_trajectory.length).tolist()
        chol_trajectory.info[f'{BTX_NOMENCLATURE}_intersections'] = []
        for btx_trajectory in trajectories_by_condition[BTX_NOMENCLATURE]:
            overlap, intersections = both_trajectories_intersect(chol_trajectory, btx_trajectory, via='kd-tree', radius_threshold=0.03, return_kd_tree_intersections=True)
            
            if overlap:
                chol_trajectory.info[f'{BTX_NOMENCLATURE}_intersections'].append(btx_trajectory.id)

                for index, intersection in enumerate(intersections):
                    if len(intersection) != 0:
                        chol_trajectory.info[f'{BTX_NOMENCLATURE}_single_intersections'][index] = 1

        chol_trajectory.save()


DatabaseHandler.disconnect()

#Plotting
"""
if len(intersection) != 0:
    plt.plot(btx_trajectory.get_noisy_x(), btx_trajectory.get_noisy_y(), color='blue')

    for chol_trajectory in intersection:
        plt.plot(chol_trajectory.get_noisy_x(), chol_trajectory.get_noisy_y(), color='orange')

    plt.show()
"""