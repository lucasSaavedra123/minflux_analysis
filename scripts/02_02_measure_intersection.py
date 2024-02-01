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

files = Trajectory.objects().distinct(field='info.file')

btx_and_chol_files = [
    '231013-105211_mbm test.txt',
    '231013-105628_mbm test-pow8pc.txt',
    '231013-110430_mbm test-pow8pc.txt',
    '231013-111321_mbm test-pow8pc.txt',
    '231013-111726_mbm test-pow8pc.txt',
    '231013-112242_mbm test-pow8pc.txt',
    '231013-112652_mbm test-pow8pc.txt',
    '231013-113251_mbm test-pow8pc.txt',
    '231013-113638_mbm test-pow8pc.txt',
    '231013-124040_mbm test.txt',
    '231013-124511_mbm test.txt',
    '231013-125044_mbm test.txt',
    '231013-125411_mbm test.txt',
    '231013-125818_mbm test.txt',
    '231013-130259_mbm test.txt',
    '231013-130748_mbm test.txt',
    '231013-131100_mbm test.txt',
    '231013-131615_mbm test.txt',
    '231013-131935_mbm test.txt',
    '231013-132310_mbm test.txt',
    '231013-132703_mbm test.txt',
    '231013-153332_mbm test.txt',
    '231013-153631_mbm test.txt',
    '231013-154043_mbm test.txt',
    '231013-154400_mbm test.txt',
    '231013-154702_mbm test.txt',
    '231013-154913_mbm test.txt',
    '231013-155220_mbm test.txt',
    '231013-155616_mbm test.txt',
    '231013-155959_mbm test.txt',
    '231013-160351_mbm test.txt',
    '231013-160951_mbm test.txt',
    '231013-161302_mbm test.txt',
    '231013-161554_mbm test.txt',
    '231013-162155_mbm test.txt',
    '231013-162602_mbm test.txt',
    '231013-162934_mbm test.txt',
    '231013-163124_mbm test.txt',
    '231013-163414_mbm test.txt',
    '231013-163548_mbm test.txt'
]


for file in tqdm.tqdm(files):
    trajectories = Trajectory.objects(info__file=file)
    trajectories_by_condition = defaultdict(lambda: [])

    if file in btx_and_chol_files:
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
    else:
        trajectories = list(trajectories)
        for trajectory_index, trajectory in tqdm.tqdm(enumerate(trajectories)):
            trajectory.info[f'single_intersections_with_others'] = np.zeros(trajectory.length).tolist()
            trajectory.info[f'intersections_with_others'] = []
            for other_trajectory in trajectories[:trajectory_index] + trajectories[trajectory_index+1:]:
                overlap, intersections = both_trajectories_intersect(trajectory, other_trajectory, via='kd-tree', radius_threshold=0.03, return_kd_tree_intersections=True)

                if overlap:
                    trajectory.info[f'intersections_with_others'].append(other_trajectory.id)

                    for index, intersection in enumerate(intersections):
                        if len(intersection) != 0:
                            trajectory.info[f'single_intersections_with_others'][index] = 1

            trajectory.save()


DatabaseHandler.disconnect()

#Plotting
"""
if len(intersection) != 0:
    plt.plot(btx_trajectory.get_noisy_x(), btx_trajectory.get_noisy_y(), color='blue')

    for chol_trajectory in intersection:
        plt.plot(chol_trajectory.get_noisy_x(), chol_trajectory.get_noisy_y(), color='orange')

    plt.show()
"""