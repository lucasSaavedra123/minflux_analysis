from collections import defaultdict

import tqdm
import numpy as np

from DatabaseHandler import DatabaseHandler
from Trajectory import Trajectory
from CONSTANTS import *
from utils import both_trajectories_intersect


DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)
number_of_intersections = 0
files = Trajectory.objects(info__dataset='Cholesterol and btx').distinct(field='info.file')

for file in tqdm.tqdm(files):
    trajectories = Trajectory.objects(info__file=file)
    trajectories_by_condition = defaultdict(lambda: [])

    for trajectory in trajectories:
        trajectories_by_condition[trajectory.info['classified_experimental_condition']].append(trajectory)

    for btx_trajectory in tqdm.tqdm(trajectories_by_condition[BTX_NOMENCLATURE]):
        btx_trajectory.info[f'{CHOL_NOMENCLATURE}_intersections'] = []
        for chol_trajectory in trajectories_by_condition[CHOL_NOMENCLATURE]:
            if both_trajectories_intersect(btx_trajectory, chol_trajectory, radius_threshold=0.001):
                btx_trajectory.info[f'{CHOL_NOMENCLATURE}_intersections'].append(chol_trajectory.id)
        
        btx_trajectory.save()

        """
        if len(intersection) != 0:
            plt.plot(btx_trajectory.get_noisy_x(), btx_trajectory.get_noisy_y(), color='blue')

            for chol_trajectory in intersection:
                plt.plot(chol_trajectory.get_noisy_x(), chol_trajectory.get_noisy_y(), color='orange')
        
            plt.show()
        """

DatabaseHandler.disconnect()
