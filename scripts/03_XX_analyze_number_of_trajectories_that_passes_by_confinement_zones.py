import pandas as pd
import matplotlib.path as mplPath
import os
import tqdm
import numpy as np
from scipy.spatial import ConvexHull, QhullError
from matplotlib.path import Path
from collections import defaultdict

from CONSTANTS import *
from utils import *
import ray
from DatabaseHandler import DatabaseHandler
from Trajectory import Trajectory


DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)

def unique(list1):

    # initialize a null list
    unique_list = []

    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)

    return unique_list

FILE_AND_ROI_FILE_CACHE = 'file_and_roi.txt'

if not os.path.exists(FILE_AND_ROI_FILE_CACHE):
    file_id_and_roi_list = [[r['info']['file'], r['info']['roi']] for r in Trajectory._get_collection().find(
        {},
        {f'id':1, 'info.roi':1, 'info.file':1}
    )]

    file_id_and_roi_list = unique(file_id_and_roi_list)

    a_file = open(FILE_AND_ROI_FILE_CACHE, 'w')
    for file, roi in file_id_and_roi_list:
        a_file.write(f'{file},{roi}\n')
    a_file.close()
else:
    file_id_and_roi_list = []
    a_file = open(FILE_AND_ROI_FILE_CACHE, 'r')
    for line in a_file.readlines():
        line = line.strip()
        line = line.split(',')
        file_id_and_roi_list.append([line[0], int(line[1])])
    a_file.close()

def t_is_inside_hull(t, hull_path):
    for point in  zip(t.get_noisy_x(), t.get_noisy_y()):
        if hull_path.contains_point((point[0],point[1])):
            return True
    return False

@ray.remote
def analyze(file_id, roi):
    DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)

    trajectories = list(Trajectory.objects(info__roi=roi, info__file=file_id))

    for trajectory in trajectories:
        if 'analysis' not in trajectory.info or 'number_of_trajectories_per_overlap' in trajectory.info['analysis']:
            continue
        other_trajectories = [t for t in trajectories if t != trajectory]
        trajectory.info['analysis']['number_of_trajectories_per_overlap'] = []

        for confined_portion in trajectory.sub_trajectories_trajectories_from_confinement_states(v_th=33, use_info=True)[1]:
            try:
                counter = 0
                points = np.zeros((confined_portion.length, 2))
                points[:,0] = confined_portion.get_noisy_x()
                points[:,1] = confined_portion.get_noisy_y()

                hull_path = mplPath.Path(points[ConvexHull(points).vertices])

                for other_trajectory in other_trajectories:
                    if t_is_inside_hull(other_trajectory, hull_path):
                        counter += 1

                trajectory.info['analysis']['number_of_trajectories_per_overlap'].append(counter)

            except QhullError:
                pass
    
        trajectory.save()

    DatabaseHandler.disconnect()

results = ray.get([analyze.remote(p[0], p[1]) for p in file_id_and_roi_list])
