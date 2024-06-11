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

def t_is_inside_hull(t, hull_path):
    for point in  zip(t.get_noisy_x(), t.get_noisy_y()):
        if hull_path.contains_point((point[0],point[1])):
            return True
    return False

@ray.remote
def analyze(dataset, file_id, roi):
    DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)

    trajectories = list(Trajectory.objects(info__dataset=dataset, info__roi=roi, info__file=file_id))

    for trajectory in trajectories:
        if 'analysis' not in trajectory.info or 'number_of_trajectories_per_overlap' in trajectory.info['analysis']:
            continue
        other_trajectories = [t for t in trajectories if t != trajectory]
        trajectory.info['analysis']['number_of_trajectories_per_overlap'] = []
        trajectory.info['analysis']['number_of_btx_trajectories_per_overlap'] = []
        trajectory.info['analysis']['number_of_chol_trajectories_per_overlap'] = []

        for confined_portion in trajectory.sub_trajectories_trajectories_from_confinement_states(v_th=33, use_info=True)[1]:
            try:
                counter = 0
                counter_btx = 0
                counter_chol = 0
                points = np.zeros((confined_portion.length, 2))
                points[:,0] = confined_portion.get_noisy_x()
                points[:,1] = confined_portion.get_noisy_y()

                hull_path = mplPath.Path(points[ConvexHull(points).vertices])

                for other_trajectory in other_trajectories:
                    if t_is_inside_hull(other_trajectory, hull_path):
                        counter += 1
                        if 'classified_experimental_condition' in other_trajectory.info:
                            if other_trajectory.info['classified_experimental_condition'] == CHOL_NOMENCLATURE:
                                counter_chol += 1
                            else:
                                counter_btx += 1

                trajectory.info['analysis']['number_of_trajectories_per_overlap'].append(counter)
                trajectory.info['analysis']['number_of_btx_trajectories_per_overlap'].append(counter_btx)
                trajectory.info['analysis']['number_of_chol_trajectories_per_overlap'].append(counter_chol)

            except QhullError:
                pass
    
        trajectory.save()

    DatabaseHandler.disconnect()

ray.init()
ray.get([analyze.remote(p[0],p[1], p[2]) for p in extract_dataset_file_roi_file()])
