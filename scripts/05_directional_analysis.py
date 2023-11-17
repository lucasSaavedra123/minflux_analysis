import itertools

import ray
import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import sem


from DatabaseHandler import DatabaseHandler
from Trajectory import Trajectory
from CONSTANTS import *


ray.init()

@ray.remote
def analyze_trajectory(trajectory_id):
    DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)
    trajectories = Trajectory.objects(id=trajectory_id)
    assert len(trajectories) == 1
    trajectory = trajectories[0]
    DatabaseHandler.disconnect()

    window_size = 11
    
    trajectory_time = trajectory.get_time()
    
    result = {
        'negative correlation': [],
        'positive correlation': [],
        'no correlation': []
    }

    try:
        segments_mean, break_points = trajectory.info['directional_coefficient_segments_mean'], trajectory.info['directional_coefficient_break_points']
    except KeyError:
        return result

    segment_durations = []
    index = 0

    for break_point in break_points:
        window_time_interval = trajectory_time[index+(window_size//2):break_point-(window_size//2)]
        if len(window_time_interval) > 1:
            segment_durations.append(window_time_interval[-1]-window_time_interval[0])
        else:
            segment_durations.append(None)
        index = break_point

    for segment_index, _ in enumerate(segments_mean):
        if segment_durations[segment_index] is not None:
            if segments_mean[segment_index] < -0.2:
                result['negative correlation'].append(segment_durations[segment_index])
            elif -0.2 <= segments_mean[segment_index] <= 0.2:
                result['no correlation'].append(segment_durations[segment_index])
            elif 0.2 < segments_mean[segment_index]:
                result['positive correlation'].append(segment_durations[segment_index])
            else:
                raise ValueError('Code Updated')

    result['negative correlation'] = float(np.sum(result['negative correlation']))
    result['positive correlation'] = float(np.sum(result['positive correlation']))
    result['no correlation'] = float(np.sum(result['no correlation']))

    return result

DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)
directional_correlation_segmentation_info_file = open('./Results/directional_correlation_segmentation_info.txt','w')

for dataset in DATASETS_LIST:
    uploaded_trajectories_ids = [str(trajectory_result['_id']) for trajectory_result in Trajectory._get_collection().find({'info.dataset': dataset, 'info.immobile': False}, {'_id':1})]

    full_result = {
        'negative correlation': [],
        'positive correlation': [],
        'no correlation': []
    }

    results = []

    for id_batch in tqdm.tqdm(batch(uploaded_trajectories_ids, n=1000)):
        results += ray.get([analyze_trajectory.remote(an_id) for an_id in id_batch])

        for result in results:
            for label in result:
                if result[label] != 0 and result[label] != []:
                    full_result[label].append(result[label])

    for label in full_result:
        directional_correlation_segmentation_info_file.write(f"{dataset},{label} -> Residence Time: {np.mean(full_result[label])}s, S.E.M: {sem(full_result[label])}s\n")

directional_correlation_segmentation_info_file.close()
DatabaseHandler.disconnect()
