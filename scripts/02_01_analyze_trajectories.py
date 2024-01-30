"""
ALL trajectories are analyzed.
"""

import ray
import numpy as np
import tqdm
from scipy.spatial import distance_matrix

from DatabaseHandler import DatabaseHandler
from Trajectory import Trajectory
from CONSTANTS import *
from scipy.spatial import ConvexHull, QhullError
from ruptures.exceptions import BadSegmentationParameters

ray.init()

def get_elliptical_information_of_data_points(X):

    def cart2pol(numpy_point):
        rho = np.linalg.norm(numpy_point)
        phi = np.arctan2(numpy_point[1], numpy_point[0])
        return rho, phi

    #Displacement Process
    distances = distance_matrix(X, X)
    point_a_index, point_b_index = np.unravel_index(np.argmax(distances, axis=None), distances.shape)
    direction_vector = X[point_a_index] - X[point_b_index]

    displacement = (direction_vector/2) + X[point_b_index]

    displaced_X = X - displacement

    #Rotation Process

    _, phi = cart2pol(direction_vector)

    if phi <= 0:
        direction_vector = displaced_X[point_b_index] - displaced_X[point_a_index]
        _, phi = cart2pol(direction_vector)

    rotation_matrix = np.array([
        [np.cos(np.pi - phi), -np.sin(np.pi - phi)],
        [np.sin(np.pi - phi), np.cos(np.pi - phi)]
    ])

    rotated_X = np.dot(rotation_matrix, X.T).T

    a = (np.max(rotated_X[:,0]) - np.min(rotated_X[:,0]))/2 # Semi-Major Axis
    b = (np.max(rotated_X[:,1]) - np.min(rotated_X[:,1]))/2 # Semi-Minor Axis
    e = np.sqrt(1-(np.power(b,2)/np.power(a,2)))#Eccentricity

    return a,b,e

@ray.remote
def analyze_trajectory(trajectory_id):
    DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)

    NUMBER_OF_POINTS_FOR_MSD = 250
    NUMBER_OF_POINTS_FOR_SUB_MSD = 50

    trajectories = Trajectory.objects(id=trajectory_id)
    assert len(trajectories) == 1
    trajectory = trajectories[0]

    #if 'analysis' in trajectory.info or trajectory.length == 1:
    if trajectory.length == 1:
        return None
    else:
        trajectory.info['analysis'] = {}

    trajectory.info['immobile'] = trajectory.is_immobile(4.295)
    trajectory.info['ratio'] = trajectory.normalized_ratio

    try:
        directional_coefficient = trajectory.directional_correlation(steps_lag=1, window_size=25)
        trajectory.info['analysis']['directional_coefficient'] = directional_coefficient
    except ValueError:
        pass

    """
    try:
        segments_mean, break_points = trajectory.directional_correlation_segmentation(steps_lag=1,window_size=11, min_size=9, return_break_points=True)
        trajectory.info['analysis']['directional_coefficient_segments_mean'] = segments_mean
        trajectory.info['analysis']['directional_coefficient_break_points'] = break_points
    except ValueError:
        pass
    except BadSegmentationParameters:
        pass
    """

    trajectory.info['analysis']['angles_analysis'] = default_angles()
    trajectory.info['analysis']['angles_by_state'] = {
        '0': {
            'label': 'non-confinement',
            'angles': default_angles()
        },
        '1': {
            'label': 'confinement',
            'angles': default_angles()
        },
    }

    trajectory.info['analysis']['confinement-area'] = []
    trajectory.info['analysis']['confinement-a'] = []
    trajectory.info['analysis']['confinement-b'] = []
    trajectory.info['analysis']['confinement-e'] = []
    trajectory.info['analysis']['confinement-steps'] = []
    trajectory.info['analysis']['confinement-k'] = []
    trajectory.info['analysis']['confinement-betha'] = []
    trajectory.info['analysis']['confinement-goodness_of_fit'] = []
    trajectory.info['analysis']['non-confinement-steps'] = []
    trajectory.info['analysis']['non-confinement-k'] = []
    trajectory.info['analysis']['non-confinement-betha'] = []
    trajectory.info['analysis']['non-confinement-goodness_of_fit'] = []

    trajectory.info['analysis']['confinement_areas_centroids'] = []

    states, intervals = trajectory.confinement_states(return_intervals=True, v_th=33)

    times = []

    for interval in intervals:
        times.append(interval[-1] - interval[0])

    trajectory.info['analysis']['residence_time'] = sum(times)
    trajectory.info['analysis']['inverse_residence_time'] = trajectory.duration - trajectory.info['analysis']['residence_time']
    trajectory.info['analysis']['confinement-states'] = states.tolist()

    if trajectory.info['dataset'] != 'Cholesterol and btx':
        selected_delta_t = DATASET_TO_DELTA_T[trajectory.info['dataset']]
    else:
        selected_delta_t = DATASET_TO_DELTA_T[DATASETS_LIST[2]] if trajectory.info['classified_experimental_condition'] == BTX_NOMENCLATURE else DATASET_TO_DELTA_T[DATASETS_LIST[3]]
    
    reconstructed_trajectory = trajectory.reconstructed_trajectory(selected_delta_t)

    if reconstructed_trajectory.length > NUMBER_OF_POINTS_FOR_MSD + 2:
        _,_,betha,k,goodness_of_fit = reconstructed_trajectory.temporal_average_mean_squared_displacement(log_log_fit_limit=NUMBER_OF_POINTS_FOR_MSD)
        trajectory.info['analysis']['betha'] = betha
        trajectory.info['analysis']['k'] = k
        trajectory.info['analysis']['goodness_of_fit'] = goodness_of_fit

    for angle in trajectory.info['analysis']['angles_analysis']:
        trajectory.info['analysis']['angles_analysis'][angle] = trajectory.turning_angles(steps_lag=int(angle))

    sub_trajectories_by_state = trajectory.sub_trajectories_trajectories_from_confinement_states(v_th=33)
    for state in sub_trajectories_by_state:
        for sub_trajectory in sub_trajectories_by_state[state]:

            if state == 1:
                try:
                    raw_trajectory = np.zeros((sub_trajectory.length, 2))
                    raw_trajectory[:,0] = sub_trajectory.get_noisy_x()
                    raw_trajectory[:,1] = sub_trajectory.get_noisy_y()
                    area = ConvexHull(raw_trajectory).volume

                    a,b,e = get_elliptical_information_of_data_points(raw_trajectory)

                    trajectory.info['analysis']['confinement_areas_centroids'].append(np.mean(raw_trajectory, axis=0).tolist())

                    trajectory.info['analysis']['confinement-steps'].append(sub_trajectory.length)
                    trajectory.info['analysis']['confinement-area'].append(area)
                    trajectory.info['analysis']['confinement-a'].append(a)
                    trajectory.info['analysis']['confinement-b'].append(b)
                    trajectory.info['analysis']['confinement-e'].append(e)

                    reconstructed_sub_trajectory = sub_trajectory.reconstructed_trajectory(selected_delta_t)
                    if reconstructed_sub_trajectory.length > NUMBER_OF_POINTS_FOR_SUB_MSD + 2:
                        _,_,betha,k,goodness_of_fit = reconstructed_sub_trajectory.temporal_average_mean_squared_displacement(log_log_fit_limit=NUMBER_OF_POINTS_FOR_SUB_MSD)
                        trajectory.info['analysis']['confinement-k'].append(k)
                        trajectory.info['analysis']['confinement-betha'].append(betha)
                        trajectory.info['analysis']['confinement-goodness_of_fit'].append(goodness_of_fit)

                except QhullError:
                    pass
            else:
                trajectory.info['analysis']['non-confinement-steps'].append(sub_trajectory.length)
                reconstructed_sub_trajectory = sub_trajectory.reconstructed_trajectory(selected_delta_t)
                if reconstructed_sub_trajectory.length > NUMBER_OF_POINTS_FOR_SUB_MSD + 2:
                    _,_,betha,k,goodness_of_fit = reconstructed_sub_trajectory.temporal_average_mean_squared_displacement(log_log_fit_limit=NUMBER_OF_POINTS_FOR_SUB_MSD)
                    trajectory.info['analysis']['non-confinement-k'].append(k)
                    trajectory.info['analysis']['non-confinement-betha'].append(betha)
                    trajectory.info['analysis']['non-confinement-goodness_of_fit'].append(goodness_of_fit)

            for angle in trajectory.info['analysis']['angles_by_state'][str(state)]['angles']:
                trajectory.info['analysis']['angles_by_state'][str(state)]['angles'][angle] += sub_trajectory.turning_angles(steps_lag=int(angle))

    trajectory.save()

    DatabaseHandler.disconnect()

DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)
uploaded_trajectories_ids = [str(trajectory_result['_id']) for trajectory_result in Trajectory._get_collection().find({}, {'_id':1})]
DatabaseHandler.disconnect()

for id_batch in tqdm.tqdm(list(batch(uploaded_trajectories_ids, n=1000))):
    ray.get([analyze_trajectory.remote(an_id) for an_id in id_batch])
