"""
ALL trajectories are analyzed.
"""

import ray
import numpy as np
import tqdm
from utils import get_elliptical_information_of_data_points

from DatabaseHandler import DatabaseHandler
from Trajectory import Trajectory
from CONSTANTS import *
from scipy.spatial import ConvexHull, QhullError


ray.init()

@ray.remote
def analyze_trajectory(trajectory_id):
    DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)

    trajectories = Trajectory.objects(id=trajectory_id)
    assert len(trajectories) == 1
    trajectory = trajectories[0]

    if 'analysis' in trajectory.info or trajectory.length == 1:
        return None
    else:
        trajectory.info['analysis'] = {}

    trajectory.info['immobile'] = trajectory.is_immobile(GS_THRESHOLD)
    trajectory.info['ratio'] = trajectory.normalized_ratio

    """
    try:
        directional_coefficient = trajectory.directional_correlation(steps_lag=1, window_size=25)
        trajectory.info['analysis']['directional_coefficient'] = directional_coefficient
    except ValueError:
        pass

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
    trajectory.info['analysis']['confinement-duration'] = []
    trajectory.info['analysis']['confinement-k'] = []
    trajectory.info['analysis']['confinement-betha'] = []
    trajectory.info['analysis']['confinement-goodness_of_fit'] = []
    #trajectory.info['analysis']['confinement-d_2_4'] = []
    trajectory.info['analysis']['non-confinement-steps'] = []
    trajectory.info['analysis']['non-confinement-duration'] = []
    trajectory.info['analysis']['non-confinement-k'] = []
    trajectory.info['analysis']['non-confinement-betha'] = []
    trajectory.info['analysis']['non-confinement-goodness_of_fit'] = []
    #trajectory.info['analysis']['non-confinement-d_2_4'] = []

    trajectory.info['analysis']['confinement_areas_centroids'] = []

    states, intervals = trajectory.confinement_states(return_intervals=True, v_th=33, transition_fix_threshold=3)

    times = []

    for interval in intervals:
        times.append(interval[-1] - interval[0])

    trajectory.info['analysis']['residence_time'] = sum(times)
    trajectory.info['analysis']['inverse_residence_time'] = trajectory.duration - trajectory.info['analysis']['residence_time']
    trajectory.info['analysis']['confinement-states'] = states.tolist()

    try:
        _,_,transport_coefficient,betha,localization_precision,goodness_of_fit = trajectory.temporal_average_mean_squared_displacement(log_log_fit_limit=MAX_T, limit_type='time', bin_width=DELTA_T, time_start=TIME_START, with_corrections=True)
        trajectory.info['analysis']['k'] = transport_coefficient*2*DIMENSION
        trajectory.info['analysis']['betha'] = betha
        trajectory.info['analysis']['goodness_of_fit'] = goodness_of_fit
        trajectory.info['analysis']['localization_precision'] = localization_precision
    except AssertionError:
        pass
    except ValueError:
        pass

    for angle in trajectory.info['analysis']['angles_analysis']:
        trajectory.info['analysis']['angles_analysis'][angle] = trajectory.turning_angles(steps_lag=int(angle))

    trajectory.info['analysis']['meanDP'] = trajectory.mean_turning_angle()
    trajectory.info['analysis']['corrDP'] = trajectory.correlated_turning_angle()
    trajectory.info['analysis']['AvgSignD'] = trajectory.directional_persistance()

    sub_trajectories_by_state = trajectory.sub_trajectories_trajectories_from_confinement_states(v_th=33, transition_fix_threshold=3, use_info=True)
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
                    trajectory.info['analysis']['confinement-duration'].append(sub_trajectory.duration)
                    trajectory.info['analysis']['confinement-steps'].append(sub_trajectory.length)
                    trajectory.info['analysis']['confinement-area'].append(area)
                    trajectory.info['analysis']['confinement-a'].append(a)
                    trajectory.info['analysis']['confinement-b'].append(b)
                    trajectory.info['analysis']['confinement-e'].append(e)

                    try:
                        _,_,transport_coefficient,betha,_,goodness_of_fit = sub_trajectory.temporal_average_mean_squared_displacement(log_log_fit_limit=SUB_MAX_T, limit_type='time', bin_width=DELTA_T, time_start=TIME_START, with_corrections=True)
                        trajectory.info['analysis']['confinement-k'].append(transport_coefficient*2*DIMENSION)
                        trajectory.info['analysis']['confinement-betha'].append(betha)
                        trajectory.info['analysis']['confinement-goodness_of_fit'].append(goodness_of_fit)
                    except AssertionError:
                        trajectory.info['analysis']['confinement-k'].append(None)
                        trajectory.info['analysis']['confinement-betha'].append(None)
                        trajectory.info['analysis']['confinement-goodness_of_fit'].append(None)
                    except ValueError:
                        trajectory.info['analysis']['confinement-k'].append(None)
                        trajectory.info['analysis']['confinement-betha'].append(None)
                        trajectory.info['analysis']['confinement-goodness_of_fit'].append(None)
                except QhullError:
                    trajectory.info['analysis']['confinement_areas_centroids'].append(None)
                    trajectory.info['analysis']['confinement-duration'].append(None)
                    trajectory.info['analysis']['confinement-steps'].append(None)
                    trajectory.info['analysis']['confinement-area'].append(None)
                    trajectory.info['analysis']['confinement-a'].append(None)
                    trajectory.info['analysis']['confinement-b'].append(None)
                    trajectory.info['analysis']['confinement-e'].append(None)
                    trajectory.info['analysis']['confinement-k'].append(None)
                    trajectory.info['analysis']['confinement-betha'].append(None)
                    trajectory.info['analysis']['confinement-goodness_of_fit'].append(None)
            else:
                trajectory.info['analysis']['non-confinement-steps'].append(sub_trajectory.length)
                trajectory.info['analysis']['non-confinement-duration'].append(sub_trajectory.duration)
                try:
                    _,_,transport_coefficient,betha,_,goodness_of_fit = sub_trajectory.temporal_average_mean_squared_displacement(log_log_fit_limit=SUB_MAX_T, limit_type='time', bin_width=DELTA_T, time_start=TIME_START, with_corrections=True)
                    trajectory.info['analysis']['non-confinement-k'].append(transport_coefficient*2*DIMENSION)
                    trajectory.info['analysis']['non-confinement-betha'].append(betha)
                    trajectory.info['analysis']['non-confinement-goodness_of_fit'].append(goodness_of_fit)
                except AssertionError:
                    trajectory.info['analysis']['non-confinement-k'].append(None)
                    trajectory.info['analysis']['non-confinement-betha'].append(None)
                    trajectory.info['analysis']['non-confinement-goodness_of_fit'].append(None)
                except ValueError:
                    trajectory.info['analysis']['non-confinement-k'].append(None)
                    trajectory.info['analysis']['non-confinement-betha'].append(None)
                    trajectory.info['analysis']['non-confinement-goodness_of_fit'].append(None)

            for angle in trajectory.info['analysis']['angles_by_state'][str(state)]['angles']:
                trajectory.info['analysis']['angles_by_state'][str(state)]['angles'][angle] += sub_trajectory.turning_angles(steps_lag=int(angle))

    trajectory.save()

    DatabaseHandler.disconnect()

DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)
uploaded_trajectories_ids = [str(trajectory_result['_id']) for trajectory_result in Trajectory._get_collection().find({}, {'_id':1})]
DatabaseHandler.disconnect()

for id_batch in tqdm.tqdm(list(batch(uploaded_trajectories_ids, n=1000))):
    ray.get([analyze_trajectory.remote(an_id) for an_id in id_batch])
