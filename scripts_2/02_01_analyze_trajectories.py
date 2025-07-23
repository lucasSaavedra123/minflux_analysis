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

    if 'analysis' not in trajectory.info:
        return None

    trajectory.info['analysis']['angles_by_state_deepspt'] = {
        'confinement': {
            'label': 'non-confinement',
            'angles': default_angles()
        },
        'subdifussive': {
            'label': 'subdifussive',
            'angles': default_angles()
        },
        'normal': {
            'label': 'normal',
            'angles': default_angles()
        },
        'directed': {
            'label': 'directed',
            'angles': default_angles()
        },
    }

    for state in ['directed','normal','subdifussive','confinement']:
        states = trajectory.info['analysis'][f'{state}-states-deepspt']

        trajectory.info['analysis'][f'{state}-steps-deepspt'] = []
        trajectory.info['analysis'][f'{state}-duration-deepspt'] = []
        trajectory.info['analysis'][f'{state}-area-deepspt'] = []
        trajectory.info['analysis'][f'{state}-a-deepspt'] = []
        trajectory.info['analysis'][f'{state}-b-deepspt'] = []
        trajectory.info['analysis'][f'{state}-e-deepspt'] = []
        trajectory.info['analysis'][f'{state}-k-deepspt'] = []
        trajectory.info['analysis'][f'{state}-betha-deepspt'] = []
        trajectory.info['analysis'][f'{state}-goodness_of_fit-deepspt'] = []
        trajectory.info['analysis'][f'{state}-anomaly_score-deepspt'] = []

        indices = np.nonzero(states[1:] != states[:-1])[0] + 1
        intervals = np.split(trajectory.get_time(), indices)
        intervals = intervals[0::2] if states[0] else intervals[1::2]
        intervals = [interval for interval in intervals if interval[-1] - interval[0] != 0]

        times = []

        for interval in intervals:
            times.append(interval[-1] - interval[0])

        trajectory.info['analysis'][f'{state}_residence_time-deepspt'] = sum(times)
        trajectory.info['analysis'][f'{state}_inverse_residence_time-deepspt'] = trajectory.duration - trajectory.info['analysis'][f'{state}_residence_time-deepspt']

        sub_trajectories_by_state = trajectory.sub_trajectories_trajectories_from_confinement_states(custom_states=states)
        for sub_trajectory in sub_trajectories_by_state[1]:
            if 'anomaly_scores' not in sub_trajectory.info['analysis']:
                trajectory.info['analysis'][f'{state}-anomaly_score-deepspt'].append(None)
            else:
                trajectory.info['analysis'][f'{state}-anomaly_score-deepspt'].append(np.mean(sub_trajectory.info['analysis']['anomaly_scores']))

            try:
                raw_trajectory = np.zeros((sub_trajectory.length, 2))
                raw_trajectory[:,0] = sub_trajectory.get_noisy_x()
                raw_trajectory[:,1] = sub_trajectory.get_noisy_y()
                area = ConvexHull(raw_trajectory).volume

                a,b,e = get_elliptical_information_of_data_points(raw_trajectory)

                trajectory.info['analysis'][f'{state}-duration-deepspt'].append(sub_trajectory.duration)
                trajectory.info['analysis'][f'{state}-steps-deepspt'].append(sub_trajectory.length)
                trajectory.info['analysis'][f'{state}-area-deepspt'].append(area)
                trajectory.info['analysis'][f'{state}-a-deepspt'].append(a)
                trajectory.info['analysis'][f'{state}-b-deepspt'].append(b)
                trajectory.info['analysis'][f'{state}-e-deepspt'].append(e)

                try:
                    _,_,transport_coefficient,betha,_,goodness_of_fit = sub_trajectory.temporal_average_mean_squared_displacement(log_log_fit_limit=SUB_MAX_T, limit_type='time', bin_width=DELTA_T, time_start=TIME_START, with_corrections=True)
                    trajectory.info['analysis'][f'{state}-k-deepspt'].append(transport_coefficient*2*DIMENSION)
                    trajectory.info['analysis'][f'{state}-betha-deepspt'].append(betha)
                    trajectory.info['analysis'][f'{state}-goodness_of_fit-deepspt'].append(goodness_of_fit)
                except AssertionError:
                    trajectory.info['analysis'][f'{state}-k-deepspt'].append(None)
                    trajectory.info['analysis'][f'{state}-betha-deepspt'].append(None)
                    trajectory.info['analysis'][f'{state}-goodness_of_fit-deepspt'].append(None)
                except ValueError:
                    trajectory.info['analysis'][f'{state}-k-deepspt'].append(None)
                    trajectory.info['analysis'][f'{state}-betha-deepspt'].append(None)
                    trajectory.info['analysis'][f'{state}-goodness_of_fit-deepspt'].append(None)
            except QhullError:
                trajectory.info['analysis'][f'{state}-duration-deepspt'].append(None)
                trajectory.info['analysis'][f'{state}-steps-deepspt'].append(None)
                trajectory.info['analysis'][f'{state}-area-deepspt'].append(None)
                trajectory.info['analysis'][f'{state}-a-deepspt'].append(None)
                trajectory.info['analysis'][f'{state}-b-deepspt'].append(None)
                trajectory.info['analysis'][f'{state}-e-deepspt'].append(None)
                trajectory.info['analysis'][f'{state}-k-deepspt'].append(None)
                trajectory.info['analysis'][f'{state}-betha-deepspt'].append(None)
                trajectory.info['analysis'][f'{state}-goodness_of_fit-deepspt'].append(None)

            for angle in trajectory.info['analysis']['angles_by_state_deepspt'][state]['angles']:
                trajectory.info['analysis']['angles_by_state_deepspt'][state]['angles'][angle] += sub_trajectory.turning_angles(steps_lag=int(angle))

    trajectory.save()

    DatabaseHandler.disconnect()

DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)
uploaded_trajectories_ids = [str(trajectory_result['_id']) for trajectory_result in Trajectory._get_collection().find({}, {'_id':1})]
DatabaseHandler.disconnect()

for id_batch in tqdm.tqdm(list(batch(uploaded_trajectories_ids, n=1000))):
    ray.get([analyze_trajectory.remote(an_id) for an_id in id_batch])
