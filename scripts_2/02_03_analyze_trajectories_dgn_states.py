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
    if 'eco_cleaned' not in trajectory.info['analysis']:
        return None

    trajectory.info['analysis']['angles_by_state_dgn'] = {
        '0': {
            'label': '0',
            'angles': default_angles()
        },
        '1': {
            'label': '1',
            'angles': default_angles()
        },
        '2': {
            'label': '2',
            'angles': default_angles()
        },
        '3': {
            'label': '3',
            'angles': default_angles()
        },
        '4': {
            'label': '4',
            'angles': default_angles()
        },
    }

    for dgn_state in range(5):
        if f'eco_state_{dgn_state}' not in trajectory.info['analysis']:
            continue
        states = trajectory.info['analysis'][f'eco_state_{dgn_state}']

        trajectory.info['analysis'][f'{dgn_state}-steps-dgn'] = []
        trajectory.info['analysis'][f'{dgn_state}-duration-dgn'] = []
        trajectory.info['analysis'][f'{dgn_state}-k-dgn'] = []
        trajectory.info['analysis'][f'{dgn_state}-betha-dgn'] = []
        trajectory.info['analysis'][f'{dgn_state}-goodness_of_fit-dgn'] = []
        trajectory.info['analysis'][f'{dgn_state}-deepspt_state'] = []

        indices = np.nonzero(states[1:] != states[:-1])[0] + 1
        intervals = np.split(trajectory.get_time(), indices)
        intervals = intervals[0::2] if states[0] else intervals[1::2]
        intervals = [interval for interval in intervals if interval[-1] - interval[0] != 0]

        times = []

        for interval in intervals:
            times.append(interval[-1] - interval[0])

        trajectory.info['analysis'][f'{dgn_state}_residence_time-dgn'] = sum(times)
        trajectory.info['analysis'][f'{dgn_state}_inverse_residence_time-dgn'] = trajectory.duration - trajectory.info['analysis'][f'{dgn_state}_residence_time-dgn']

        sub_trajectories_by_state = trajectory.sub_trajectories_trajectories_from_confinement_states(custom_states=states)
        for sub_trajectory in sub_trajectories_by_state[1]:
            if 'deepspt_segmenter_result' in sub_trajectory.info['analysis']:
                deepspt_states = sub_trajectory.info['analysis']['deepspt_segmenter_result']
                trajectory.info['analysis'][f'{dgn_state}-deepspt_state'].append(max(set(deepspt_states), key=deepspt_states.count))
            else:
                trajectory.info['analysis'][f'{dgn_state}-deepspt_state'].append(None)

            try:
                assert sub_trajectory.length > 1
                trajectory.info['analysis'][f'{dgn_state}-duration-dgn'].append(sub_trajectory.duration)
                trajectory.info['analysis'][f'{dgn_state}-steps-dgn'].append(sub_trajectory.length)
                try:
                    _,_,transport_coefficient,betha,_,goodness_of_fit = sub_trajectory.temporal_average_mean_squared_displacement(log_log_fit_limit=SUB_MAX_T, limit_type='time', bin_width=DELTA_T, time_start=TIME_START, with_corrections=True)
                    trajectory.info['analysis'][f'{dgn_state}-k-dgn'].append(transport_coefficient*2*DIMENSION)
                    trajectory.info['analysis'][f'{dgn_state}-betha-dgn'].append(betha)
                    trajectory.info['analysis'][f'{dgn_state}-goodness_of_fit-dgn'].append(goodness_of_fit)
                except AssertionError:
                    trajectory.info['analysis'][f'{dgn_state}-k-dgn'].append(None)
                    trajectory.info['analysis'][f'{dgn_state}-betha-dgn'].append(None)
                    trajectory.info['analysis'][f'{dgn_state}-goodness_of_fit-dgn'].append(None)
                except ValueError:
                    trajectory.info['analysis'][f'{dgn_state}-k-dgn'].append(None)
                    trajectory.info['analysis'][f'{dgn_state}-betha-dgn'].append(None)
                    trajectory.info['analysis'][f'{dgn_state}-goodness_of_fit-dgn'].append(None)
            except QhullError:
                trajectory.info['analysis'][f'{dgn_state}-duration-dgn'].append(None)
                trajectory.info['analysis'][f'{dgn_state}-steps-dgn'].append(None)
                trajectory.info['analysis'][f'{dgn_state}-k-dgn'].append(None)
                trajectory.info['analysis'][f'{dgn_state}-betha-dgn'].append(None)
                trajectory.info['analysis'][f'{dgn_state}-goodness_of_fit-dgn'].append(None)
            except AssertionError:
                trajectory.info['analysis'][f'{dgn_state}-duration-dgn'].append(None)
                trajectory.info['analysis'][f'{dgn_state}-steps-dgn'].append(None)
                trajectory.info['analysis'][f'{dgn_state}-k-dgn'].append(None)
                trajectory.info['analysis'][f'{dgn_state}-betha-dgn'].append(None)
                trajectory.info['analysis'][f'{dgn_state}-goodness_of_fit-dgn'].append(None)

            for angle in trajectory.info['analysis']['angles_by_state_dgn'][str(dgn_state)]['angles']:
                trajectory.info['analysis']['angles_by_state_dgn'][str(dgn_state)]['angles'][angle] += sub_trajectory.turning_angles(steps_lag=int(angle))

    trajectory.save()

    DatabaseHandler.disconnect()

DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)
uploaded_trajectories_ids = [str(trajectory_result['_id']) for trajectory_result in Trajectory._get_collection().find({}, {'_id':1})]
DatabaseHandler.disconnect()

for id_batch in tqdm.tqdm(list(batch(uploaded_trajectories_ids, n=1000))):
    ray.get([analyze_trajectory.remote(an_id) for an_id in id_batch])
