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

def analyze_trajectory(trajectory_id):
    trajectories = Trajectory.objects(id=trajectory_id)
    assert len(trajectories) == 1
    trajectory = trajectories[0]

    if 'analysis' not in trajectory.info:
        return None

    trajectory.info['analysis']['confinement-k'] = []
    trajectory.info['analysis']['confinement-betha'] = []
    trajectory.info['analysis']['confinement-goodness_of_fit'] = []
    trajectory.info['analysis']['non-confinement-k'] = []
    trajectory.info['analysis']['non-confinement-betha'] = []
    trajectory.info['analysis']['non-confinement-goodness_of_fit'] = []

    try:
        _,_,betha,k,goodness_of_fit = trajectory.temporal_average_mean_squared_displacement(log_log_fit_limit=MAX_T, limit_type='time', bin_width=DELTA_T)
        trajectory.info['analysis']['betha'] = betha
        trajectory.info['analysis']['k'] = k
        trajectory.info['analysis']['goodness_of_fit'] = goodness_of_fit
    except AssertionError:
        try:
            del trajectory.info['analysis']['betha']
            del trajectory.info['analysis']['k']
            del trajectory.info['analysis']['goodness_of_fit']
        except KeyError:
            pass
    except ValueError:
        try:
            del trajectory.info['analysis']['betha']
            del trajectory.info['analysis']['k']
            del trajectory.info['analysis']['goodness_of_fit']
        except KeyError:
            pass

    sub_trajectories_by_state = trajectory.sub_trajectories_trajectories_from_confinement_states(v_th=33, transition_fix_threshold=5, use_info=True)
    for state in sub_trajectories_by_state:
        for sub_trajectory in sub_trajectories_by_state[state]:

            if state == 1:
                try:
                    _,_,betha,k,goodness_of_fit = sub_trajectory.temporal_average_mean_squared_displacement(log_log_fit_limit=SUB_MAX_T, limit_type='time', bin_width=DELTA_T)
                    trajectory.info['analysis']['confinement-k'].append(k)
                    trajectory.info['analysis']['confinement-betha'].append(betha)
                    trajectory.info['analysis']['confinement-goodness_of_fit'].append(goodness_of_fit)
                except AssertionError:
                    pass
                except ValueError:
                    pass
            else:
                try:
                    _,_,betha,k,goodness_of_fit = sub_trajectory.temporal_average_mean_squared_displacement(log_log_fit_limit=SUB_MAX_T, limit_type='time', bin_width=DELTA_T)
                    trajectory.info['analysis']['non-confinement-k'].append(k)
                    trajectory.info['analysis']['non-confinement-betha'].append(betha)
                    trajectory.info['analysis']['non-confinement-goodness_of_fit'].append(goodness_of_fit)
                except AssertionError:
                    pass
                except ValueError:
                    pass
    trajectory.save()

DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)
i = 0
for id_batch in tqdm.tqdm(Trajectory._get_collection().find({}, {'_id':1})):
    if i >= 50000:
        break
    analyze_trajectory(str(id_batch['_id']))
    i+=1
DatabaseHandler.disconnect()
