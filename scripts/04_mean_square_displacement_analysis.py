import ray
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

from DatabaseHandler import DatabaseHandler
from Trajectory import Trajectory
from CONSTANTS import *


NUMBER_OF_POINTS_FOR_MSD = 250

ray.init()

@ray.remote
def analyze_trajectory(trajectory_id):
    DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)

    trajectories = Trajectory.objects(id=trajectory_id)
    assert len(trajectories) == 1
    trajectory = trajectories[0]

    reconstructed_trajectory = trajectory.reconstructed_trajectory(DATASET_TO_DELTA_T[trajectory.info['dataset']])

    if reconstructed_trajectory.length > NUMBER_OF_POINTS_FOR_MSD + 1:
        _,msd,_,_,_ = reconstructed_trajectory.temporal_average_mean_squared_displacement(log_log_fit_limit=NUMBER_OF_POINTS_FOR_MSD)
        msd = msd[:NUMBER_OF_POINTS_FOR_MSD]
    else:
        return None

    DatabaseHandler.disconnect()

    return reconstructed_trajectory, msd

DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)

for dataset in DATASETS_LIST:
    uploaded_trajectories_ids = [str(trajectory_result['_id']) for trajectory_result in Trajectory._get_collection().find({'info.dataset': dataset, 'info.immobile': False}, {'_id':1})]

    results = []

    for id_batch in tqdm.tqdm(batch(uploaded_trajectories_ids, n=1000)):
        results += ray.get([analyze_trajectory.remote(an_id) for an_id in id_batch])

    results = [result for result in results if result is not None]

    number_of_analyzed_trajectories = len(results)
    reconstructed_trajectories_results = [result[0] for result in results]
    msd_results = np.vstack([result[1] for result in results])

    t_lag = np.arange(0,NUMBER_OF_POINTS_FOR_MSD * DATASET_TO_DELTA_T[dataset], DATASET_TO_DELTA_T[dataset])

    for i in range(number_of_analyzed_trajectories):
        plt.loglog(t_lag, msd_results[i],color='cyan')

    plt.loglog(t_lag,t_lag,color='black', linestyle='dashed')

    plt.xlim([plt.xlim()[0], max(t_lag)])
    plt.savefig(f"./Graphics/{dataset}_msd.png", dpi=300)
    plt.clf()

    ea_ta_msd = np.mean(msd_results, axis=0)
    ea_msd, intervals = Trajectory.ensamble_average_mean_square_displacement(reconstructed_trajectories_results, number_of_points_for_msd=NUMBER_OF_POINTS_FOR_MSD)

    plt.plot(t_lag,ea_ta_msd,color='red')
    plt.plot(t_lag,intervals[0],color='blue', linestyle='dashed')
    plt.plot(t_lag,intervals[1],color='blue', linestyle='dashed')
    plt.plot(t_lag,ea_msd,color='blue')
    plt.xlim([plt.xlim()[0], max(t_lag)])
    plt.savefig(f"./Graphics/{dataset}_ea_msd.png", dpi=300)
    plt.clf()

DatabaseHandler.disconnect()
