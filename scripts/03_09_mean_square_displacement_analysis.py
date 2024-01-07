import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

from DatabaseHandler import DatabaseHandler
from Trajectory import Trajectory
from CONSTANTS import *


NUMBER_OF_POINTS_FOR_MSD = 250

def analyze_trajectory(trajectory_id, dataset):
    trajectories = Trajectory.objects(id=trajectory_id)
    assert len(trajectories) == 1
    trajectory = trajectories[0]
    reconstructed_trajectory = trajectory.reconstructed_trajectory(DATASET_TO_DELTA_T[dataset])

    if reconstructed_trajectory.length > NUMBER_OF_POINTS_FOR_MSD + 1:
        _,msd,_,_,_ = reconstructed_trajectory.temporal_average_mean_squared_displacement(log_log_fit_limit=NUMBER_OF_POINTS_FOR_MSD)
        msd = msd[:NUMBER_OF_POINTS_FOR_MSD]
    else:
        return None


    return reconstructed_trajectory, msd

DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)

new_datasets_list = DATASETS_LIST.copy()
new_datasets_list = DATASETS_LIST[:-1]
new_datasets_list.append(BTX_NOMENCLATURE)
new_datasets_list.append(CHOL_NOMENCLATURE)

for index, dataset in enumerate(new_datasets_list):
    if index < 5:
        continue
    print(dataset)
    SEARCH_FIELD = 'info.dataset' if index < 4 else 'info.classified_experimental_condition'

    uploaded_trajectories_ids = [str(trajectory_result['_id']) for trajectory_result in Trajectory._get_collection().find({SEARCH_FIELD: dataset, 'info.immobile': False}, {'_id':1})]

    if dataset == 'fPEG-Chol':
        dataset = 'CholesterolPEGKK114'

    results = []

    for id_batch in tqdm.tqdm(batch(uploaded_trajectories_ids, n=1000)):
        results += [analyze_trajectory(an_id, dataset) for an_id in id_batch]

    results = [result for result in results if result is not None]

    number_of_analyzed_trajectories = len(results)
    reconstructed_trajectories_results = [result[0] for result in results]
    msd_results = np.vstack([result[1] for result in results])
    #print("n->", len(msd_results))
    t_lag = np.arange(0,NUMBER_OF_POINTS_FOR_MSD * DATASET_TO_DELTA_T[dataset], DATASET_TO_DELTA_T[dataset])

    for i in range(number_of_analyzed_trajectories):
        plt.loglog(t_lag, msd_results[i],color='cyan')

    plt.loglog(t_lag,t_lag,color='black', linestyle='dashed')

    plt.xlim([plt.xlim()[0], max(t_lag)])
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.subplots_adjust(left=0.17)
    plt.savefig(f"{index}_{dataset}_msd.png", dpi=300)
    plt.clf()

    ea_ta_msd = np.mean(msd_results, axis=0)
    ea_msd, intervals = Trajectory.ensamble_average_mean_square_displacement(reconstructed_trajectories_results, number_of_points_for_msd=NUMBER_OF_POINTS_FOR_MSD)

    plt.plot(t_lag,ea_ta_msd,color='red')
    plt.plot(t_lag,intervals[0],color='blue', linestyle='dashed')
    plt.plot(t_lag,intervals[1],color='blue', linestyle='dashed')
    plt.plot(t_lag,ea_msd,color='blue')
    plt.xlim([plt.xlim()[0], max(t_lag)])
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.subplots_adjust(left=0.20)
    plt.savefig(f"{index}_{dataset}_ea_msd.png", dpi=300)
    plt.clf()

DatabaseHandler.disconnect()
