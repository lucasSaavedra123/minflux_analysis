import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

from DatabaseHandler import DatabaseHandler
from Trajectory import Trajectory
from CONSTANTS import *
from scipy.optimize import curve_fit

import ray
ray.init()

NUMBER_OF_POINTS_FOR_MSD = 250

@ray.remote
def analyze_trajectory(trajectory_id, dataset):
    DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)
    trajectories = Trajectory.objects(id=trajectory_id)
    assert len(trajectories) == 1
    trajectory = trajectories[0]
    DatabaseHandler.disconnect()

    try:
        t_vec,msd,_,_,_ = trajectory.temporal_average_mean_squared_displacement(log_log_fit_limit=NUMBER_OF_POINTS_FOR_MSD, bin_width=0.001)
        msd = msd[:NUMBER_OF_POINTS_FOR_MSD]
    except AssertionError as e:
        return None
    except ValueError as e:
        return None

    return trajectory, msd, t_vec

#import warnings
#warnings.simplefilter('error')

INDIVIDUAL_DATASETS = [
    'Control',
    'CDx',
    'BTX680R',
    'CholesterolPEGKK114',
    'CK666-BTX680',
    'CK666-CHOL',
    'BTX640-CHOL-50-nM',
    'BTX640-CHOL-50-nM-LOW-DENSITY',
]

new_datasets_list = INDIVIDUAL_DATASETS.copy()

for combined_dataset in [
    'Cholesterol and btx',
    'CK666-BTX680-CHOL',
    'BTX680-fPEG-CHOL-50-nM',
    'BTX680-fPEG-CHOL-100-nM',
]:
    new_datasets_list.append((combined_dataset, BTX_NOMENCLATURE))
    new_datasets_list.append((combined_dataset, CHOL_NOMENCLATURE))

for index, dataset in enumerate(new_datasets_list):
    print(dataset)
    SEARCH_FIELD = {'info.dataset': dataset, 'info.immobile': False} if index < len(INDIVIDUAL_DATASETS) else {'info.dataset': dataset[0], 'info.classified_experimental_condition':dataset[1], 'info.immobile': False}
    DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)
    uploaded_trajectories_ids = [str(trajectory_result['_id']) for trajectory_result in Trajectory._get_collection().find(SEARCH_FIELD, {'_id':1})]
    DatabaseHandler.disconnect()

    results = []

    for id_batch in tqdm.tqdm(batch(uploaded_trajectories_ids, n=1000)):
        results += ray.get([analyze_trajectory.remote(an_id, dataset) for an_id in id_batch])

    results = [result for result in results if result is not None]

    number_of_analyzed_trajectories = len(results)
    reconstructed_trajectories_results = [result[0] for result in results]
    msd_results = np.vstack([result[1] for result in results])
    t_lag = [result[2] for result in results][0][:NUMBER_OF_POINTS_FOR_MSD]

    #print("n->", len(msd_results))
    #t_lag = np.arange(0,NUMBER_OF_POINTS_FOR_MSD * DATASET_TO_DELTA_T[dataset], DATASET_TO_DELTA_T[dataset])
    #t_lag = np.linspace(DATASET_TO_DELTA_T[dataset], DATASET_TO_DELTA_T[dataset] * 250, 250 - 2)

    for i in range(number_of_analyzed_trajectories):
        plt.loglog(t_lag, msd_results[i],color='cyan')

    ea_ta_msd = np.mean(msd_results, axis=0)
    popt, _ = curve_fit(lambda t,b,k: np.log(k) + (np.log(t) * b), t_lag, np.log(ea_ta_msd), bounds=((0, 0), (2, np.inf)), maxfev=2000)
    print(popt[0], popt[1])

    plt.loglog(t_lag,t_lag, color='black', linestyle='dashed')
    plt.loglog(t_lag,popt[1]*(t_lag**popt[0]), color='red')

    plt.xlim([0.001, NUMBER_OF_POINTS_FOR_MSD * 0.001])
    #plt.ylim([1e-6, 1])
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.subplots_adjust(left=0.17)
    plt.savefig(f"{index}_{dataset}_msd.png", dpi=300)
    plt.clf()

    t_vec, ea_msd, intervals = Trajectory.ensemble_average_mean_square_displacement(reconstructed_trajectories_results, number_of_points_for_msd=NUMBER_OF_POINTS_FOR_MSD)

    plt.plot(t_lag,ea_ta_msd,color='red')
    plt.plot(t_vec,intervals[0],color='blue', linestyle='dashed')
    plt.plot(t_vec,intervals[1],color='blue', linestyle='dashed')
    plt.plot(t_vec,ea_msd,color='blue')
    plt.xlim([0.001, max(t_lag)])
    plt.ylim([0, 0.07])
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.subplots_adjust(left=0.20)
    plt.savefig(f"{index}_{dataset}_ea_msd.png", dpi=300)
    plt.clf()

DatabaseHandler.disconnect()
