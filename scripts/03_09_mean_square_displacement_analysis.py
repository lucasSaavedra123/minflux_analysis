import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

from DatabaseHandler import DatabaseHandler
from Trajectory import Trajectory
from CONSTANTS import *
from scipy.optimize import curve_fit

NUMBER_OF_POINTS_FOR_MSD = 250

import warnings

def analyze_trajectory(trajectory_id, dataset):
    trajectories = Trajectory.objects(id=trajectory_id)
    assert len(trajectories) == 1
    trajectory = trajectories[0]
    reconstructed_trajectory = trajectory.reconstructed_trajectory(DATASET_TO_DELTA_T[dataset])

    if trajectory.length <= 1 or trajectory.is_immobile(4.295):
        return None

    try:
        #if trajectory.length > NUMBER_OF_POINTS_FOR_MSD + 2:
        t_vec,msd,_,_,_ = trajectory.temporal_average_mean_squared_displacement(log_log_fit_limit=NUMBER_OF_POINTS_FOR_MSD, bin_width=0.001)
        msd = msd[:NUMBER_OF_POINTS_FOR_MSD]
        #else:
            #return None
    except AssertionError as e:
        #print(e)
        return None
    except ValueError as e:
        return None

    return trajectory, msd, t_vec

DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)
warnings.simplefilter('error')

new_datasets_list = DATASETS_LIST.copy()
new_datasets_list = DATASETS_LIST[:-1]
new_datasets_list.append(BTX_NOMENCLATURE)
new_datasets_list.append(CHOL_NOMENCLATURE)

for index, dataset in enumerate(new_datasets_list):
    print(dataset)

    SEARCH_FIELD = 'info.dataset' if index < 4 else 'info.classified_experimental_condition'

    uploaded_trajectories_ids = [str(trajectory_result['_id']) for trajectory_result in Trajectory._get_collection().find({SEARCH_FIELD: dataset}, {'_id':1})]

    if dataset == 'fPEG-Chol':
        dataset = 'CholesterolPEGKK114'

    results = []

    for id_batch in tqdm.tqdm(batch(uploaded_trajectories_ids, n=1000)):
        results += [analyze_trajectory(an_id, dataset) for an_id in id_batch]

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

    plt.xlim([0.001, max(t_lag)])
    plt.ylim([1e-6, 1])
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.subplots_adjust(left=0.17)
    plt.savefig(f"{index}_{dataset}_msd.png", dpi=300)
    plt.clf()

    ea_msd, intervals = Trajectory.ensemble_average_mean_square_displacement(reconstructed_trajectories_results, number_of_points_for_msd=NUMBER_OF_POINTS_FOR_MSD)

    plt.plot(t_lag,ea_ta_msd,color='red')
    plt.plot(t_lag,intervals[0],color='blue', linestyle='dashed')
    plt.plot(t_lag,intervals[1],color='blue', linestyle='dashed')
    plt.plot(t_lag,ea_msd,color='blue')
    plt.xlim([0.001, max(t_lag)])
    plt.ylim([0, 0.07])
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.subplots_adjust(left=0.20)
    plt.savefig(f"{index}_{dataset}_ea_msd.png", dpi=300)
    plt.clf()

DatabaseHandler.disconnect()
