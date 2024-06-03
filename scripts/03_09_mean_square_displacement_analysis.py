from collections import defaultdict
import tqdm
import numpy as np
import matplotlib.pyplot as plt

from DatabaseHandler import DatabaseHandler
from Trajectory import Trajectory
from CONSTANTS import *
from scipy.optimize import curve_fit


def analyze_trajectory(trajectory_id, dataset):
    trajectories = Trajectory.objects(id=trajectory_id)
    assert len(trajectories) == 1
    trajectory = trajectories[0]

    try:
        t_vec,msd,_,_,_,_ = trajectory.temporal_average_mean_squared_displacement(log_log_fit_limit=MAX_T, limit_type='time', bin_width=DELTA_T, time_start=TIME_START, with_corrections=True)
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

DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)

for index, dataset in enumerate(new_datasets_list):
    print(dataset)
    SEARCH_FIELD = {'info.dataset': dataset, 'info.immobile': False} if index < len(INDIVIDUAL_DATASETS) else {'info.dataset': dataset[0], 'info.classified_experimental_condition':dataset[1], 'info.immobile': False}
    SEARCH_FIELD.update({'info.analysis.goodness_of_fit': {'$lt':GOODNESS_OF_FIT_MAXIMUM}})
    uploaded_trajectories_ids = [str(trajectory_result['_id']) for trajectory_result in Trajectory._get_collection().find(SEARCH_FIELD, {'_id':1})]

    results = []

    for id_batch in tqdm.tqdm(list(batch(uploaded_trajectories_ids, n=100))):
        results += [analyze_trajectory(an_id, dataset) for an_id in id_batch]

    results = [result for result in results if result is not None]

    number_of_analyzed_trajectories = len(results)
    reconstructed_trajectories_results = [result[0] for result in results]
    msd_results = [result[1] for result in results]
    t_lags = [result[2] for result in results]

    ea_ta_msd = defaultdict(lambda: [])

    #print("n->", len(msd_results))
    #t_lag = np.arange(0,NUMBER_OF_POINTS_FOR_MSD * DATASET_TO_DELTA_T[dataset], DATASET_TO_DELTA_T[dataset])
    #t_lag = np.linspace(DATASET_TO_DELTA_T[dataset], DATASET_TO_DELTA_T[dataset] * 250, 250 - 2)

    for t_lag, msd_result in zip(t_lags, msd_results):
        msd_result = msd_result[t_lag < MAX_T]
        t_lag = t_lag[t_lag < MAX_T]
        plt.loglog(t_lag, msd_result, color='#BEB7A4', linewidth=0.1)#'gray', linewidth=0.1)

        for t, m in zip(t_lag, msd_result):
            ea_ta_msd[t].append(m)

    for t in ea_ta_msd:
        ea_ta_msd[t] = np.mean(ea_ta_msd[t])

    time_msd = [[t, ea_ta_msd[t]] for t in ea_ta_msd]
    aux = np.array(sorted(time_msd, key=lambda x: x[0]))
    ea_ta_msd_t_vec, ea_ta_msd = aux[:,0], aux[:,1]
    
    brown_line = np.linspace(min(ea_ta_msd_t_vec), max(ea_ta_msd_t_vec), 100)
    
    plt.loglog(ea_ta_msd_t_vec, ea_ta_msd, color='#FF1B1C', linewidth=2)#'red')
    plt.loglog(brown_line,brown_line, color='black', linestyle='dashed', linewidth=2)

    popt, _ = curve_fit(lambda t,b,k: k * (t ** b), ea_ta_msd_t_vec, ea_ta_msd, bounds=((0, 0), (2, np.inf)), maxfev=2000)
    print(popt[0], popt[1])

    """
    plt.loglog(t_lag,popt[1]*(t_lag**popt[0]), color='red')
    """
    plt.xlim([min(ea_ta_msd_t_vec), max(ea_ta_msd_t_vec)])
    plt.ylim([10e-6, 20e-1])
    plt.xticks(fontsize=50)
    plt.yticks(fontsize=50)
    plt.subplots_adjust(left=0.31, right=0.983, top=0.968, bottom=0.137)
    plt.yticks([1e-5, 1e-3, 1e-1])
    plt.savefig(f"{index}_{dataset}_msd.png", dpi=300)
    plt.clf()
    """
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
    """

DatabaseHandler.disconnect()
