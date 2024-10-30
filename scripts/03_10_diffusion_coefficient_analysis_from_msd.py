from collections import defaultdict
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.spatial import ConvexHull, QhullError

from DatabaseHandler import DatabaseHandler
from Trajectory import Trajectory
from CONSTANTS import *
from scipy.stats import sem
from scipy.optimize import curve_fit
mpl.rcParams['axes.linewidth'] = 2

def equation_anomalous(x, T, B, LOCALIZATION_PRECISION):
    TERM_1 = T*((x*DELTA_T)**(B-1))*2*DIMENSION*DELTA_T*x*(1-((2*R)/x))
    TERM_2 = 2*DIMENSION*(LOCALIZATION_PRECISION**2)
    return TERM_1 + TERM_2 

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
    
    n = np.arange(1,len(t_vec)+1, 1)
    correction = 1-((2*R)/n)

    A = msd/(4*t_vec*correction)
    B = (0.007**2)/(t_vec*correction)

    return trajectory, A-B, t_vec#trajectory, msd, t_vec

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
    fig, ax = plt.subplots(1,1)

    for immobile in [True, False]:
        SEARCH_FIELD = {'info.dataset': dataset, 'info.immobile': immobile} if index < len(INDIVIDUAL_DATASETS) else {'info.dataset': dataset[0], 'info.classified_experimental_condition':dataset[1], 'info.immobile': immobile}
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
        ea_ta_msd_std = defaultdict(lambda: [])

        #print("n->", len(msd_results))
        #t_lag = np.arange(0,NUMBER_OF_POINTS_FOR_MSD * DATASET_TO_DELTA_T[dataset], DATASET_TO_DELTA_T[dataset])
        #t_lag = np.linspace(DATASET_TO_DELTA_T[dataset], DATASET_TO_DELTA_T[dataset] * 250, 250 - 2)

        for t_lag, msd_result in zip(t_lags, msd_results):
            msd_result = msd_result[t_lag < MAX_T]
            t_lag = t_lag[t_lag < MAX_T]
            #ax.plot(t_lag, msd_result, color='#BEB7A4', linewidth=0.1)#'gray', linewidth=0.1)

            for t, m in zip(t_lag, msd_result):
                ea_ta_msd[t].append(m)
                ea_ta_msd_std[t].append(m)

        for t in ea_ta_msd:
            ea_ta_msd_std[t] = sem(ea_ta_msd[t])#np.std(ea_ta_msd[t])
            ea_ta_msd[t] = np.mean(ea_ta_msd[t])

        time_msd = [[t, ea_ta_msd[t]] for t in ea_ta_msd]
        aux = np.array(sorted(time_msd, key=lambda x: x[0]))
        ea_ta_msd_t_vec, ea_ta_msd = aux[:,0], aux[:,1]

        time_msd = [[t, ea_ta_msd_std[t]] for t in ea_ta_msd_std]
        aux = np.array(sorted(time_msd, key=lambda x: x[0]))
        _, ea_ta_msd_std = aux[:,0], aux[:,1]

        #ax.errorbar(ea_ta_msd_t_vec, ea_ta_msd, yerr=ea_ta_msd_std, color='#FF1B1C' if not immobile else 'black', linewidth=2)
        ax.plot(ea_ta_msd_t_vec, ea_ta_msd, color='#FF1B1C' if not immobile else 'black', linewidth=2)
        ax.set_xscale("log")
        ax.set_yscale("log")

        popt, _ = curve_fit(lambda t,b,k: k * (t ** b), ea_ta_msd_t_vec, ea_ta_msd, bounds=((0, 0), (2, np.inf)), maxfev=2000)
        print(popt[0], popt[1])

        """
        plt.loglog(t_lag,popt[1]*(t_lag**popt[0]), color='red')
        """
        ax.set_xlim([min(ea_ta_msd_t_vec), max(ea_ta_msd_t_vec)])
        #ax.set_ylim([0, 1])
        plt.xticks(fontsize=45)
        plt.yticks(fontsize=45)
        ax.tick_params(which='major', direction='out', length=6, width=2, pad=5)
        ax.tick_params(which='minor', direction='out', length=3, width=2, pad=5)
        plt.subplots_adjust(left=0.31, right=0.983, top=0.968, bottom=0.137)
        #plt.yticks([1e-5, 1e-3, 1e-1])
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

    plt.savefig(f"{index}_{dataset}_apparent_diffusion_coefficient.png", dpi=300)
    plt.clf()

DatabaseHandler.disconnect()
