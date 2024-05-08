import tqdm
import numpy as np
from scipy.optimize import minimize, LinearConstraint
from Trajectory import Trajectory
from DatabaseHandler import DatabaseHandler
from CONSTANTS import *
from scipy.stats import sem
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd


DELTA_T = 0.0002
DIMENSION = 2
R = 1/6
SEGMENT_LENGTH = 500

def equation_free(x, D, LOCALIZATION_PRECISION):
    TERM_1 = 2*DIMENSION*D*DELTA_T*(x-(2*R))
    TERM_2 = 2*DIMENSION*(LOCALIZATION_PRECISION**2)
    return TERM_1 + TERM_2

def equation_hop(x, DM, DU, L_HOP, LOCALIZATION_PRECISION):
    TERM_1_1_1 = (DU-DM)/DU
    TERM_1_1_2 = (L_HOP**2)/(6*DIMENSION*x*DELTA_T)
    TERM_1_1_3 = 1 - (np.exp(-((12*DU*x*DELTA_T)/(L_HOP**2))))

    TERM_1_1 = 2*DIMENSION*DELTA_T
    TERM_1_2 = DM + (TERM_1_1_1*TERM_1_1_2*TERM_1_1_3)
    TERM_1_3 = (x-(2*R))
    TERM_1 = TERM_1_1 * TERM_1_2 * TERM_1_3

    TERM_2 = 2*DIMENSION*(LOCALIZATION_PRECISION**2)
    return TERM_1 + TERM_2

def equation_confined(x, DU, L_HOP, LOCALIZATION_PRECISION):
    return equation_hop(x, 0, DU, L_HOP, LOCALIZATION_PRECISION)

def free_fitting(X,Y):
    def eq_4_obj_raw(x, y, d, delta): return np.sum((1/x)*(y - equation_free(x, d, delta))**2)

    select_indexes = np.unique(np.geomspace(1,len(X), len(X)).astype(int))-1
    X = X[select_indexes]
    Y = Y[select_indexes]

    eq_4_obj = lambda coeffs: eq_4_obj_raw(X, Y, *coeffs)
    res_eq_4s = []

    for _ in range(100):        
        x0=[np.random.uniform(100, 100000), np.random.uniform(1, 100)]
        res_eq_4 = minimize(eq_4_obj, x0=x0, bounds=[(100, None), (1, None)])
        res_eq_4s.append(res_eq_4)

    return min(res_eq_4s, key=lambda r: r.fun)

def hop_fitting(X,Y):
    def eq_9_obj_raw(x, y, dm, du, l_hop, delta): return np.sum((1/x)*(y - equation_hop(x, dm, du, l_hop, delta))**2)

    select_indexes = np.unique(np.geomspace(1,len(X), len(X)).astype(int))-1
    X = X[select_indexes]
    Y = Y[select_indexes]

    eq_9_obj = lambda coeffs: eq_9_obj_raw(X, Y, *coeffs)
    res_eq_9s = []

    for _ in range(100):        
        x0=[np.random.uniform(100, 100000), np.random.uniform(100, 100000), np.random.uniform(10, 1000), np.random.uniform(1, 100)]
        res_eq_9 = minimize(eq_9_obj, x0=x0, bounds=[(100, None), (100, None), (10, None), (1, None)], constraints=[LinearConstraint([-1,1,0,0], lb=0, ub=np.inf)])
        res_eq_9s.append(res_eq_9)

    return min(res_eq_9s, key=lambda r: r.fun)

def confined_fitting(X,Y):
    def eq_9_obj_raw(x, y, du, l, delta): return np.sum((1/x)*(y - equation_confined(x, du, l, delta))**2)

    select_indexes = np.unique(np.geomspace(1,len(X), len(X)).astype(int))-1
    X = X[select_indexes]
    Y = Y[select_indexes]

    eq_9_obj = lambda coeffs: eq_9_obj_raw(X, Y, *coeffs)
    res_eq_9s = []

    for _ in range(100):        
        x0=[np.random.uniform(100, 100000), np.random.uniform(10, 1000), np.random.uniform(1, 100)]
        res_eq_9 = minimize(eq_9_obj, x0=x0, bounds=[(100, None), (10, None), (1, None)])
        res_eq_9s.append(res_eq_9)

    return min(res_eq_9s, key=lambda r: r.fun)

fitting_dictionary = {
    'free': {
        'fitting': free_fitting,
        'equation': equation_free,
        'number_of_free_parameters': 2,
        'min_msd': None,
        'mean_msd_result': None,
        'color': 'blue',
        'title': 'Free Diffusion',
    },
    'hop': {
        'fitting': hop_fitting,
        'equation': equation_hop,
        'number_of_free_parameters': 4,
        'min_msd': None,
        'mean_msd_result': None,
        'color': 'red',
        'title': 'Hop Diffusion'
    },
    'confined': {
        'fitting': confined_fitting,
        'equation': equation_confined,
        'number_of_free_parameters': 3,
        'min_msd': None,
        'mean_msd_result': None,
        'color': 'green',
        'title': 'Confined Diffusion'
    }
}

DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)

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
    basic_query_dict = {'info.dataset': dataset, 'info.immobile': False} if index < len(INDIVIDUAL_DATASETS) else {'info.dataset': dataset[0], 'info.classified_experimental_condition':dataset[1], 'info.immobile': False}
    for a_key in fitting_dictionary:
        fitting_dictionary[a_key]['msds'] = []
        fitting_dictionary[a_key]['t_msds'] = []
        fitting_dictionary[a_key]['segments'] = []
    counter = 0
    limit = None
    for t in tqdm.tqdm(Trajectory._get_collection().find(basic_query_dict, {'_id':1, 'x':1, 'y':1, 't':1,'info.analysis.betha':1})):
        counter += 1
        if limit is not None and counter > limit:
            break
        trajectory = Trajectory(
            x=np.array(t['x'])*1000,
            y=np.array(t['y'])*1000,
            t=np.append(0,np.cumsum(np.round(np.diff(t['t']),4))),
            noisy=True
        )

        if 'betha' not in t['info']['analysis'] or t['info']['analysis']['betha'] > 1.1:
            continue

        segments = [trajectory.build_noisy_subtrajectory_from_range(i, i+SEGMENT_LENGTH) for i in range(0,trajectory.length, SEGMENT_LENGTH)]

        for segment in segments:
            if segment.length != SEGMENT_LENGTH:
                continue

            t_msd, msd = segment.calculate_msd_curve(bin_width=DELTA_T)
            #msd = msd[:int(len(msd)*0.20)]
            n = len(msd)
            Y = np.array(msd)
            X = (np.array(t_msd[:len(Y)])/DELTA_T).astype(int)#np.array(range(1,len(Y)+1))

            X_aux = X[:int(len(msd)*0.20)]
            Y_aux = Y[:int(len(msd)*0.20)]

            fitting_cache = {}

            #plt.plot(X_aux,Y_aux, color='black')

            for a_key in fitting_dictionary:
                fitting_cache[a_key] = fitting_dictionary[a_key]['fitting'](X_aux,Y_aux)
                #plt.plot(X_aux,fitting_dictionary[a_key]['equation'](X_aux, *fitting_cache[a_key].x), color=fitting_dictionary[a_key]['color'])
                #print(a_key, *fitting_cache[a_key].x)
                fitting_cache[a_key] = n * np.log(fitting_cache[a_key].fun/n) + fitting_dictionary[a_key]['number_of_free_parameters'] * np.log(n)
            MODEL_WITH_LESS_BIC = min(fitting_cache, key=fitting_cache.get)
            #plt.title(MODEL_WITH_LESS_BIC)
            #plt.show()
            #segment.animate_plot()

            fitting_dictionary[MODEL_WITH_LESS_BIC]['msds'].append(Y)
            fitting_dictionary[MODEL_WITH_LESS_BIC]['t_msds'].append(X)
            fitting_dictionary[MODEL_WITH_LESS_BIC]['segments'].append(segment)

    msds_sum = sum([len(fitting_dictionary[a_key]['msds']) for a_key in fitting_dictionary])
    #f, axarr = plt.subplots(len(fitting_dictionary.keys()))
    percentage = {}
    for key_index, a_key in enumerate(fitting_dictionary):
        percentage[a_key] = 100*(len(fitting_dictionary[a_key]['msds'])/msds_sum)

        t_msds_dict = defaultdict(lambda: [])
        t_error_msds_dict = defaultdict(lambda: [])

        for t, msd in zip(fitting_dictionary[a_key]['t_msds'], fitting_dictionary[a_key]['msds']):
            for t_i, msd_i in zip(t,msd):
                t_msds_dict[t_i].append(msd_i)

        for t_i in t_msds_dict:
            t_error_msds_dict[t_i] = np.std(t_msds_dict[t_i])/np.sqrt(len(t_msds_dict[t_i]))
            t_msds_dict[t_i] = np.mean(t_msds_dict[t_i])

        aux = np.array(sorted(list(zip(list(t_msds_dict.keys()), list(t_msds_dict.values()))), key=lambda x: x[0]))
        average_msd_t, average_msd = aux[:,0], aux[:,1]

        aux = np.array(sorted(list(zip(list(t_error_msds_dict.keys()), list(t_error_msds_dict.values()))), key=lambda x: x[0]))
        _, error_msd = aux[:,0], aux[:,1]

        average_msd = average_msd
        average_msd_t = average_msd_t
        error_msd = error_msd
        n = len(average_msd_t)

        fitting_dictionary[a_key]['error_msds'] = error_msd
        fitting_dictionary[a_key]['msds'] = average_msd
        fitting_dictionary[a_key]['x_msds'] = average_msd_t
        fitting_dictionary[a_key]['mean_msd_result'] = fitting_dictionary[a_key]['fitting'](fitting_dictionary[a_key]['x_msds'][:int(n*0.20)], fitting_dictionary[a_key]['msds'][:int(n*0.20)])
        #print(key_index, *fitting_dictionary[a_key]['mean_msd_result'].x)
        #fake_x = np.arange(average_msd_t[0],average_msd_t[:25][-1]+1,0.1)

        #axarr[key_index].errorbar((fitting_dictionary[a_key]['x_msds']*DELTA_T)[:25], fitting_dictionary[a_key]['msds'][:25], yerr=fitting_dictionary[a_key]['error_msds'][:25], color=fitting_dictionary[a_key]['color'], linewidth=1, fmt ='o')
        #axarr[key_index].plot((fake_x*DELTA_T), fitting_dictionary[a_key]['equation'](fake_x, *fitting_dictionary[a_key]['mean_msd_result'].x), color='black', linewidth=2)
        #axarr[key_index].set_title(fitting_dictionary[a_key]['title'])
        #axarr[key_index].set_ylabel(r'$MSD [nm^{2} s^{-1}]$')

        #if key_index == len(fitting_dictionary.keys())-1:
        #    axarr[key_index].set_xlabel(r'$t_{lag} [s]$')

    """
    limit = axarr[1].get_ylim()

    first_line = X_HOP * du
    #first_line -= min(first_line)

    second_line = X_HOP * dm
    #offset = second_line[-1] - hop_mean_msd[-1]
    #second_line -= offset

    #axarr[1].plot(second_line, 'black', linewidth=1)
    axarr[1].set_ylim(limit)
    """
    #du = ('{:,}'.format(int(du))) 
    #dm = ('{:,}'.format(int(dm))) 
    #d = ('{:,}'.format(int(d))) 

    #axarr[0].text(15*DELTA_T, 500, '$D ='+d+' nm^2 s^-1$', fontsize = 10)
    #axarr[1].text(15*DELTA_T, 1500, '$D_{\mu} ='+du+' nm^2 s^-1$', fontsize = 10)
    #axarr[1].text(15*DELTA_T, 500, '$D_{M} ='+dm+' nm^2 s^-1$', fontsize = 10)
    #plt.savefig(f'rickert_{dataset}.png')
    #plt.show()
    #exit()
    result_file = open(f"./Results/{dataset}_hop_vs_free_vs_confined.txt", 'w')
    with pd.ExcelWriter(f"./Results/{dataset}_hop_vs_free_vs_confined.xlsx") as writer:
        for a_key in fitting_dictionary:
            pd.DataFrame({
                'msds': fitting_dictionary[a_key]['msds'],
                'x_msds': fitting_dictionary[a_key]['x_msds'] * DELTA_T,
                'error_msds': fitting_dictionary[a_key]['error_msds'],
            }).to_excel(writer, sheet_name=a_key, index=False)
            result_file.write(f'{a_key}, {percentage[a_key]}, {fitting_dictionary[a_key]["mean_msd_result"].x}\n')
    result_file.close()

DatabaseHandler.disconnect()