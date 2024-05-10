import tqdm
import numpy as np
from Trajectory import Trajectory
from DatabaseHandler import DatabaseHandler
from CONSTANTS import *
from scipy.stats import sem
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
from utils import *


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
    'BTX680R',
    'CK666-BTX680',
    'CholesterolPEGKK114',
    'CK666-CHOL',
    #'Control',
    #'BTX640-CHOL-50-nM',
    #'BTX640-CHOL-50-nM-LOW-DENSITY',
    #'CDx',
]

new_datasets_list = INDIVIDUAL_DATASETS.copy()

for combined_dataset in [
    'Cholesterol and btx',
    'CK666-BTX680-CHOL',
    #'BTX680-fPEG-CHOL-50-nM',
    #'BTX680-fPEG-CHOL-100-nM',
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
    #min_msd = float('inf')
    for t in tqdm.tqdm(Trajectory._get_collection().find(basic_query_dict, {'_id':1, 'x':1, 'y':1, 't':1,'info.analysis.betha':1})):
        counter += 1
        if limit is not None and counter > limit:
            break

        trajectory = Trajectory(
            x=np.array(t['x'])*1000,
            y=np.array(t['y'])*1000,
            t=t['t'],
            noisy=True
        )

        if 'betha' not in t['info']['analysis'] or t['info']['analysis']['betha'] > 1.1:
            continue

        segments = [trajectory.build_noisy_subtrajectory_from_range(i, i+SEGMENT_LENGTH) for i in range(0,trajectory.length, SEGMENT_LENGTH)]

        for segment in segments:
            if segment.length != SEGMENT_LENGTH:
                continue

            t_msd, msd = segment.calculate_msd_curve(bin_width=DELTA_T, return_variances=False)
            Y = np.array(msd)
            X = (np.array(t_msd)/DELTA_T)#np.array(range(1,len(Y)+1))
            X_aux = X[t_msd<MAX_T]
            Y_aux = Y[t_msd<MAX_T]
            #min_msd = min(min_msd, X_aux[-1])
            fitting_cache = {}
            n = len(X_aux)

            #plt.plot(X_aux,Y_aux, color='black')

            for a_key in fitting_dictionary:
                fitting_cache[a_key] = fitting_dictionary[a_key]['fitting'](X_aux,Y_aux)
                fun = np.sum((Y_aux - fitting_dictionary[a_key]['equation'](X_aux, *fitting_cache[a_key].x))**2)
                #plt.plot(X_aux,fitting_dictionary[a_key]['equation'](X_aux, *fitting_cache[a_key].x), color=fitting_dictionary[a_key]['color'])
                #print(a_key, *fitting_cache[a_key].x)
                fitting_cache[a_key] = n * np.log(fun/n) + fitting_dictionary[a_key]['number_of_free_parameters'] * np.log(n)#n * np.log(fitting_cache[a_key].fun/n) + fitting_dictionary[a_key]['number_of_free_parameters'] * np.log(n)
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
        variance_msds_dict = defaultdict(lambda: [])

        for t, msd in zip(fitting_dictionary[a_key]['t_msds'], fitting_dictionary[a_key]['msds']):
            for t_i, msd_i in zip(t,msd):
                t_msds_dict[t_i].append(msd_i)

        for t_i in t_msds_dict:
            variance_msds_dict[t_i] = np.var(t_msds_dict[t_i])
            t_error_msds_dict[t_i] = np.std(t_msds_dict[t_i])#/np.sqrt(len(t_msds_dict[t_i]))
            t_msds_dict[t_i] = np.mean(t_msds_dict[t_i])

        aux = np.array(sorted(list(zip(list(t_msds_dict.keys()), list(t_msds_dict.values()))), key=lambda x: x[0]))
        average_msd_t, average_msd = aux[:,0], aux[:,1]

        aux = np.array(sorted(list(zip(list(t_error_msds_dict.keys()), list(t_error_msds_dict.values()))), key=lambda x: x[0]))
        _, error_msd = aux[:,0], aux[:,1]

        aux = np.array(sorted(list(zip(list(variance_msds_dict.keys()), list(variance_msds_dict.values()))), key=lambda x: x[0]))
        _, variance_msd = aux[:,0], aux[:,1]

        average_msd_t_m = average_msd_t * DELTA_T

        average_msd = average_msd[average_msd_t_m < MAX_T]#[:min_msd]
        error_msd = error_msd[average_msd_t_m < MAX_T]#[:min_msd]
        variance_msd = variance_msd[average_msd_t_m < MAX_T]#[:min_msd]
        average_msd_t = average_msd_t[average_msd_t_m < MAX_T]#[:min_msd]

        fitting_dictionary[a_key]['error_msds'] = error_msd
        fitting_dictionary[a_key]['msds'] = average_msd
        fitting_dictionary[a_key]['x_msds'] = average_msd_t
        fitting_dictionary[a_key]['mean_msd_result'] = fitting_dictionary[a_key]['fitting'](fitting_dictionary[a_key]['x_msds'], fitting_dictionary[a_key]['msds'])

    result_file = open(f"./Results/{dataset}_hop_vs_free_vs_confined.txt", 'w')
    with pd.ExcelWriter(f"./Results/{dataset}_hop_vs_free_vs_confined.xlsx") as writer:
        for a_key in fitting_dictionary:
            pd.DataFrame({
                'x_msds': fitting_dictionary[a_key]['x_msds'] * DELTA_T,
                'msds': fitting_dictionary[a_key]['msds'],
                'error_msds': fitting_dictionary[a_key]['error_msds'],
            }).to_excel(writer, sheet_name=a_key, index=False)

            DISCRETE_X = np.linspace(1,int(MAX_T/DELTA_T),1000)#int(min_msd), 1000)
            DISCRETE_Y = fitting_dictionary[a_key]['equation'](DISCRETE_X, *fitting_dictionary[a_key]['mean_msd_result'].x)

            if a_key == 'hop':
                PARAMETERS = fitting_dictionary[a_key]['mean_msd_result'].x
                Y_DU = DISCRETE_X * ((DISCRETE_Y[1] - DISCRETE_Y[0])/(DISCRETE_X[1] - DISCRETE_X[0]))
                Y_DM = equation_hop(DISCRETE_X, PARAMETERS[0], PARAMETERS[0], PARAMETERS[2], PARAMETERS[3])

                OFFSET = Y_DU[0] - DISCRETE_Y[0]
                Y_DU -= OFFSET

                OFFSET = Y_DM[-1] - DISCRETE_Y[-1]
                Y_DM -= OFFSET

                pd.DataFrame({
                    'x': DISCRETE_X*DELTA_T,
                    'y': DISCRETE_Y,
                    'y_du': Y_DU,
                    'y_dm': Y_DM
                }).to_excel(writer, sheet_name=a_key+'_discrete', index=False)
            else:
                pd.DataFrame({
                    'x': DISCRETE_X*DELTA_T,
                    'y': DISCRETE_Y,
                }).to_excel(writer, sheet_name=a_key+'_discrete', index=False)
            result_file.write(f'{a_key}, {percentage[a_key]}, {fitting_dictionary[a_key]["mean_msd_result"].x}\n')

    result_file.close()

DatabaseHandler.disconnect()