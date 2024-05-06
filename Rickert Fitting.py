import tqdm
import numpy as np
from scipy.optimize import minimize, LinearConstraint
from Trajectory import Trajectory
from DatabaseHandler import DatabaseHandler
from CONSTANTS import *
from scipy.stats import sem
import matplotlib.pyplot as plt


DELTA_T = 0.0001
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
    def eq_4_obj_raw(x, y, d, delta): return np.sum((y - equation_free(x, d, delta))**2)

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
    def eq_9_obj_raw(x, y, dm, du, l_hop, delta): return np.sum((y - equation_hop(x, dm, du, l_hop, delta))**2)

    select_indexes = np.unique(np.geomspace(1,len(X), len(X)).astype(int))-1
    X = X[select_indexes]
    Y = Y[select_indexes]

    eq_9_obj = lambda coeffs: eq_9_obj_raw(X, Y, *coeffs)
    res_eq_9s = []

    for _ in range(100):        
        x0=[np.random.uniform(100, 100000), np.random.uniform(100, 100000), np.random.uniform(10, 1000), np.random.uniform(1, 100)]
        res_eq_9 = minimize(eq_9_obj, x0=x0, bounds=[(100, None), (100, None), (10, None), (1, None)], constraints=[LinearConstraint([-5,1,0,0], lb=0, ub=np.inf)])
        res_eq_9s.append(res_eq_9)

    return min(res_eq_9s, key=lambda r: r.fun)

def confined_fitting(X,Y):
    def eq_9_obj_raw(x, y, du, l, delta): return np.sum((y - equation_confined(x, du, l, delta))**2)

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

datasets = [
    'Control', 'CholesterolPEGKK114'
]

DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)
for dataset in datasets:
    for a_key in fitting_dictionary:
        fitting_dictionary[a_key]['msds'] = []
        fitting_dictionary[a_key]['segments'] = []
    counter = 0
    limit = None
    for t in tqdm.tqdm(Trajectory._get_collection().find({'info.immobile':False, 'info.dataset': dataset}, {'_id':1, 'x':1, 'y':1, 't':1,'info.analysis.betha':1})):
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
            msd = msd[:100]#[:int(len(msd)*0.5)]
            Y = np.array(msd)
            X = np.array(range(1,len(Y)+1))
            T = t_msd[:len(Y)]
            n = len(Y)

            fitting_cache = {}

            #plt.plot(X,Y, color='black')

            for a_key in fitting_dictionary:
                fitting_cache[a_key] = fitting_dictionary[a_key]['fitting'](X,Y)
                #plt.plot(X,fitting_dictionary[a_key]['equation'](X, *fitting_cache[a_key].x), color=fitting_dictionary[a_key]['color'])
                #print(a_key, *fitting_cache[a_key].x)
                fitting_cache[a_key] = n * np.log(fitting_cache[a_key].fun/n) + fitting_dictionary[a_key]['number_of_free_parameters'] * np.log(n)
            MODEL_WITH_LESS_BIC = min(fitting_cache, key=fitting_cache.get)
            #plt.title(MODEL_WITH_LESS_BIC)
            #plt.show()
            #segment.animate_plot()

            fitting_dictionary[MODEL_WITH_LESS_BIC]['msds'].append(Y)
            fitting_dictionary[MODEL_WITH_LESS_BIC]['segments'].append(segment)

    msds_sum = sum([len(fitting_dictionary[a_key]['msds']) for a_key in fitting_dictionary])
    f, axarr = plt.subplots(len(fitting_dictionary.keys()))
    for key_index, a_key in enumerate(fitting_dictionary):
        percentage = 100*(len(fitting_dictionary[a_key]['msds'])/msds_sum)
        print(f'{a_key}:', round(percentage, 2))

        fitting_dictionary[a_key]['min_msd'] = np.min([len(y) for y in fitting_dictionary[a_key]['msds']])
        fitting_dictionary[a_key]['msds'] = [msd[:fitting_dictionary[a_key]['min_msd']] for msd in fitting_dictionary[a_key]['msds']]
        fitting_dictionary[a_key]['error_msds'] = np.std(fitting_dictionary[a_key]['msds'], axis=0)/np.sqrt(fitting_dictionary[a_key]['min_msd'])
        fitting_dictionary[a_key]['msds'] = np.mean(fitting_dictionary[a_key]['msds'], axis=0)
        fitting_dictionary[a_key]['x_msds'] = np.arange(1,len(fitting_dictionary[a_key]['msds'])+1,1)
        fitting_dictionary[a_key]['mean_msd_result'] = fitting_dictionary[a_key]['fitting'](fitting_dictionary[a_key]['x_msds'], fitting_dictionary[a_key]['msds'])
        print(key_index, *fitting_dictionary[a_key]['mean_msd_result'].x)
        fake_x = np.arange(1,len(fitting_dictionary[a_key]['msds'])+1,0.1)

        axarr[key_index].errorbar((fitting_dictionary[a_key]['x_msds']*DELTA_T)[::5], fitting_dictionary[a_key]['msds'][::5], yerr=fitting_dictionary[a_key]['error_msds'][::5], color=fitting_dictionary[a_key]['color'], linewidth=1, fmt ='o')
        axarr[key_index].plot((fake_x*DELTA_T)[:230], fitting_dictionary[a_key]['equation'](fake_x, *fitting_dictionary[a_key]['mean_msd_result'].x)[:230], color='black', linewidth=2)
        axarr[key_index].set_title(fitting_dictionary[a_key]['title'])
        axarr[key_index].set_ylabel(r'$MSD [nm^{2} s^{-1}]$')

        if key_index == len(fitting_dictionary.keys())-1:
            axarr[key_index].set_xlabel(r'$t_{lag} [s]$')

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
    plt.savefig(f'rickert_{dataset}.png')
    exit()

DatabaseHandler.disconnect()