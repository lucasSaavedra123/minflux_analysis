
import tqdm
import numpy as np
from scipy.optimize import minimize, LinearConstraint
from Trajectory import Trajectory
from DatabaseHandler import DatabaseHandler
from CONSTANTS import *
from scipy.stats import sem
import matplotlib.pyplot as plt

class DummyObject():
    def __init__(self, x, fun):
        self.fun = fun
        self.x = x

DELTA_T = 0.001
DIMENSION = 2
R = 1/6
SEGMENT_LENGTH = 500

def equation_free(x, D, LOCALIZATION_PRECISION):
    TERM_1 = 2*DIMENSION*D*DELTA_T*(x-(2*R))
    TERM_2 = 2*DIMENSION*(LOCALIZATION_PRECISION**2)
    return TERM_1 + TERM_2

def equation_anomalous(x, D_BETHA, BETHA, LOCALIZATION_PRECISION):
    TERM_1 = 2*DIMENSION*D_BETHA*DELTA_T*((x**BETHA)-(2*R))
    TERM_2 = 2*DIMENSION*(LOCALIZATION_PRECISION**2)
    return TERM_1 + TERM_2

def equation_hop(x, DM, DU, L_HOP, LOCALIZATION_PRECISION):
    TERM_1_1_1 = (DU-DM)/DU
    TERM_1_1_2 = (L_HOP**2)/(6*DIMENSION*x*DELTA_T)
    TERM_1_1_3 = 1 - (np.exp(-((12*x*DELTA_T*DU)/(L_HOP**2))))

    TERM_1_1 = DM + (TERM_1_1_1*TERM_1_1_2*TERM_1_1_3)

    TERM_1_2 = (x-(2*R))
    TERM_1_3 = 2*DIMENSION*DELTA_T
    TERM_1 = TERM_1_1 * TERM_1_2 * TERM_1_3

    TERM_2 = 2*DIMENSION*(LOCALIZATION_PRECISION**2)
    return TERM_1 + TERM_2

def free_fitting(X,Y):
    def eq_4_obj_raw(x, y, d, delta): return np.sum((y - equation_free(x, d, delta))**2)

    select_indexes = np.unique(np.geomspace(1,len(X), len(X)).astype(int))-1
    X = X[select_indexes]
    Y = Y[select_indexes]

    eq_4_obj = lambda coeffs: eq_4_obj_raw(X, Y, *coeffs)
    res_eq_4s = []

    for _ in range(100):        
        x0=[np.random.uniform(100, 100000), np.random.uniform(1, 100)]
        res_eq_4 = minimize(eq_4_obj, x0=x0, bounds=[(1, None), (1, None)])
        res_eq_4s.append(res_eq_4)

    return min(res_eq_4s, key=lambda r: r.fun)

def hop_fitting(X,Y):
    def eq_9_obj_raw(x, y, dm, du, l_hop, delta): return np.sum((y - equation_hop(x, dm, du, l_hop, delta))**2)

    select_indexes = np.unique(np.geomspace(1,len(X), len(X)).astype(int))-1
    X = X[select_indexes]
    Y = Y[select_indexes]

    eq_9_obj = lambda coeffs: eq_9_obj_raw(X, Y, *coeffs)
    res_eq_9s = []

    for _ in range(99):        
        x0=[np.random.uniform(1000, 100000), np.random.uniform(1000, 100000), np.random.uniform(1, 1000), np.random.uniform(1, 100)]
        res_eq_9 = minimize(eq_9_obj, x0=x0, bounds=[(1000, None), (1000, None), (1, None), (1, None)], constraints=[LinearConstraint([-5,1,0,0], lb=0, ub=np.inf)])
        res_eq_9s.append(res_eq_9)

    return min(res_eq_9s, key=lambda r: r.fun)

def anomalous_fitting(X,Y):
    def eq_a_obj_raw(x, y, dbetha, betha, delta): return np.sum((y - equation_anomalous(x, dbetha, betha, delta))**2)

    select_indexes = np.unique(np.geomspace(1,len(X), len(X)).astype(int))-1
    X = X[select_indexes]
    Y = Y[select_indexes]

    eq_a_obj = lambda coeffs: eq_a_obj_raw(X, Y, *coeffs)
    res_eq_as = []

    for _ in range(99):        
        x0=[np.random.uniform(1, 100000), np.random.uniform(0, 2), np.random.uniform(1,100)]
        res_eq_9 = minimize(eq_a_obj, x0=x0, bounds=[(1, None), (0, 2), (1, None)])
        res_eq_as.append(res_eq_9)

    return min(res_eq_as, key=lambda r: r.fun)

datasets = [
    'Control', 'CholesterolPEGKK114'
]

DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)
for dataset in datasets:
    msd_results = []
    classification = {'hop': [], 'free': [], 'anomalous': []}
    alphas = {'hop': [], 'free': [], 'anomalous': []}
    i = 0
    for t in tqdm.tqdm(Trajectory._get_collection().find({'info.immobile':False, 'info.dataset': dataset}, {'_id':1, 'x':1, 'y':1, 't':1,'info.analysis.betha':1})):
        if i > 100:
           break 
        trajectory = Trajectory(
            x=np.array(t['x'])*1000,
            y=np.array(t['y'])*1000,
            t=np.append(0,np.cumsum(np.round(np.diff(t['t']),4))),
            noisy=True
        )

        if 'betha' not in t['info']['analysis']:
            continue

        segments = [trajectory.build_noisy_subtrajectory_from_range(i, i+SEGMENT_LENGTH) for i in range(0,trajectory.length, SEGMENT_LENGTH)]

        for segment in segments:
            if segment.length != SEGMENT_LENGTH:
                continue

            t_msd, msd = segment.calculate_msd_curve(bin_width=DELTA_T)
            msd = msd[:int(len(msd)*0.20)]
            Y = np.array(msd)
            msd_results.append(Y)
            X = np.array(range(1,len(Y)+1))
            T = t_msd[:len(Y)]
            n = len(Y)
            print(n)
            res_eq_4 = free_fitting(X,Y)
            res_eq_9 = hop_fitting(X,Y)
            res_eq_a = anomalous_fitting(X,Y)

            BIC_4 = n * np.log(res_eq_4.fun/n) + 2 * np.log(n)
            BIC_9 = n * np.log(res_eq_9.fun/n) + 4 * np.log(n)
            BIC_A = n * np.log(res_eq_a.fun/n) + 3 * np.log(n)

            MIN_BIC = min([BIC_4, BIC_9, BIC_A])

            if MIN_BIC == BIC_4:
                label = 'free'
            elif MIN_BIC == BIC_9:
                label = 'hop'

                plt.title(f"{label}, {res_eq_9.x[0]}, {res_eq_9.x[1]}")
                plt.plot(X*DELTA_T,Y,color='black')
                plt.plot(X*DELTA_T,equation_free(X,*res_eq_4.x), color='blue')
                plt.plot(X*DELTA_T,equation_hop(X,*res_eq_9.x), color='green')
                plt.plot(X*DELTA_T,equation_anomalous(X,*res_eq_a.x), color='red')

                X_DU = X*DELTA_T
                Y_DU = X_DU*res_eq_9.x[1]
                Y_DM = X_DU*res_eq_9.x[0]

                offset = Y[-1] - Y_DU[-1]
                Y_DU += offset

                offset = Y[0] - Y_DM[0]
                Y_DM += offset

                y_lim = plt.ylim()

                plt.plot(X_DU, Y_DU, color='gray')
                plt.plot(X_DU, Y_DM, color='yellow')

                plt.ylim(y_lim)
                plt.show()
                trajectory.animate_plot()
            elif MIN_BIC == BIC_A:
                label = 'anomalous'

            classification[label].append(Y)
            alphas[label].append(t['info']['analysis']['betha'])
        i+=1

    print('HOP:', len(classification['hop'])/(len(classification['hop'])+len(classification['free'])))
    print('FREE:', len(classification['free'])/(len(classification['hop'])+len(classification['free'])))

    print('HOP:', np.mean(alphas['hop']), sem(alphas['hop']))
    print('FREE:', np.mean(alphas['free']), sem(alphas['free']))

    min_hop_msd = np.min([len(y) for y in classification['hop']])
    classification['hop'] = [msd[:min_hop_msd] for msd in classification['hop']]

    min_free_msd = np.min([len(y) for y in classification['free']])
    classification['free'] = [msd[:min_free_msd] for msd in classification['free']]

    hop_mean_msd = np.mean(classification['hop'], axis=0)
    free_mean_msd = np.mean(classification['free'], axis=0)

    X_HOP = np.array(range(1,len(hop_mean_msd)+1))
    X_FREE = np.array(range(1,len(free_mean_msd)+1))

    dm, du, delta, lhop = hop_fitting(X_HOP, hop_mean_msd).x
    d, delta_f = free_fitting(X_FREE, free_mean_msd).x

    yerr_hop = np.std(classification['hop'], axis=0)/np.sqrt(len(classification['hop'][0]))
    yerr_free = np.std(classification['free'], axis=0)/np.sqrt(len(classification['free'][0]))

    f, axarr = plt.subplots(2)
    axarr[0].errorbar((X_FREE*DELTA_T)[:25], free_mean_msd[:25], yerr=yerr_free[:25], color='blue', linewidth=1, fmt ='o')
    axarr[0].plot((X_FREE*DELTA_T)[:25], equation_free(X_FREE, d, delta_f)[:25], color='black', linewidth=3)
    #axarr[0].set_title('Free diffusion')
    axarr[0].set_ylabel('MSD [nm2/s]')

    axarr[1].errorbar((X_HOP*DELTA_T)[:25], hop_mean_msd[:25], yerr=yerr_hop[:25], color='orange', linewidth=1, fmt ='o')
    axarr[1].plot((X_HOP*DELTA_T)[:25], equation_hop(X_HOP, dm, du, delta, lhop)[:25], color='black', linewidth=3)
    #axarr[1].set_title('Hop diffusion')
    axarr[1].set_xlabel('Time lag [s]')
    axarr[1].set_ylabel('MSD [nm2/s]')

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
    du = ('{:,}'.format(int(du))) 
    dm = ('{:,}'.format(int(dm))) 
    d = ('{:,}'.format(int(d))) 

    axarr[0].text(15*DELTA_T, 500, '$D ='+d+' nm^2 s^-1$', fontsize = 10)
    axarr[1].text(15*DELTA_T, 1500, '$D_{\mu} ='+du+' nm^2 s^-1$', fontsize = 10)
    axarr[1].text(15*DELTA_T, 500, '$D_{M} ='+dm+' nm^2 s^-1$', fontsize = 10)

    plt.savefig(f'rickert_{dataset}.png')
    exit()
DatabaseHandler.disconnect()