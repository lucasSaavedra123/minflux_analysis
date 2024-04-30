
import tqdm
import numpy as np
from scipy.optimize import minimize
from Trajectory import Trajectory
from DatabaseHandler import DatabaseHandler
from CONSTANTS import *

import matplotlib.pyplot as plt

DELTA_T = 0.001
DIMENSION = 2
R = 1
L = 1

def equation_free(x, D, LOCALIZATION_PRECISION):
    TERM_1 = 2*DIMENSION*D*DELTA_T*(x-(2*R))
    TERM_2 = 2*DIMENSION*(LOCALIZATION_PRECISION**2)
    return TERM_1 + TERM_2

def equation_hop(x, DM, DU, LOCALIZATION_PRECISION, L_HOP):
    TERM_1_1_1 = (DU-DM)/DU
    TERM_1_1_2 = (0.26*(L_HOP**2))/(2*DIMENSION*x*DELTA_T)
    TERM_1_1_3 = 1 - (np.exp(-((2*DIMENSION*x*DELTA_T*DU)/(0.52*(L**2)))))

    TERM_1_1 = DM + (TERM_1_1_1*TERM_1_1_2*TERM_1_1_3)

    TERM_1_2 = DELTA_T * (x-(2*R))
    TERM_1_3 = 2*DIMENSION
    TERM_1 = TERM_1_1 * TERM_1_2 * TERM_1_3

    TERM_2 = 2*DIMENSION*(LOCALIZATION_PRECISION**2)
    return TERM_1 + TERM_2

def free_fitting(X,Y):
    def eq_4_obj_raw(x, y, d, delta): return np.sum((y - equation_free(x, d, delta))**2)

    eq_4_obj = lambda coeffs: eq_4_obj_raw(X, Y, *coeffs)
    res_eq_4s = []

    for _ in range(99):        
        x0=[np.random.uniform(100, 100000), np.random.uniform(1, 100)]
        res_eq_4 = minimize(eq_4_obj, x0=x0, bounds=[(0, None), (0, None)])
        res_eq_4s.append(res_eq_4)

    return min(res_eq_4s, key=lambda r: r.fun)

def hop_fitting(X,Y):
    def eq_9_obj_raw(x, y, dm, du, delta, l_hop): return np.sum((y - equation_hop(x, dm, du, delta, l_hop))**2)

    eq_9_obj = lambda coeffs: eq_9_obj_raw(X, Y, *coeffs)
    res_eq_9s = []

    for _ in range(99):        
        x0=[np.random.uniform(100, 100000), np.random.uniform(100, 100000), np.random.uniform(1, 100), np.random.uniform(1, 1000)]
        res_eq_9 = minimize(eq_9_obj, x0=x0, bounds=[(0, None), (0, None), (0, None), (0, None)], constraints=[{'type':'eq', 'fun': lambda t: (t[1]/t[0]) - 5}])
        res_eq_9s.append(res_eq_9)

    return min(res_eq_9s, key=lambda r: r.fun)

msd_results = []

classification = {'hop': [], 'free': []}

DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)
i = 0
for t in tqdm.tqdm(Trajectory._get_collection().find({'info.immobile':False, 'info.dataset': 'Control'}, {'_id':1, 'x':1, 'y':1, 't':1})):
    if i > 100:
        break

    trajectory = Trajectory(
        x=np.array(t['x'])*1000,
        y=np.array(t['y'])*1000,
        t=np.append(0,np.cumsum(np.round(np.diff(t['t']),4))),
        noisy=True
    )

    if trajectory.length < 100:
        continue
    else:
        i+=1

    _, msd = trajectory.calculate_msd_curve(bin_width=DELTA_T)
    msd = msd[:25]
    Y = np.array(msd)
    msd_results.append(Y)
    X = np.array(range(1,len(Y)+1))

    n = len(Y)
    
    res_eq_4 = free_fitting(X,Y)
    res_eq_9 = hop_fitting(X,Y)

    BIC_4 = n * np.log(res_eq_4.fun/n) + 2 * np.log(n)
    BIC_9 = n * np.log(res_eq_9.fun/n) + 4 * np.log(n)
    
    label = 'hop' if BIC_9 < BIC_4 else 'free'

    #plt.title(label)
    #plt.plot(X,Y,color='black')
    #plt.plot(X,equation_free(X,*res_eq_4.x), color='blue')
    #plt.plot(X,equation_hop(X,*res_eq_9.x), color='green')
    #plt.show()
    #print(label)
    #trajectory.animate_plot()

    classification[label].append(Y)

print('HOP:', len(classification['hop'])/(len(classification['hop'])+len(classification['free'])))
print('FREE:', len(classification['free'])/(len(classification['hop'])+len(classification['free'])))

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
axarr[0].errorbar(X_FREE[:25]*DELTA_T, free_mean_msd[:25], yerr=yerr_free[:25], color='blue', linewidth=1, fmt ='o')
axarr[0].plot(X_FREE[:25]*DELTA_T, equation_free(X_FREE[:25], d, delta_f), color='black', linewidth=3)
axarr[0].set_title('Free diffusion')
axarr[0].set_ylabel('MSD [nm2/s]')

axarr[1].errorbar(X_HOP[:25]*DELTA_T, hop_mean_msd[:25], yerr=yerr_hop[:25], color='orange', linewidth=1, fmt ='o')
axarr[1].plot(X_HOP[:25]*DELTA_T, equation_hop(X_HOP[:25], dm, du, delta, lhop), color='black', linewidth=3)
axarr[1].set_title('Hop diffusion')
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

plt.show()

DatabaseHandler.disconnect()