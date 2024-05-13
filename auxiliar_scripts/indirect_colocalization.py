from collections import defaultdict
import os
import pandas as pd
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import MultiPoint, Point
from scipy.spatial import Delaunay

from DatabaseHandler import DatabaseHandler
from Trajectory import Trajectory
from CONSTANTS import *
#from spit.colocalize import colocalize_from_locs
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)

files = [
    '231013-105211_mbm test.txt',
    '231013-105628_mbm test-pow8pc.txt',
    '231013-110430_mbm test-pow8pc.txt',
    '231013-111321_mbm test-pow8pc.txt',
    '231013-111726_mbm test-pow8pc.txt',
    '231013-112242_mbm test-pow8pc.txt',
    '231013-112652_mbm test-pow8pc.txt',
    '231013-113251_mbm test-pow8pc.txt',
    '231013-113638_mbm test-pow8pc.txt',
    '231013-124040_mbm test.txt',
    '231013-124511_mbm test.txt',
    '231013-125044_mbm test.txt',
    '231013-125411_mbm test.txt',
    '231013-125818_mbm test.txt',
    '231013-130259_mbm test.txt',
    '231013-130748_mbm test.txt',
    '231013-131100_mbm test.txt',
    '231013-131615_mbm test.txt',
    '231013-131935_mbm test.txt',
    '231013-132310_mbm test.txt',
    '231013-132703_mbm test.txt',
    '231013-153332_mbm test.txt',
    '231013-153631_mbm test.txt',
    '231013-154043_mbm test.txt',
    '231013-154400_mbm test.txt',
    '231013-154702_mbm test.txt',
    '231013-154913_mbm test.txt',
    '231013-155220_mbm test.txt',
    '231013-155616_mbm test.txt',
    '231013-155959_mbm test.txt',
    '231013-160351_mbm test.txt',
    '231013-160951_mbm test.txt',
    '231013-161302_mbm test.txt',
    '231013-161554_mbm test.txt',
    '231013-162155_mbm test.txt',
    '231013-162602_mbm test.txt',
    '231013-162934_mbm test.txt',
    '231013-163124_mbm test.txt',
    '231013-163414_mbm test.txt',
    '231013-163548_mbm test.txt'
]


def se_sobrelapan(rango1, rango2):
    # Verificar si el rango1 se encuentra completamente a la izquierda del rango2
    if rango1[1] < rango2[0]:
        return False
    # Verificar si el rango1 se encuentra completamente a la derecha del rango2
    elif rango1[0] > rango2[1]:
        return False
    # Si no se cumple ninguna de las condiciones anteriores, los rangos se sobrelapan
    else:
        return True

np.random.shuffle(files)

for file_name in tqdm.tqdm(files):
    #fig, axs = plt.subplots(nrows=2)

    """
    data = {
        'channel': [],
        'x': [],
        'y': [],
        't': [],
        'id': [],
    }
    """
    trajectories_divided = {'btx':[],'chol':[]}
    chol_geometries = []
    btx_geometries = []

    #ax = plt.figure().add_subplot(projection='3d')

    #Discriminate trajectories and keep confined zones of Chol
    for t in tqdm.tqdm(Trajectory.objects(info__file=file_name)):
        label = 'chol' if np.mean(t.info['dcr']) > TDCR_THRESHOLD else 'btx'
        if 'immobile' in t.info and not t.info['immobile']:
            trajectories_divided[label].append(t)
            """
            if label == 'chol':
                axs[1].axvspan(t.get_time()[0],t.get_time()[-1],0,1, color='orange', alpha=0.5)
            else:
                axs[0].axvspan(t.get_time()[0],t.get_time()[-1],0,1, color='blue', alpha=0.5)
            """
            """
            data['channel'] += t.length * [label]
            data['id'] += t.length * [t.info['id']]
            data['x'] += t.get_noisy_x().tolist()
            data['y'] += t.get_noisy_y().tolist()
            data['t'] += t.get_time().tolist()
            """
            """
            if label == 'chol':
                for subt_chol in t.sub_trajectories_trajectories_from_confinement_states(v_th=33, use_info=True)[1]:
                    #t.info['analysis'] = {}
                    #t.info['analysis']['confinement-states'] = t.confinement_states(return_intervals=False, v_th=33)
                    chol_confined_zones.append((subt_chol, MultiPoint([p for p in zip(subt_chol.get_noisy_x(), subt_chol.get_noisy_y())]).convex_hull))
            """

            if label == 'chol':
                chol_geometries.append((t, Delaunay(np.array(list(zip(t.get_noisy_x(), t.get_noisy_y(), t.get_time()))))))
                #ax.plot(t.get_noisy_x(), t.get_time(), t.get_noisy_y(), color='orange')
            else:
                btx_geometries.append((t, np.array(list(zip(t.get_noisy_x(), t.get_noisy_y(), t.get_time())))))
                #ax.plot(t.get_noisy_x(), t.get_time(), t.get_noisy_y(), color='blue')

    def in_hull(p, hull):
        """
        Test if points in `p` are in `hull`

        `p` should be a `NxK` coordinates of `N` points in `K` dimensions
        `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the 
        coordinates of `M` points in `K`dimensions for which Delaunay triangulation
        will be computed
        """
        if not isinstance(hull,Delaunay):
            hull = Delaunay(hull)

        return hull.find_simplex(p)>=0
    #plt.show()
    """
    #print("Chol", 1/np.min(trajectories_divided['chol']))
    #print("btx", 1/np.min(trajectories_divided['btx']))
    #plt.hist(trajectories_divided['chol'], color='orange', alpha=0.5)
    #plt.hist(trajectories_divided['btx'], color='blue', alpha=0.5)
    #plt.show()
    """

    for btx in tqdm.tqdm(btx_geometries):
        for chol in chol_geometries:
            #if any([chol[1].contains(Point(btx[0].get_noisy_x()[index_btx], btx[0].get_noisy_y()[index_btx], btx[0].get_time()[index_btx])) for index_btx in range(btx[0].length)]):
            if any(in_hull(btx[1], chol[1])):
                ax = plt.figure().add_subplot(projection='3d')
                ax.plot(btx[0].get_noisy_x(), btx[0].get_time(), btx[0].get_noisy_y(), color='blue')
                ax.plot(chol[0].get_noisy_x(), chol[0].get_time(), chol[0].get_noisy_y(), color='orange')
                plt.show()

    """
    for btx in trajectories_divided['btx']:
        for j in chol_confined_zones:
            index_btx = 0

            if j[1].contains(Point(btx.get_noisy_x()[index_btx], btx.get_noisy_y()[index_btx])):
                t_0 = btx.get_time()[index_btx]
                index_btx += 1

                while index_btx < btx.length and j[1].contains(Point(btx.get_noisy_x()[index_btx], btx.get_noisy_y()[index_btx])):
                    index_btx += 1

                t_1 = btx.get_time()[index_btx-1]
    
                t = t_1 - t_0
                if se_sobrelapan([t_0, t_1], [j[0].get_time()[0], j[0].get_time()[-1]]):
                    plt.plot(np.linspace(t_0, t_1, 100), [1]*100)
                    plt.plot(np.linspace(j[0].get_time()[0], j[0].get_time()[-1], 100), [2]*100)
                    plt.title(t)
                    plt.show()
    """
DatabaseHandler.disconnect()