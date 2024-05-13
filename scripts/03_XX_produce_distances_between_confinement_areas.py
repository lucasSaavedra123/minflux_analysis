"""
XXX_XXX
"""
import tqdm
import numpy as np
from collections import defaultdict
from scipy.spatial import Delaunay
import math
import glob
import matplotlib.pyplot as plt

from DatabaseHandler import DatabaseHandler
from Trajectory import Trajectory
from CONSTANTS import *

import pointpats
from sklearn.neighbors import NearestNeighbors

DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, 'MINFLUX_DATA')

distances = defaultdict(lambda: [])

a_file = open('file_and_roi.txt')

files_rois = [i.strip().split(',') for i in a_file.readlines()]
files_rois = [(i[0],int(i[1])) for i in files_rois]
"""
for file in Trajectory.objects().distinct(field="info.file"):
    files_rois += [(file, roi) for roi in Trajectory.objects(info__file=file).distinct(field="info.roi")]
"""
for file_roi in tqdm.tqdm(files_rois):
    trajectories = Trajectory.objects(info__file=file_roi[0], info__roi=file_roi[1])

    x_centroids = []
    y_centroids = []

    for ti in [t for t in trajectories if t.length > 1]:
        for tc in ti.sub_trajectories_trajectories_from_confinement_states(use_info=True)[1]:
            x_centroids.append(np.mean(tc.get_noisy_x()))
            y_centroids.append(np.mean(tc.get_noisy_y()))
    
    coordinates = np.array(list(zip(x_centroids, y_centroids)))
    """
    simplices = Delaunay(coordinates).simplices

    list_of_edges = []

    def less_first(a, b):
        return [a,b] if a < b else [b,a]

    for simplex in simplices:
        if len(simplex) == 3:
            set_to_iterate = [[0,1],[0,2],[1,2]]
        elif len(simplex) == 4:
            set_to_iterate = [[0,1],[0,2],[1,2],[0,3],[1,3],[2,3]]
        else:
            raise Exception(f'Simplex of size {len(simplex)} are not allowed')

        for e1, e2 in set_to_iterate:
            if less_first(simplex[e1],simplex[e2]) not in list_of_edges:
                list_of_edges.append(less_first(simplex[e1],simplex[e2]))
    """
    nn_distances, _ = NearestNeighbors(n_neighbors=2, algorithm='brute').fit(coordinates).kneighbors(coordinates)
    ann = np.mean(nn_distances[:,1])

    ann_realizations = []

    for _ in range(999):
        X = np.random.uniform(coordinates[:,0].min(),coordinates[:,0].max(), size=len(coordinates))
        Y = np.random.uniform(coordinates[:,1].min(),coordinates[:,1].max(), size=len(coordinates))
        X = np.array(list(zip(X,Y)))
        nn_distances, _ = NearestNeighbors(n_neighbors=2, algorithm='brute').fit(X).kneighbors(X)
        ann_realizations.append(np.mean(nn_distances[:,1]))

    plt.scatter(X[:,0], X[:,1], color='red')
    plt.scatter(coordinates[:,0], coordinates[:,1], color='black')
    plt.show()

    q_5 = np.quantile(ann_realizations, 0.05)
    q_95 = np.quantile(ann_realizations, 0.95)

    plt.axvline(x = q_5, color = 'b')
    plt.axvline(x = q_95, color = 'b')
    plt.axvline(x = ann, color = 'r')
    plt.hist(ann_realizations)
    plt.show()

    if ann < q_5:
        distances[ti.info['dataset']].append('c')
    elif q_5 <= ann <= q_95:
        distances[ti.info['dataset']].append('r')
    elif q_95 < ann:
        distances[ti.info['dataset']].append('d')
"""
for key in distances:
    np.savetxt(f"./Results/confinement_distances_inside_roi_{key}.txt", distances[key])
"""

for key in distances:
    np.savetxt(f"./Results/confinement_distances_classification_inside_roi_{key}.txt", distances[key], fmt="%s")

import matplotlib.pyplot as plt
import pandas as pd

dataset = {
    'dataset': [],
    'mean': [],
    'std': [],
    'sem': []
}

for a in glob.glob(f"./Results/confinement_distances_inside_roi_*.txt"):
    array = np.loadtxt(a)
    dataset['dataset'].append(a)
    dataset['mean'].append(int(np.mean(array)))
    dataset['sem'].append(np.std(array, ddof=1) / np.sqrt(np.size(array)))
    dataset['std'].append(int(np.std(array)))

    #plt.hist(array, density=True, bins='auto', histtype='stepfilled', alpha=0.2, color='red')
    #plt.xlim([0,1000])
    #plt.show()

pd.DataFrame(dataset).to_csv('distances_confinement_areas.csv')

dataset = {
    'dataset': [],
    'd': [],
    'r': [],
    'c': []
}

from collections import Counter

for a in glob.glob(f"./Results/confinement_distances_classification_inside_roi_*.txt"):
    array = np.loadtxt(a, fmt="%s")
    print(Counter(array))
