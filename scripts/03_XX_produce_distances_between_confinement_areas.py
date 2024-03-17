"""
XXX_XXX
"""
import tqdm
import numpy as np
from collections import defaultdict
from scipy.spatial import Delaunay
import math
import glob

from DatabaseHandler import DatabaseHandler
from Trajectory import Trajectory
from CONSTANTS import *


DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, 'MINFLUX_DATA')

distances = defaultdict(lambda: [])
files_rois = []
"""
for file in Trajectory.objects().distinct(field="info.file"):
    files_rois += [(file, roi) for roi in Trajectory.objects(info__file=file).distinct(field="info.roi")]

for file_roi in tqdm.tqdm(files_rois):
    trajectories = Trajectory.objects(info__file=file_roi[0], info__roi=file_roi[1])

    x_centroids = []
    y_centroids = []

    for ti in [t for t in trajectories if t.length > 1]:
        for tc in ti.sub_trajectories_trajectories_from_confinement_states(use_info=True)[1]:
            x_centroids.append(np.mean(tc.get_noisy_x()))
            y_centroids.append(np.mean(tc.get_noisy_y()))

    coordinates = list(zip(x_centroids, y_centroids))

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

    for edge in list_of_edges:
        distances[ti.info['dataset']].append(math.dist(coordinates[edge[0]], coordinates[edge[1]]) * 1000)
for key in distances:
    np.savetxt(f"./Results/confinement_distances_inside_roi_{key}.txt", distances[key])
"""

import matplotlib.pyplot as plt

for a in glob.glob(f"./Results/confinement_distances_inside_roi_*.txt"):
    array = np.loadtxt(a)
    print(a, np.mean(array), np.std(array, ddof=1) / np.sqrt(np.size(array)))
    plt.hist(array, bins=1000)

plt.show()