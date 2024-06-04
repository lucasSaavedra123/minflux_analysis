"""
All BTX trajectories are plotted with their corresponding overlap Chol ones.
"""
from collections import defaultdict

import tqdm
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300
from DatabaseHandler import DatabaseHandler
from Trajectory import Trajectory
from CONSTANTS import *
from utils import *


DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)

CHOL_AND_BTX_DATASETS = [
    'Cholesterol and btx',
    'CK666-BTX680-CHOL',
    'BTX680-fPEG-CHOL-50-nM',
    'BTX680-fPEG-CHOL-100-nM',
]

files = [info[1] for info in extract_dataset_file_roi_file() if info[0] in CHOL_AND_BTX_DATASETS]
files = list(set(files))

for file in tqdm.tqdm(files):
    trajectories = Trajectory.objects(info__file=file)
    for index, trajectory in enumerate(trajectories):
        if trajectory.info['classified_experimental_condition'] == BTX_NOMENCLATURE and 'number_of_confinement_zones' in trajectory.info and trajectory.info['number_of_confinement_zones'] != 0 and 0.30 <= trajectory.info[f'number_of_confinement_zones_with_{CHOL_NOMENCLATURE}']/trajectory.info['number_of_confinement_zones'] <= 0.40:
            trajectory.plot_confinement_states(v_th=33, non_confinement_color='green', confinement_color='green', show=False, alpha=1, plot_confinement_convex_hull=False, color_confinement_convex_hull='#ffff00', alpha_confinement_convex_hull=0.5)
            base_polygons = []
            extra_polygons = []

            for sub_t in trajectory.sub_trajectories_trajectories_from_confinement_states(v_th=33)[1]:
                xx, yy = MultiPoint(list(zip(sub_t.get_noisy_x(), sub_t.get_noisy_y()))).convex_hull.exterior.coords.xy
                xx, yy =  xx.tolist(), yy.tolist()

                base_polygons.append(sort_vertices_anti_clockwise_and_remove_duplicates(list(zip(xx, yy))))

            for chol_trajectory_id in trajectory.info[f'{CHOL_NOMENCLATURE}_intersections']:
                aux_t = Trajectory.objects(id=chol_trajectory_id)[0]
                aux_t.plot_confinement_states(v_th=33, non_confinement_color='red', confinement_color='red', show=False, alpha=1, plot_confinement_convex_hull=False, color_confinement_convex_hull='#ff00ff', alpha_confinement_convex_hull=0.5)

                for sub_t in aux_t.sub_trajectories_trajectories_from_confinement_states(v_th=33)[1]:
                    xx, yy = MultiPoint(list(zip(sub_t.get_noisy_x(), sub_t.get_noisy_y()))).convex_hull.exterior.coords.xy
                    xx, yy =  xx.tolist(), yy.tolist()

                    extra_polygons.append(sort_vertices_anti_clockwise_and_remove_duplicates(list(zip(xx, yy))))

            for b_p in base_polygons:
                for e_p in extra_polygons:
                    polygon_intersection = intersect(b_p, e_p)
                    if len(polygon_intersection) != 0:
                        polygon_intersection = list(polygon_intersection)
                        polygon_intersection.append(polygon_intersection[0])
                        x, y = zip(*polygon_intersection)
                        #plt.plot(x, y, 'o-')
                        plt.fill(x, y, alpha=1, color='#ffff00')

            if trajectory.info['number_of_confinement_zones'] != 0:
                print(trajectory.info[f'number_of_confinement_zones_with_{CHOL_NOMENCLATURE}']/trajectory.info['number_of_confinement_zones'])

                plt.gca().set_aspect('equal')
                plt.tight_layout()
                plt.show()
            else:
                plt.clf()

DatabaseHandler.disconnect()
