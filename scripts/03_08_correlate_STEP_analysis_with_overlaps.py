"""
All BTX trajectories are plotted with their corresponding overlap Chol ones.
"""
from collections import defaultdict

import tqdm
import matplotlib.pyplot as plt

from DatabaseHandler import DatabaseHandler
from Trajectory import Trajectory
from CONSTANTS import *
from utils import both_trajectories_intersect


DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)

#files = Trajectory.objects(info__dataset='Cholesterol and btx').distinct(field='info.file')

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


for file in tqdm.tqdm(files):
    trajectories = Trajectory.objects(info__file=file)
    for index, trajectory in enumerate(trajectories):
        if trajectory.info['classified_experimental_condition'] == BTX_NOMENCLATURE:
            trajectory.plot_confinement_states(v_th=33, non_confinement_color='green', confinement_color='green', show=False, alpha=0.75, plot_confinement_convex_hull=True, color_confinement_convex_hull='green')

            for chol_trajectory_id in trajectory.info[f'{CHOL_NOMENCLATURE}_intersections']:
                Trajectory.objects(id=chol_trajectory_id)[0].plot_confinement_states(v_th=33, non_confinement_color='red', confinement_color='red', show=False, alpha=0.75, plot_confinement_convex_hull=True, color_confinement_convex_hull='red')

            if trajectory.info['number_of_confinement_zones'] != 0:
                print(trajectory.info[f'number_of_confinement_zones_with_{CHOL_NOMENCLATURE}']/trajectory.info['number_of_confinement_zones'])

                plt.gca().set_aspect('equal')
                plt.tight_layout()
                plt.show()
            else:
                plt.clf()

DatabaseHandler.disconnect()
