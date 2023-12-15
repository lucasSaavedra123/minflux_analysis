from collections import defaultdict

import tqdm
import numpy as np

from DatabaseHandler import DatabaseHandler
from Trajectory import Trajectory
from CONSTANTS import *
from utils import both_trajectories_intersect
import matplotlib.pyplot as plt

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
    trajectories_by_condition = defaultdict(lambda: [])

    for trajectory in trajectories:
        if trajectory.info['classified_experimental_condition'] == BTX_NOMENCLATURE:
            states = trajectory.info[f'{CHOL_NOMENCLATURE}_single_intersections']
        else:
            states = trajectory.info[f'{BTX_NOMENCLATURE}_single_intersections']

        state_to_color = {1:'red', 0:'black'}
        states_as_color = np.vectorize(state_to_color.get)(states)

        x,y = trajectory.get_noisy_x().tolist(), trajectory.get_noisy_y().tolist()

        for i,(x1, x2, y1,y2) in enumerate(zip(x, x[1:], y, y[1:])):
            plt.plot([x1, x2], [y1, y2], states_as_color[i])

        plt.show()