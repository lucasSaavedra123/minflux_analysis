import os
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tqdm

from Trajectory import Trajectory
from utils import *
from CONSTANTS import *
from DatabaseHandler import DatabaseHandler

DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)
"""
for trajectory in Trajectory.objects():
    if not('analysis' in trajectory.info and 'step_result' in trajectory.info['analysis'] and not trajectory.info['immobile']):
        continue
    
    cm = plt.cm.get_cmap('copper_r')

    X = np.zeros((trajectory.length,2))
    X[:,0] = trajectory.get_noisy_x()
    X[:,1] = trajectory.get_noisy_y()

    DX = np.zeros((trajectory.length-1,2))
    DX[:,0] = X[:-1,0]
    DX[:,1] = X[1:,0]
    DX = (DX[:,0] - DX[:,1]) ** 2

    DY = np.zeros((trajectory.length-1,2))
    DY[:,0] = X[:-1,1]
    DY[:,1] = X[1:,1]
    DY = (DY[:,0] - DY[:,1]) ** 2

    D = np.sqrt(DX + DY)
    T = np.diff(trajectory.get_time())

    V = D/T
    V = np.insert(V,0,0)

    D = np.insert(D,0,0)

    plt.scatter(trajectory.get_noisy_x(), trajectory.get_noisy_y(), c=trajectory.info['analysis']['step_result'], s=35, cmap=cm)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel(r'$D ({\mu m}^{2}/{s})$')
    plt.xlabel(r'X [$\mu m$]')
    plt.ylabel(r'Y [$\mu m$]')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()

    plt.plot(trajectory.get_noisy_x(), trajectory.get_noisy_y(), c='grey', zorder=-99, linewidth='1')
    plt.show()
"""

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
            plt.scatter(trajectory.get_noisy_x(), trajectory.get_noisy_y(), c=trajectory.info['analysis']['step_result'], s=35, cmap=plt.cm.get_cmap('summer_r'))
            for chol_trajectory_id in trajectory.info[f'{CHOL_NOMENCLATURE}_intersections']:
                chol_trajectory = Trajectory.objects(id=chol_trajectory_id)[0]
                plt.scatter(chol_trajectory.get_noisy_x(), chol_trajectory.get_noisy_y(), c=chol_trajectory.info['analysis']['step_result'], s=35, cmap=plt.cm.get_cmap('copper_r'))

        plt.show()

DatabaseHandler.disconnect()
