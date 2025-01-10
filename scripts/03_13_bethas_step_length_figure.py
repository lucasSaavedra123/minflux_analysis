"""
ALL trajectories are analyzed.
"""
from matplotlib import pyplot as plt
import numpy as np

from DatabaseHandler import DatabaseHandler
from Trajectory import Trajectory
from CONSTANTS import *


DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)

INDIVIDUAL_DATASETS = [
    'BTX680R',
    'CK666-BTX680',
    'CholesterolPEGKK114',
    'CK666-CHOL',
]

titles = [
    'CF®680R-BTX',
    'CF®680R-BTX (with CK666)',
    'fPEG-Chol',
    'fPEG-Chol (with CK666)',
]

fig_scatter, ax_scatter = plt.subplots(1,4)
#fig_hist2d, ax_hist2d = plt.subplots(1,4)
ax_scatter[0].set_ylabel("β")
#ax_hist2d[0].set_ylabel("β")

for dataset_i, dataset in enumerate(INDIVIDUAL_DATASETS):
    trajectories = Trajectory.objects(info__dataset=dataset)
    trajectories = trajectories

    step_length = []
    bethas = []

    for trajectory in trajectories:
        if 'analysis' not in trajectory.info:
            continue
        confinements = trajectory.sub_trajectories_trajectories_from_confinement_states(v_th=33, transition_fix_threshold=3, use_info=True)[1]
        for confinement_i, confinement in enumerate(confinements):
            positions = np.zeros((confinement.length, 2))
            positions[:,0] = confinement.get_noisy_x()
            positions[:,1] = confinement.get_noisy_y()
            positions_diff = np.diff(positions, axis=0)
            if trajectory.info['analysis']['confinement-betha'][confinement_i] is not None:
                step_length.append(np.mean(np.linalg.norm(positions_diff, axis=1)))
                bethas.append(trajectory.info['analysis']['confinement-betha'][confinement_i])

    ax_scatter[dataset_i].set_title(titles[dataset_i])
    ax_scatter[dataset_i].scatter(np.array(step_length)*1000, bethas, color='red', linewidths=0.25, edgecolors='black')
    ax_scatter[dataset_i].set_xlabel("Step length (nm)")
    ax_scatter[dataset_i].set_ylim([0,2])
    ax_scatter[dataset_i].set_xlim([5,30])
    ax_scatter[dataset_i].set_aspect(1./ax_scatter[dataset_i].get_data_ratio())#.set_aspect('equal', adjustable='box')

    if dataset_i > 0:
        ax_scatter[dataset_i].get_yaxis().set_ticks([])
    """
    ax_hist2d[dataset_i].set_title(titles[dataset_i])
    ax_hist2d[dataset_i].hist2d(np.array(step_length)*1000, bethas, 30, range=[[5,30],[0,2]])
    ax_hist2d[dataset_i].set_xlabel("Step length (nm)")
    ax_hist2d[dataset_i].set_ylim([0,2])
    ax_hist2d[dataset_i].set_xlim([5,30])
    ax_hist2d[dataset_i].set_aspect(1./ax_hist2d[dataset_i].get_data_ratio())#.set_aspect('equal', adjustable='box')

    if dataset_i > 0:
        ax_hist2d[dataset_i].get_yaxis().set_ticks([])
    """

mng = plt.get_current_fig_manager()
mng.full_screen_toggle()
#plt.show()
fig_scatter.savefig('scatter.svg')
#fig_hist2d.savefig('hist.svg')


DatabaseHandler.disconnect()
