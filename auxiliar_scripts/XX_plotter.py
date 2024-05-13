import random

import numpy as np
import matplotlib.pyplot as plt

from Trajectory import Trajectory
from DatabaseHandler import DatabaseHandler
from CONSTANTS import IP_ADDRESS, DATASET_TO_DELTA_T

DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, 'MINFLUX_DATA')

"""
trajectories = [trajectory for trajectory in Trajectory.objects(info__file='230818-132720_mbm test.txt') if trajectory.length > 1]

for trajectory in trajectories:
    color = 'black' if trajectory.info['immobile'] else 'red'

    plt.plot(
        trajectory.get_noisy_x(),
        trajectory.get_noisy_y(),
        color=color
    )

#plt.tight_layout()
plt.xlabel('X [μm]', fontname="Arial", fontsize=20)
plt.ylabel('Y [μm]', fontname="Arial", fontsize=20)

plt.xticks(fontname="Arial", fontsize=20)
plt.yticks(fontname="Arial", fontsize=20)

plt.axis('square')
plt.show()
"""

trajectory_results = list(Trajectory._get_collection().find({'info.immobile':False}, {'_id':1}))

selected_trajectory_results = random.sample(trajectory_results, 4)
"""
for trajectory_result in selected_trajectory_results:
    trajectory = Trajectory.objects(id=str(trajectory_result['_id']))[0]

    color = 'black' if trajectory.info['immobile'] else 'red'

    plt.plot(
        trajectory.get_noisy_x(),
        trajectory.get_noisy_y(),
        color='red'
    )

    reconstructed_trajectory = trajectory.reconstructed_trajectory(DATASET_TO_DELTA_T[trajectory.info['dataset']])

    plt.plot(
        reconstructed_trajectory.get_noisy_x(),
        reconstructed_trajectory.get_noisy_y(),
        color='blue'
    )

    #plt.tight_layout()
    plt.xlabel('X [μm]', fontname="Arial", fontsize=20)
    plt.ylabel('Y [μm]', fontname="Arial", fontsize=20)

    plt.xticks(fontname="Arial", fontsize=20)
    plt.yticks(fontname="Arial", fontsize=20)

    plt.axis('square')
    plt.show()
"""
selected_trajectory_results = random.sample(trajectory_results, 7)

offset_dictionary = {
    6: np.array([0,0]),
    1: np.array([0,1]),
    2: np.array([1,0]),
    3: np.array([1,1]),
    4: np.array([-1,0]),
    5: np.array([0,-1]),
    0: np.array([-1,-1]),
}

offset = 0.5

for index, trajectory_result in enumerate(selected_trajectory_results):
    trajectory = Trajectory.objects(id=str(trajectory_result['_id']))[0]
    trajectory.plot_confinement_states(v_th=33, show=False)
    plt.axis('square')
    plt.savefig(str(index)+".tiff", dpi=300)
    plt.clf()

"""
plt.xlabel('X [μm]', fontname="Arial", fontsize=20)
plt.ylabel('Y [μm]', fontname="Arial", fontsize=20)

plt.xticks(fontname="Arial", fontsize=20)
plt.yticks(fontname="Arial", fontsize=20)

plt.axis('square')
plt.show()
"""
DatabaseHandler.disconnect()
