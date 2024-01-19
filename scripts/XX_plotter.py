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

    x = trajectory.get_noisy_x()
    y = trajectory.get_noisy_y()

    x = (x - np.mean(x)) + (offset_dictionary[index][0] * offset)
    y = (y - np.mean(y)) + (offset_dictionary[index][1] * offset)

    state_to_color = {1:'green', 0:'black'}
    states_as_color = np.vectorize(state_to_color.get)(trajectory.confinement_states(v_th=33, window_size=3))

    for i,(x1, x2, y1,y2) in enumerate(zip(x, x[1:], y, y[1:])):
        plt.plot([x1, x2], [y1, y2], states_as_color[i])

plt.xlabel('X [μm]', fontname="Arial", fontsize=20)
plt.ylabel('Y [μm]', fontname="Arial", fontsize=20)

plt.xticks(fontname="Arial", fontsize=20)
plt.yticks(fontname="Arial", fontsize=20)

plt.axis('square')
plt.show()

DatabaseHandler.disconnect()
