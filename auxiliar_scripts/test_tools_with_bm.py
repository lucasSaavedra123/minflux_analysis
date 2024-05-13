from utils import irregular_brownian_motion
from Trajectory import Trajectory
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import matplotlib.pyplot as plt
"""
LIMITS = [10,25,50,100,200,500]

TRAJECTORY_LENGTHS = 1000
NUMBER_OF_TRAJECTORIES = 10

trajectories = []
errors_for_no_reconstruction = {}

for limit in LIMITS:
    errors_for_no_reconstruction[limit] = []

for _ in tqdm.tqdm(range(NUMBER_OF_TRAJECTORIES)):
    alpha = np.random.uniform(0.1,1.9,size=1)[0]
    simulation_result = irregular_fractional_brownian_motion(TRAJECTORY_LENGTHS, alpha, 2)

    trajectories.append(Trajectory(
        x=simulation_result[0][0,:],
        y=simulation_result[0][1,:],
        t = simulation_result[1],
        exponent=alpha
    ))

    for limit in LIMITS:
        _, _, new_alpha, _, _ = trajectories[-1].temporal_average_mean_squared_displacement(with_noise=False, log_log_fit_limit=limit)
        errors_for_no_reconstruction[limit].append(np.abs(new_alpha-alpha))

for limit in LIMITS:
    errors_for_no_reconstruction[limit] = np.mean(errors_for_no_reconstruction[limit])

plt.plot(LIMITS, list(errors_for_no_reconstruction.values()), label='Original', marker='X')

for reconstruction_time in [10e-6]:#, 50e-6,100e-6, 300e-6, 500e-6, 700e-6, 1000e-6]:
    error = {}
    for limit in LIMITS:
        error[limit] = []

        for t in trajectories:
            gt_alpha = t.anomalous_exponent

            reconstructed_trajectory = t.reconstructed_trajectory(reconstruction_time, with_noise=False)

            _, _, alpha, _, _ = reconstructed_trajectory.temporal_average_mean_squared_displacement(with_noise=False, log_log_fit_limit=limit)
            error[limit].append(np.abs(gt_alpha - alpha))

        error[limit] = np.mean(error[limit])

    plt.plot(list(error.keys()), list(error.values()), label=f'{reconstruction_time*1e6}', marker='X')

plt.plot()
plt.legend()
plt.show()
"""
"""
TRAJECTORY_LENGTHS = 1000
NUMBER_OF_TRAJECTORIES = 1000

trajectories = []
for _ in tqdm.tqdm(range(NUMBER_OF_TRAJECTORIES)):
    alpha = np.random.uniform(0.1,1.9,size=1)[0]
    simulation_result = irregular_fractional_brownian_motion(TRAJECTORY_LENGTHS, alpha, 2)

    trajectories.append(Trajectory(
        x=simulation_result[0][0,:],
        y=simulation_result[0][1,:],
        t = simulation_result[1],
        exponent=alpha
    ))

error = {
    'without_reconstruction': {},
    'with_reconstruction': {}
}


for limit in [10,25,50,100,200,500]:
    error['without_reconstruction'][limit] = []
    error['with_reconstruction'][limit] = []

    for t in trajectories:
        gt_alpha = t.anomalous_exponent

        reconstructed_trajectory = t.reconstructed_trajectory(500e-6, with_noise=False)

        _, _, alpha, _, _ = t.temporal_average_mean_squared_displacement(with_noise=False, log_log_fit_limit=limit)
        error['without_reconstruction'][limit].append(np.abs(gt_alpha-alpha))

        _, _, alpha, _, _ = reconstructed_trajectory.temporal_average_mean_squared_displacement(with_noise=False, log_log_fit_limit=limit)
        error['with_reconstruction'][limit].append(np.abs(gt_alpha-alpha))

    error['without_reconstruction'][limit] = np.mean(error['without_reconstruction'][limit])
    error['with_reconstruction'][limit] = np.mean(error['with_reconstruction'][limit])

plt.plot(list(error['without_reconstruction'].keys()), list(error['without_reconstruction'].values()), label='Without reconstruction', marker='X')
plt.plot(list(error['with_reconstruction'].keys()), list(error['with_reconstruction'].values()), label='With reconstruction', marker='X')
plt.legend()
plt.show()
"""

"""
TRAJECTORY_LENGTHS = [500,750,1000,1250,1500,1750,2000]
NUMBER_OF_TRAJECTORIES = 10

trajectories = {}
error = {}

for length in TRAJECTORY_LENGTHS:
    trajectories = []
    error = {}
    for _ in tqdm.tqdm(range(NUMBER_OF_TRAJECTORIES)):
        alpha = np.random.uniform(0.1,1.9,size=1)[0]
        simulation_result = irregular_fractional_brownian_motion(length, alpha, 2)

        trajectories.append(Trajectory(
            x=simulation_result[0][0,:],
            y=simulation_result[0][1,:],
            t = simulation_result[1],
            exponent=alpha
        ))

    limits_used = []

    for limit in [10,25,50,100,200]:
        if length < limit:
            break
        else:
            limits_used.append(limit)

        error[limit] = []
        error[limit] = []

        for t in trajectories:
            gt_alpha = t.anomalous_exponent

            reconstructed_trajectory = t.reconstructed_trajectory(500e-6, with_noise=False)

            _, _, alpha, _, _ = reconstructed_trajectory.temporal_average_mean_squared_displacement(with_noise=False, log_log_fit_limit=limit)
            error[limit].append(np.abs(gt_alpha-alpha))

        error[limit] = np.mean(error[limit])

    plt.plot(limits_used, list(error.values()), label=f'{length}', marker='X')

plt.legend()
plt.show()
"""

TRAJECTORY_LENGTHS = [500,750,1000,1250,1500,1750,2000]
NUMBER_OF_TRAJECTORIES = 10

trajectories = {}
error = {}

for length in TRAJECTORY_LENGTHS:
    trajectories = []
    error = {}
    for _ in tqdm.tqdm(range(NUMBER_OF_TRAJECTORIES)):
        alpha = np.random.uniform(0.1,1.9,size=1)[0]
        simulation_result = irregular_brownian_motion(length, alpha, 2)

        trajectories.append(Trajectory(
            x=simulation_result[0][0,:],
            y=simulation_result[0][1,:],
            t = simulation_result[1],
            exponent=alpha
        ))

    limits_used = []

    for limit in [10,25,50,100,200]:
        if length < limit:
            break
        else:
            limits_used.append(limit)

        error[limit] = []
        error[limit] = []

        for t in trajectories:
            gt_alpha = t.anomalous_exponent

            reconstructed_trajectory = t.reconstructed_trajectory(500e-6, with_noise=False)

            _, _, alpha, _, _ = reconstructed_trajectory.temporal_average_mean_squared_displacement(with_noise=False, log_log_fit_limit=limit)
            error[limit].append(np.abs(gt_alpha-alpha))

        error[limit] = np.mean(error[limit])

    plt.plot(limits_used, list(error.values()), label=f'{length}', marker='X')

plt.legend()
plt.show()