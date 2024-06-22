"""
Upon intersection measure, we measure which
BTX confinement zones overlap with those
of Chol confinement.
"""
import os
import ray
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from DatabaseHandler import DatabaseHandler
from Trajectory import Trajectory
from CONSTANTS import *
from utils import measure_overlap, extract_dataset_file_roi_file, transform_trajectories_with_confinement_states_from_mongo_to_dataframe, measure_overlap_with_iou
from andi_datasets.models_phenom import models_phenom

PIXEL = 0.100#um
def andi_datasets_to_trajectories(trajs, labels, particle_type=None):
    trajectories = []
    for i in range(trajs.shape[1]):
        trajectories.append(Trajectory(
            x=trajs[:,i,0]*PIXEL,
            y=trajs[:,i,1]*PIXEL,
            t=list(range(len(trajs[:,i,0]))),
            noisy=True,
            info={'analysis': {'confinement-states': (labels[:,i,-1]!=2).astype(int)}, 'classified_experimental_condition': particle_type, 'trajectory_id': str(i)+'_'+particle_type}
        ))
    return trajectories

ray.init()
@ray.remote
def parallel_get_random_value_with_iou(ROI, L, D, mean_radius, length, label):
    try:
        if np.isnan(mean_radius):
            NC = 0
        else:
            NC = int(0.10 * (ROI**2) / (2 * np.pi * (mean_radius**2)))
        trajs, labels = models_phenom().confinement(1, length, L=L,r=mean_radius/PIXEL,Nc=NC,deltaT=0.0003, Ds=[[np.random.uniform(*D),0],[np.random.uniform(*D),0]], alphas=[[1,0], [1,0]])
        return andi_datasets_to_trajectories(trajs, labels, particle_type=label)[0]
    except KeyError:
        return None

@ray.remote
def parallel_get_free_random_value_with_iou(ROI, L, D, length, label):
    try:
        trajs, labels = models_phenom().single_state(1, length, L=L, deltaT=0.0003, Ds=[np.random.uniform(*D),0], alphas=[1,0])
        return andi_datasets_to_trajectories(trajs, labels, particle_type=label)[0]
    except KeyError:
        return None

def get_random_value_with_iou(trajectories):
    D = [0.0001,1] #um2/s^-1
    D = [D[0]/(PIXEL**2), D[1]/(PIXEL**2)]

    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')

    for trajectory in trajectories:
        min_x, max_x = min(min_x, np.min(trajectory.get_noisy_x())), max(max_x, np.max(trajectory.get_noisy_x()))
        min_y, max_y = min(min_y, np.min(trajectory.get_noisy_y())), max(max_y, np.max(trajectory.get_noisy_y()))

    ROI = max(max_x-min_x, max_y-min_y)
    L = ROI/PIXEL

    #(ROI, D, mean_radius, length, label)
    original_chol_infos = [(ROI, L, D, np.mean([np.sqrt((a/np.pi)) for a in t.info['analysis']['confinement-area'] if a is not None]), t.length, CHOL_NOMENCLATURE) for t in trajectories if t.info['classified_experimental_condition'] == CHOL_NOMENCLATURE and 'analysis' in t.info]
    original_btx_infos = [(ROI, L, D, np.mean([np.sqrt((a/np.pi)) for a in t.info['analysis']['confinement-area'] if a is not None]), t.length, BTX_NOMENCLATURE) for t in trajectories if t.info['classified_experimental_condition'] == BTX_NOMENCLATURE and 'analysis' in t.info]

    chol_trajectories = ray.get([parallel_get_random_value_with_iou.remote(*info) for info in original_chol_infos])
    btx_trajectories = ray.get([parallel_get_random_value_with_iou.remote(*info) for info in original_btx_infos])

    dataframe = transform_trajectories_with_confinement_states_from_mongo_to_dataframe(chol_trajectories+btx_trajectories)

    #dataframe_plot = dataframe[dataframe['confinement-overlaps']!=0].copy()
    #dataframe_plot = dataframe_plot[dataframe_plot['confinement-states']==1]
    #dataframe_plot = dataframe[dataframe['color']=='red'].copy()
    #plt.scatter(dataframe_plot.x, dataframe_plot.y, c=dataframe_plot.color,s=0.5,alpha=0.1)
    #plt.xlabel('X [um]')
    #plt.ylabel('Y [um]')
    #plt.show()
    dataframe = dataframe[dataframe['confinement-states'] == 1]
    return measure_overlap_with_iou(dataframe, bin_size=0.007)

def get_non_confinement_random_value_with_iou(trajectories):
    D = [0.0001,1] #um2/s^-1
    D = [D[0]/(PIXEL**2), D[1]/(PIXEL**2)]

    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')

    for trajectory in trajectories:
        min_x, max_x = min(min_x, np.min(trajectory.get_noisy_x())), max(max_x, np.max(trajectory.get_noisy_x()))
        min_y, max_y = min(min_y, np.min(trajectory.get_noisy_y())), max(max_y, np.max(trajectory.get_noisy_y()))

    ROI = max(max_x-min_x, max_y-min_y)
    L = ROI/PIXEL

    #(ROI, D, length, label)
    original_chol_infos = [(ROI, L, D, t.length, CHOL_NOMENCLATURE) for t in trajectories if t.info['classified_experimental_condition'] == CHOL_NOMENCLATURE and 'analysis' in t.info]
    original_btx_infos = [(ROI, L, D, t.length, BTX_NOMENCLATURE) for t in trajectories if t.info['classified_experimental_condition'] == BTX_NOMENCLATURE and 'analysis' in t.info]

    chol_trajectories = ray.get([parallel_get_free_random_value_with_iou.remote(*info) for info in original_chol_infos])
    btx_trajectories = ray.get([parallel_get_free_random_value_with_iou.remote(*info) for info in original_btx_infos])

    dataframe = transform_trajectories_with_confinement_states_from_mongo_to_dataframe(chol_trajectories+btx_trajectories)

    #dataframe_plot = dataframe[dataframe['confinement-overlaps']!=0].copy()
    #dataframe_plot = dataframe_plot[dataframe_plot['confinement-states']==1]
    #dataframe_plot = dataframe[dataframe['color']=='red'].copy()
    #plt.scatter(dataframe_plot.x, dataframe_plot.y, c=dataframe_plot.color,s=0.5,alpha=0.1)
    #plt.xlabel('X [um]')
    #plt.ylabel('Y [um]')
    #plt.show()
    dataframe = dataframe[dataframe['confinement-states'] == 0]
    return measure_overlap_with_iou(dataframe, bin_size=0.007)

CHOL_AND_BTX_DATASETS = [
    'Cholesterol and btx',
    'CK666-BTX680-CHOL',
    'BTX680-fPEG-CHOL-50-nM',
    'BTX680-fPEG-CHOL-100-nM',
]

file_and_rois = [info for info in extract_dataset_file_roi_file() if info[0] in CHOL_AND_BTX_DATASETS]
print('Number of samples to analyze:', len(file_and_rois))

os.makedirs('confinements_overlaps_significant_test_files', exist_ok=True)

for dataset, file, roi in file_and_rois:
    print(f'./confinements_overlaps_significant_test_files/{dataset}_{file}_{roi}.txt')
    if not os.path.exists(f'./confinements_overlaps_significant_test_files/{dataset}_{file}_{roi}.txt'):
        DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)
        trajectories = list(Trajectory.objects(info__file=file, info__dataset=dataset, info__roi=roi))
        DatabaseHandler.disconnect()

        real_dataframe = transform_trajectories_with_confinement_states_from_mongo_to_dataframe(trajectories)
        real_dataframe = real_dataframe[real_dataframe['confinement-states'] == 1]
        real_value = measure_overlap_with_iou(real_dataframe, bin_size=0.007)
        simulated_values = [get_random_value_with_iou(trajectories) for _ in tqdm.tqdm(range(99))]
        simulated_values.append(real_value)
        np.savetxt(f'./confinements_overlaps_significant_test_files/{dataset}_{file}_{roi}.txt', simulated_values)

os.makedirs('non_confinement_overlaps_significant_test_files', exist_ok=True)

for dataset, file, roi in file_and_rois:
    print(f'./non_confinement_overlaps_significant_test_files/{dataset}_{file}_{roi}.txt')
    if not os.path.exists(f'./non_confinement_overlaps_significant_test_files/{dataset}_{file}_{roi}.txt'):
        DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)
        trajectories = list(Trajectory.objects(info__file=file, info__dataset=dataset, info__roi=roi))
        DatabaseHandler.disconnect()

        real_dataframe = transform_trajectories_with_confinement_states_from_mongo_to_dataframe(trajectories)
        real_dataframe = real_dataframe[real_dataframe['confinement-states'] == 0]
        real_value = measure_overlap_with_iou(real_dataframe, bin_size=0.007)
        simulated_values = [get_non_confinement_random_value_with_iou(trajectories) for _ in tqdm.tqdm(range(99))]
        simulated_values.append(real_value)
        np.savetxt(f'./non_confinement_overlaps_significant_test_files/{dataset}_{file}_{roi}.txt', simulated_values)
