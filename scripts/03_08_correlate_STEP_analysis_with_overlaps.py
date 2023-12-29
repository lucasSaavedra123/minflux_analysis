import itertools

import tqdm
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
import warnings

from DatabaseHandler import DatabaseHandler
from Trajectory import Trajectory
from CONSTANTS import *
from utils import *

warnings.filterwarnings('error') 
warnings.filterwarnings("ignore", category=DeprecationWarning) 

DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)

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


intensities_chol, diffusion_chol, intensities_chol_with_btx, diffusion_chol_with_btx = np.array([]), np.array([]), np.array([]), np.array([])
intensities_btx, diffusion_btx, intensities_btx_with_chol, diffusion_btx_with_chol = np.array([]), np.array([]), np.array([]), np.array([])


for file in tqdm.tqdm(files):
    for trajectory in Trajectory.objects(info__file=file):
        if trajectory.length == 1 or not ('intensity' in trajectory.info and len(trajectory.info['intensity']) != 0):
            continue

        if trajectory.info['classified_experimental_condition'] == BTX_NOMENCLATURE:
            intersections_with_chol = np.array(trajectory.info[f'{CHOL_NOMENCLATURE}_single_intersections'])
            assert len(intersections_with_chol) == trajectory.length

            there_is_intersection = intersections_with_chol == 1
            there_is_no_intersection = intersections_with_chol == 0

            intensities_btx = np.append(intensities_btx, np.array(trajectory.info['intensity'])[there_is_no_intersection])
            diffusion_btx = np.append(diffusion_btx, np.array(trajectory.info['analysis']['step_result'])[there_is_no_intersection])

            intensities_btx_with_chol = np.append(intensities_btx_with_chol, np.array(trajectory.info['intensity'])[there_is_intersection])
            diffusion_btx_with_chol = np.append(diffusion_btx_with_chol, np.array(trajectory.info['analysis']['step_result'])[there_is_intersection])
        else:
            intersections_with_btx = np.array(trajectory.info[f'{BTX_NOMENCLATURE}_single_intersections'])
            assert len(intersections_with_btx) == trajectory.length

            there_is_intersection = intersections_with_btx == 1
            there_is_no_intersection = intersections_with_btx == 0

            intensities_chol = np.append(intensities_chol, np.array(trajectory.info['intensity'])[there_is_no_intersection])
            diffusion_chol = np.append(diffusion_chol, np.array(trajectory.info['analysis']['step_result'])[there_is_no_intersection])

            intensities_chol_with_btx = np.append(intensities_chol_with_btx, np.array(trajectory.info['intensity'])[there_is_intersection])
            diffusion_chol_with_btx = np.append(diffusion_chol_with_btx, np.array(trajectory.info['analysis']['step_result'])[there_is_intersection])

pd.DataFrame({
    'intensity': intensities_btx,
    'diffusion': diffusion_btx,
}).to_csv('Results/STEP_correlation_btx.csv', index=False)

pd.DataFrame({
    'intensity_with_intersection': intensities_btx_with_chol,
    'diffusion_with_intersection': diffusion_btx_with_chol,
}).to_csv('Results/STEP_correlation_btx_with_chol.csv', index=False)

pd.DataFrame({
    'intensity': intensities_chol,
    'diffusion': diffusion_chol,
}).to_csv('Results/STEP_correlation_chol.csv', index=False)

pd.DataFrame({
    'intensity_with_intersection': intensities_chol_with_btx,
    'diffusion_with_intersection': diffusion_chol_with_btx,
}).to_csv('Results/STEP_correlation_chol_with_btx.csv', index=False)

DatabaseHandler.disconnect()
