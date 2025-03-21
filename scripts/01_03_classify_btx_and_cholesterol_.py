"""
With the DCR value selected on 01_02_dcr_analysis.py, 
the classification is done
"""

import tqdm
import numpy as np

from DatabaseHandler import DatabaseHandler
from Trajectory import Trajectory
from CONSTANTS import *


DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)

for dataset in [
    'Cholesterol and btx',
    'CK666-BTX680-CHOL',
    'BTX680-fPEG-CHOL-50-nM',
    'BTX680-fPEG-CHOL-100-nM',
]:
    for trajectory_id in tqdm.tqdm(Trajectory._get_collection().find({'info.dataset': dataset}, {'_id':1})):
        trajectory = Trajectory.objects(id=str(trajectory_id['_id']))[0]
        trajectory.info['classified_experimental_condition'] = CHOL_NOMENCLATURE if np.mean(trajectory.info['dcr']) > TDCR_THRESHOLD else BTX_NOMENCLATURE
        trajectory.save()

DatabaseHandler.disconnect()
