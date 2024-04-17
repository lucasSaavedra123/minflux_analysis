from CONSTANTS import *
from DatabaseHandler import DatabaseHandler
from Trajectory import Trajectory
from collections import defaultdict


counter = defaultdict(lambda: [])

DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)

for result in Trajectory._get_collection().find({'info.immobile':False}, {f'id':1, 'info.roi':1, 'info.file':1, 'info.dataset':1, 'info.classified_experimental_condition':1}):
    if 'classified_experimental_condition' in result['info']:
        label = result['info']['dataset']+'_'+result['info']['classified_experimental_condition']
    else:
        label = result['info']['dataset']

    new_roi_info = [result['info']['file'],result['info']['roi']]

    if new_roi_info not in counter[label]:
        counter[label].append(new_roi_info)

for i_key in counter:
    counter[i_key] = len(counter[i_key])

print(counter)

DatabaseHandler.disconnect()