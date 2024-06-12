from DatabaseHandler import DatabaseHandler
from Trajectory import Trajectory
from CONSTANTS import *
import os


def unique(list1):
    unique_list = []
    for x in list1:
        if x not in unique_list:
            unique_list.append(x)
    return unique_list

DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)

if not os.path.exists(FILE_AND_ROI_FILE_CACHE):
    file_id_and_roi_list = [[r['info']['dataset'],r['info']['file'], r['info']['roi']] for r in Trajectory._get_collection().find(
        {},
        {f'id':1, 'info.roi':1, 'info.file':1, 'info.dataset': 1}
    )]

    file_id_and_roi_list = unique(file_id_and_roi_list)

    a_file = open(FILE_AND_ROI_FILE_CACHE, 'w')
    for dataset, file, roi in file_id_and_roi_list:
        a_file.write(f'{dataset},{file},{roi}\n')
    a_file.close()

DatabaseHandler.disconnect()
