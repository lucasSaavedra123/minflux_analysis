import matplotlib.pyplot as plt
from DatabaseHandler import DatabaseHandler
from CONSTANTS import *
from Trajectory import Trajectory
from collections import defaultdict
from utils import extract_dataset_file_roi_file
import tqdm
from IPython import embed

DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)


#THIS IS FOR TRAJECTORY DELETION
"""
files_to_delete = [
    '',
]

for a_file in files_to_delete:
    for t in Trajectory.objects(info__dataset='BTX680R', info__file=a_file):
        t.delete()
exit()
"""
def equal_trajectories(t1,t2):
    equal_x = all([t1_x==t2_x for t1_x, t2_x in zip(t1['x'], t2['x'])]) and len(t1['x'])==len(t2['x'])
    equal_y = all([t1_y==t2_y for t1_y, t2_y in zip(t1['y'], t2['y'])]) and len(t1['y'])==len(t2['y'])
    return equal_x and equal_y

def datasets_overlap(t1s, t2s):
    for t1 in t1s:
        for t2 in t2s:
            if equal_trajectories(t1,t2):
                return True

    return False

def combine_edges(edges):
  combinations = []

  for edge in edges:
    found = False
    for combination in combinations:
      if edge[0] in combination or edge[1] in combination:
        combination.append(edge[0])
        combination.append(edge[1])
        found = True

    if not found:
      combinations.append([edge[0], edge[1]])

  return [list(set(c)) for c in combinations]

dataset_file_roi = extract_dataset_file_roi_file()
datasets = ['Cholesterol and btx']#set([a[0] for a in dataset_file_roi])
print(datasets)
for dataset in datasets:
    print(dataset)
    trajectories_info_by_file = defaultdict(lambda: [])
    number_of_rois_by_file = defaultdict(lambda: [])

    for doc in tqdm.tqdm(Trajectory._get_collection().find({'info.dataset':dataset}, {f'x':1,'y':1, 'info.roi':1, 'info.file':1})):
        trajectories_info_by_file[doc['info']['file']].append(doc)
        number_of_rois_by_file[doc['info']['file']].append(doc['info']['roi'])

    for file in number_of_rois_by_file:
       number_of_rois_by_file[file] = len(list(set(number_of_rois_by_file[file])))

    pairs = []

    files = list(trajectories_info_by_file.keys())

    for i, a_file in tqdm.tqdm(enumerate(files)):
        for another_file in tqdm.tqdm(files[i:]):
            if a_file == another_file:
                continue

            first_file_trajectories = trajectories_info_by_file[a_file]
            second_second_trajectories = trajectories_info_by_file[another_file]

            if datasets_overlap(first_file_trajectories, second_second_trajectories):
                print(a_file, another_file)
                """
                infos = [{'info.dataset':dataset, 'info.file':a_file}, {'info.dataset':dataset, 'info.file':another_file}]

                fig, ax = plt.subplots(1,2)

                for i, info in enumerate(infos):
                    ax[i].set_title(info['info.file'])

                    for d in trajectories_info_by_file[info['info.file']]:
                        ax[i].plot(d['x'], d['y'], color=['red', 'blue'][i])

                plt.show()
                """

DatabaseHandler.disconnect()