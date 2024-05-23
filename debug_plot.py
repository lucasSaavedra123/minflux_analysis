import matplotlib.pyplot as plt
from DatabaseHandler import DatabaseHandler
from CONSTANTS import *
from Trajectory import Trajectory


DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)

infos = [
    {'info.dataset':'Cholesterol and btx', 'info.file':'231013-124511_mbm test.txt'},
    {'info.dataset':'Cholesterol and btx', 'info.file':'231013-132703_mbm test.txt'}]

fig, ax = plt.subplots(1,2)

for i, info in enumerate(infos):
    ax[i].set_title(info['info.file'])
    documents = Trajectory._get_collection().find(info, {f'x':1,'y':1})

    for d in documents:
        ax[i].plot(d['x'], d['y'], color=['red', 'blue'][i])

plt.show()


DatabaseHandler.disconnect()