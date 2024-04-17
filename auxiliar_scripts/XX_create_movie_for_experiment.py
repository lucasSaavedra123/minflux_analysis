"""
All important results like areas, axis lengths, etc. 
are produced within this file.
"""
from DatabaseHandler import DatabaseHandler
from Trajectory import Trajectory
from CONSTANTS import *
from utils import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tqdm
import moviepy.editor as mp


APPLY_GS_CRITERIA = True

#Uncomment this part if you want to animate an entire file

DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)
trajectories = list(Trajectory.objects(info__file='231013-162602_mbm test.txt', info__roi=12))
DatabaseHandler.disconnect()

fig, ax = plt.subplots()

trajectories_to_dataframe = {
    'x': [],
    'y': [],
    't': [],
    'track_id': []
}


trajectory_line_dict = {}

min_t, max_t = None, None
min_x, max_x = None, None
min_y, max_y = None, None
i = 0
for trajectory in trajectories:
    if i > 10:
        break
    trajectory_line_dict[trajectory] = {}
    trajectory_line_dict[trajectory]['dataframe'] = pd.DataFrame({
        'x': trajectory.get_noisy_x(),
        'y': trajectory.get_noisy_y(),
        't': trajectory.get_time(),
    })

    trajectory_line_dict[trajectory]['line'] = ax.plot(trajectory.get_noisy_x()[0], trajectory.get_noisy_y()[0])[0]
    i += 1
all_info = pd.concat([trajectory_line_dict[d]['dataframe'] for d in trajectory_line_dict])
all_info = all_info.sort_values('t')
ax.set(xlim=[all_info['x'].min(), all_info['x'].max()], ylim=[all_info['y'].min(), all_info['y'].max()], xlabel='X', ylabel='Y')

def update(time):
    # for each frame, update the data stored on each artist.
    lines = []
    for trajectory in trajectory_line_dict:
        dataframe = trajectory_line_dict[trajectory]['dataframe']
        line = trajectory_line_dict[trajectory]['line']

        dataframe = dataframe[dataframe['t'] <= time]

        if len(dataframe) != 0:
            x_f = dataframe['x'].tolist()
            y_f = dataframe['y'].tolist()
            line.set_xdata(x_f)
            line.set_ydata(y_f)
            lines.append(line)

    ax.set_title(f'{time}s')
    return lines

anim = animation.FuncAnimation(fig=fig, func=update, frames=all_info['t'].unique(), interval=1)
writervideo = animation.PillowWriter(fps=60) 
anim.save('./animation.gif', writer=writervideo)

DatabaseHandler.disconnect()
