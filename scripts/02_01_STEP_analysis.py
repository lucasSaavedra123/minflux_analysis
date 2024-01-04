"""
STEP is executed to every trajectory.
"""

import numpy as np
import tqdm
from torch import from_numpy
from step.data import *
from step.models import *
from step.utils import *
from fastai.vision.all import *
import matplotlib.pyplot as plt
from DatabaseHandler import DatabaseHandler
from Trajectory import Trajectory
from CONSTANTS import *


dim = 2
dls = DataLoaders.from_dsets([], []) # Empty train and validation datasets
model = XResAttn(dim, n_class=1, stem_szs=(64,), conv_blocks=[1, 1, 1],
                block_szs=[128, 256, 512], pos_enc=False,
                n_encoder_layers=4, dim_ff=512, nhead_enc=8,
                linear_layers=[], norm=False, yrange=(-3.1, 3.1), time_included=True)

model.to(default_device())
learn_diff = Learner(dls, model, loss_func=L1LossFlat(), model_dir='./models')
learn_diff.load(f'custom_xresattn_bm_2d_1_to_4_cp')
learn_diff.model.eval()

def predict(trajectory):
    x = np.zeros((1, trajectory.length, 3))
    x[0,:,0] = trajectory.get_noisy_x() - trajectory.get_noisy_x()[0]
    x[0,:,1] = trajectory.get_noisy_y() - trajectory.get_noisy_y()[0]
    x[0,:,2] = trajectory.get_time() - trajectory.get_time()[0]

    result = learn_diff.model(from_numpy(x).float()).squeeze().detach().numpy()

    return 10**result

def analyze_trajectory(trajectory_id):
    trajectories = Trajectory.objects(id=trajectory_id)
    assert len(trajectories) == 1
    trajectory = trajectories[0]

    #if trajectory.length == 1 or 'step_result' in trajectory.info['analysis']:
    if trajectory.length == 1:
        return None
    else:
        trajectory.info['analysis']['step_result'] = None

    trajectory.info['analysis']['step_result'] = {
        'confinement': [],
        'non-confinement': []
    }

    s = trajectory.sub_trajectories_trajectories_from_confinement_states(v_th=33)

    for t in s[0]:
        if t.length > 100:
            trajectory.info['analysis']['step_result']['non-confinement'].append(np.mean(predict(t).tolist()))

    for t in s[1]:
        if t.length > 100:
            trajectory.info['analysis']['step_result']['confinement'].append(np.mean(predict(t).tolist()))
    """
    cm = plt.cm.get_cmap('copper_r')

    plt.plot(trajectory.info['analysis']['step_result'])
    plt.show()

    plt.scatter(trajectory.get_noisy_x(), trajectory.get_noisy_y(), c=trajectory.info['analysis']['step_result'], s=35, cmap=cm)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel(r'$D ({\mu m}^{2}/{s})$')
    plt.xlabel(r'X [$\mu m$]')
    plt.ylabel(r'Y [$\mu m$]')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()

    plt.plot(trajectory.get_noisy_x(), trajectory.get_noisy_y(), c='grey', zorder=-99, linewidth='1')
    plt.show()
    """
    trajectory.save()

DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)

for a_result in tqdm.tqdm(Trajectory._get_collection().find({}, {'_id':1})):
    analyze_trajectory(str(a_result['_id']))

DatabaseHandler.disconnect()
