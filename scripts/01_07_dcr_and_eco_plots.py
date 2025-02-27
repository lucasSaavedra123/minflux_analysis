from CONSTANTS import *
from DatabaseHandler import DatabaseHandler
from Trajectory import Trajectory
from scipy.signal import savgol_filter
import ruptures as rpt
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)

CHOL_AND_BTX_DATASETS = [
    'Cholesterol and btx',
    'CK666-BTX680-CHOL',
    'BTX680-fPEG-CHOL-50-nM',
    'BTX680-fPEG-CHOL-100-nM',
]

get_paper_plots = False

if not get_paper_plots:
    datasets_to_iterate = DATASETS_LIST
else:
    datasets_to_iterate = [
        'BTX680R',
        'CholesterolPEGKK114',
        'Cholesterol and btx'
    ]

    index_to_select = {
        datasets_to_iterate[0]: 93,
        datasets_to_iterate[1]: 5,
        datasets_to_iterate[2]: 1
    }

for dataset in datasets_to_iterate:
    trajectories_info = list(Trajectory._get_collection().find({'info.dataset':dataset, 'info.immobile': False}, {'x':1,'y':1,'t':1,'info.intensity':1, 'info.dcr':1}))

    if get_paper_plots:
        trajectories_info = trajectories_info[index_to_select[dataset]:index_to_select[dataset]+1]

    for i, info in enumerate(trajectories_info[:100]):
        if 'info' not in info:
            continue
        elif 'dcr' not in info['info']:
            continue
        elif 'intensity' not in info['info']:
            continue
        elif len(info['info']['intensity']) < 1000:
            continue
        
        t = info['t']
        dcr = info['info']['dcr']
        efo = info['info']['intensity']
        eco = np.diff(t) * efo[:-1]

        dcr_smoothed = savgol_filter(dcr, 50, 1)
        efo_smoothed = savgol_filter(efo, 50, 1)
        eco_smoothed = savgol_filter(eco, 50, 1)

        within_range = ((dcr_smoothed < 0.55).astype(int) * (dcr_smoothed > 0.40).astype(int)).astype(bool)

        if True:#within_range.any():
            font = {'family' :'arial','size': 22}

            mpl.rc('font', **font)

            fig, ax = plt.subplots(2,1)

            steps = list(range(1, len(t)+1))

            # Normalize the array vals so they can be mapped to a color
            c_norm = mpl.colors.Normalize(vmin=0, vmax=1)

            # Pick a colormap
            #c_map  = mpl.cm.cividis
            c_map = mpl.colors.LinearSegmentedColormap.from_list('custom_grad', (
                    # Edit this gradient at https://eltos.github.io/gradient/#0:FF7100-37:FF7100-40:000000-55:000000-58:00B8FF-100:00B8FF
                    (0.000, (0.000, 0.722, 1.000)),
                    (0.370, (0.000, 0.722, 1.000)),
                    (0.400, (0.000, 0.000, 0.000)),
                    (0.550, (0.000, 0.000, 0.000)),
                    (0.580, (1.000, 0.443, 0.000)),
                    (1.000, (1.000, 0.443, 0.000))
                )
            )

            # Scalar mappable of normalized array to colormap
            s_map  = mpl.cm.ScalarMappable(cmap=c_map, norm=c_norm)
            s_map.set_array([])

            ax[0].plot(dcr, color='#ABABAB')
            #ax[0].plot(dcr_smoothed, color='red')
            
            for point_i in range(1,len(t)-1):
                ax[0].plot(steps[point_i-1:point_i+1], dcr_smoothed[point_i-1:point_i+1], color=s_map.to_rgba(dcr_smoothed[point_i]), linewidth=3)
            
            ax[0].set_ylim([0,1])
            #ax[0].set_xlabel('Step')
            ax[0].set_ylabel('DCR')
            ax[0].axhline(0.40, linestyle='--', color='black')
            ax[0].axhline(0.55, linestyle='--', color='black')

            ax[1].plot(eco, color='#ABABAB')

            for point_i in range(1,len(t)-1):
                ax[1].plot(steps[point_i-1:point_i+1], eco_smoothed[point_i-1:point_i+1], color=s_map.to_rgba(dcr_smoothed[point_i]), linewidth=3)

            #ax[1].set_ylim([0,1_000_000])
            ax[1].set_ylim([0,2_000])
            ax[1].ticklabel_format(axis='y',style='sci')
            ax[1].set_xlabel('Step')
            ax[1].set_ylabel('ECO') 

            plt.tight_layout()
            plt.savefig(f'./dcr_intensities/{dataset}_{str(i if not get_paper_plots else index_to_select[dataset]).zfill(9)}.svg')

            fig, ax = plt.subplots(1,1)

            for point_i in range(1,len(t)-1):
                ax.plot(info['x'][point_i-1:point_i+1], info['y'][point_i-1:point_i+1], color=s_map.to_rgba(dcr_smoothed[point_i]), linewidth=3)

            ax.set_xlabel('X [μm]')
            ax.set_ylabel('Y [μm]')

            ax.set_aspect('equal')
            
            xlim = ax.get_xlim()
            x_middle = (xlim[1]+xlim[0])/2
            ylim = ax.get_ylim()
            y_middle = (ylim[1]+ylim[0])/2

            if xlim[1] - xlim[0] > ylim[1] - ylim[0]:
                ax.set_xlim([x_middle-((xlim[1] - xlim[0])/2), x_middle+((xlim[1] - xlim[0])/2)])
                ax.set_ylim([y_middle-((xlim[1] - xlim[0])/2), y_middle+((xlim[1] - xlim[0])/2)])
            else:
                ax.set_xlim([x_middle-((ylim[1] - ylim[0])/2), x_middle+((ylim[1] - ylim[0])/2)])
                ax.set_ylim([y_middle-((ylim[1] - ylim[0])/2), y_middle+((ylim[1] - ylim[0])/2)])

            plt.subplots_adjust(left=0.31, right=0.983, top=0.968, bottom=0.137)
            plt.tight_layout()
            plt.savefig(f'./dcr_intensities/{dataset}_{str(i if not get_paper_plots else index_to_select[dataset]).zfill(9)}_trajectory.png', dpi=600)

DatabaseHandler.disconnect()