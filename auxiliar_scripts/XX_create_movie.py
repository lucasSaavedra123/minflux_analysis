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
from moviepy.video.fx.all import crop
from moviepy.editor import *
import numpy as np
from scipy import interpolate

APPLY_GS_CRITERIA = True


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

for file in tqdm.tqdm(files):
    plot_counter = 0
    trajectories = Trajectory.objects(info__file=file)
    for index, trajectory in enumerate(trajectories):
        if trajectory.info['classified_experimental_condition'] == BTX_NOMENCLATURE and trajectory.info['number_of_confinement_zones'] != 0 and 0.30 <= trajectory.info[f'number_of_confinement_zones_with_{CHOL_NOMENCLATURE}']/trajectory.info['number_of_confinement_zones'] <= 0.40:
            """
            x, y = trajectory.get_noisy_x(), trajectory.get_noisy_y()
            tck,u = interpolate.splprep([x,y], s=20)
            u=np.linspace(0,1,num=trajectory.length,endpoint=True)
            out = interpolate.splev(u,tck)
            plt.figure()
            plt.plot(x, y, 'ro', out[0], out[1], 'b')
            plt.legend(['Points', 'Interpolated B-spline', 'True'],loc='best')
            plt.axis([min(x)-1, max(x)+1, min(y)-1, max(y)+1])
            plt.title('B-Spline interpolation')
            plt.show()
            exit()
            """
            receptor_polygons = []
            for sub_t in trajectory.sub_trajectories_trajectories_from_confinement_states(v_th=33, use_info=True)[1]:
                xx, yy = MultiPoint(list(zip(sub_t.get_noisy_x(), sub_t.get_noisy_y()))).convex_hull.exterior.coords.xy
                xx, yy =  xx.tolist(), yy.tolist()
                receptor_polygons.append((sub_t, sort_vertices_anti_clockwise_and_remove_duplicates(list(zip(xx, yy)))))

            for chol_trajectory_id in trajectory.info[f'{CHOL_NOMENCLATURE}_intersections']:
                chol_polygons = []
                aux_t = Trajectory.objects(id=chol_trajectory_id)[0]

                for sub_t in aux_t.sub_trajectories_trajectories_from_confinement_states(v_th=33, use_info=True)[1]:
                    xx, yy = MultiPoint(list(zip(sub_t.get_noisy_x(), sub_t.get_noisy_y()))).convex_hull.exterior.coords.xy
                    xx, yy =  xx.tolist(), yy.tolist()
                    chol_polygons.append((sub_t, sort_vertices_anti_clockwise_and_remove_duplicates(list(zip(xx, yy)))))

                for chol_polygon in chol_polygons:
                    for receptor_polygon in receptor_polygons:
                        polygon_intersection = intersect(receptor_polygon[1], chol_polygon[1])
                        if len(polygon_intersection) != 0:
                            chol_time_mean = np.mean(chol_polygon[0].get_time())
                            receptor_time_mean = np.mean(receptor_polygon[0].get_time())

                            if receptor_time_mean > chol_time_mean:
                                receptor_time_offset = 0
                                chol_time_offset = np.mean(receptor_time_mean) - np.mean(chol_time_mean)
                            else:
                                receptor_time_offset =  np.mean(chol_time_mean) - np.mean(receptor_time_mean)
                                chol_time_offset = 0

                            plt.style.use('dark_background')
                            fig, ax = plt.subplots()
                            ax.set_aspect('equal', adjustable='box')
                            receptor_trajectory_line_dict = {}
                            receptor_trajectory_line_dict['line'] = ax.plot(trajectory.get_noisy_x()[0], trajectory.get_noisy_y()[0], color='green', linewidth=4)[0]

                            receptor_trajectory_line_dict['dataframe'] = pd.DataFrame({
                                'x': trajectory.get_noisy_x(),
                                'y': trajectory.get_noisy_y(),
                                't': trajectory.get_time() + receptor_time_offset,
                            })

                            chol_trajectory_line_dict = {}
                            chol_trajectory_line_dict['line'] = ax.plot(aux_t.get_noisy_x()[0], aux_t.get_noisy_y()[0], color='red', linewidth=4)[0]

                            chol_trajectory_line_dict['dataframe'] = pd.DataFrame({
                                'x': aux_t.get_noisy_x(),
                                'y': aux_t.get_noisy_y(),
                                't': aux_t.get_time() + chol_time_offset,
                            })

                            all_info = pd.concat([chol_trajectory_line_dict['dataframe'], receptor_trajectory_line_dict['dataframe']])
                            all_info = all_info.sort_values('t')

                            intersection_centroid = np.mean(np.array(polygon_intersection), axis=0)

                            edge_offset = 0.15

                            tiempo_min = max(chol_trajectory_line_dict['dataframe']['t'].min(), receptor_trajectory_line_dict['dataframe']['t'].min())
                            tiempo_max = min(chol_trajectory_line_dict['dataframe']['t'].max(), receptor_trajectory_line_dict['dataframe']['t'].max())

                            # Filtrar filas con solapamiento de tiempo en df1
                            chol_trajectory_line_dict['dataframe'] = chol_trajectory_line_dict['dataframe'][(chol_trajectory_line_dict['dataframe']['t'] >= tiempo_min) & (chol_trajectory_line_dict['dataframe']['t'] <= tiempo_max)]
                            receptor_trajectory_line_dict['dataframe'] = receptor_trajectory_line_dict['dataframe'][(receptor_trajectory_line_dict['dataframe']['t'] >= tiempo_min) & (receptor_trajectory_line_dict['dataframe']['t'] <= tiempo_max)]

                            ax.set(
                                xlim=[
                                    intersection_centroid[0] - edge_offset, intersection_centroid[0]+edge_offset
                                ], 
                                ylim=[
                                    intersection_centroid[1] - edge_offset, intersection_centroid[1]+edge_offset
                                ],
                                #xlabel='X [μm]',
                                #ylabel='Y [μm]'
                            )

                            ax.get_xaxis().set_visible(False)
                            ax.get_yaxis().set_visible(False)
                            #ax.get_xaxis().set_ticklabels([])
                            #ax.get_yaxis().set_ticklabels([])
                            frames = np.unique((sorted(chol_trajectory_line_dict['dataframe']['t'].tolist() + receptor_trajectory_line_dict['dataframe']['t'].tolist())))

                            tck,u = interpolate.splprep([receptor_trajectory_line_dict['dataframe']['x'].tolist(),receptor_trajectory_line_dict['dataframe']['y'].tolist()], s=10)
                            u=np.linspace(0,1,num=len(frames),endpoint=True)
                            camera_track_for_receptor = interpolate.splev(u,tck)

                            tck,u = interpolate.splprep([chol_trajectory_line_dict['dataframe']['x'].tolist(),chol_trajectory_line_dict['dataframe']['y'].tolist()], s=10)
                            u=np.linspace(0,1,num=len(frames),endpoint=True)
                            camera_track_for_chol = interpolate.splev(u,tck)

                            camera_track_for_receptor = pd.DataFrame({
                                'x': camera_track_for_receptor[0],
                                'y': camera_track_for_receptor[1],
                                't': frames,
                            })

                            camera_track_for_chol = pd.DataFrame({
                                'x': camera_track_for_chol[0],
                                'y': camera_track_for_chol[1],
                                't': frames,
                            })

                            def update_one(time):
                                # for each frame, update the data stored on each artist.
                                lines = []

                                for trajectory_line_dict in [chol_trajectory_line_dict, receptor_trajectory_line_dict]:
                                    dataframe = trajectory_line_dict['dataframe']
                                    line = trajectory_line_dict['line']

                                    dataframe = dataframe[dataframe['t'] <= time]

                                    if len(dataframe) != 0:
                                        x_f = dataframe['x'].tolist()
                                        y_f = dataframe['y'].tolist()
                                        line.set_xdata(x_f)
                                        line.set_ydata(y_f)
                                        lines.append(line)

                                ax.set_title(f'{np.round(time, 3)}s')
                                return lines + [ax.scatter(
                                    [intersection_centroid[0]],
                                    [intersection_centroid[1]],
                                    color='white',
                                    marker='+')
                                ]

                            def update_two(time):
                                # for each frame, update the data stored on each artist.
                                lines = []

                                for trajectory_line_dict in [chol_trajectory_line_dict, receptor_trajectory_line_dict]:
                                    dataframe = trajectory_line_dict['dataframe']
                                    line = trajectory_line_dict['line']

                                    dataframe = dataframe[dataframe['t'] <= time]
                                    camera_track = camera_track_for_receptor[camera_track_for_receptor['t'] <= time]

                                    if len(dataframe) != 0:
                                        if trajectory_line_dict == receptor_trajectory_line_dict:
                                            ax.set(
                                                xlim=[
                                                    camera_track['x'].tolist()[-1] - edge_offset, camera_track['x'].tolist()[-1]+edge_offset
                                                ], 
                                                ylim=[
                                                    camera_track['y'].tolist()[-1] - edge_offset, camera_track['y'].tolist()[-1]+edge_offset
                                                ],
                                                #xlabel='X [μm]',
                                                #ylabel='Y [μm]'
                                            )

                                        x_f = dataframe['x'].tolist()
                                        y_f = dataframe['y'].tolist()
                                        line.set_xdata(x_f)
                                        line.set_ydata(y_f)
                                        lines.append(line)

                                ax.set_title(f'{np.round(time, 3)}s')
                                return lines + [ax.scatter(
                                    [intersection_centroid[0]],
                                    [intersection_centroid[1]],
                                    color='white',
                                    marker='+')
                                ]

                            def update_three(time):
                                # for each frame, update the data stored on each artist.
                                lines = []

                                for trajectory_line_dict in [chol_trajectory_line_dict, receptor_trajectory_line_dict]:
                                    dataframe = trajectory_line_dict['dataframe']
                                    line = trajectory_line_dict['line']

                                    dataframe = dataframe[dataframe['t'] <= time]
                                    camera_track = camera_track_for_chol[camera_track_for_chol['t'] <= time]

                                    if len(dataframe) != 0:
                                        if trajectory_line_dict == chol_trajectory_line_dict:
                                            ax.set(
                                                xlim=[
                                                    camera_track['x'].tolist()[-1] - edge_offset, camera_track['x'].tolist()[-1]+edge_offset
                                                ], 
                                                ylim=[
                                                    camera_track['y'].tolist()[-1] - edge_offset, camera_track['y'].tolist()[-1]+edge_offset
                                                ],
                                                #xlabel='X [μm]',
                                                #ylabel='Y [μm]'
                                            )

                                        x_f = dataframe['x'].tolist()
                                        y_f = dataframe['y'].tolist()
                                        line.set_xdata(x_f)
                                        line.set_ydata(y_f)
                                        lines.append(line)

                                ax.set_title(f'{np.round(time, 3)}s')
                                return lines + [ax.scatter(
                                    [intersection_centroid[0]],
                                    [intersection_centroid[1]],
                                    color='white',
                                    marker='+')
                                ]

                            anim_one = animation.FuncAnimation(fig=fig, func=update_one, frames=frames)
                            anim_two = animation.FuncAnimation(fig=fig, func=update_two, frames=frames)
                            anim_three = animation.FuncAnimation(fig=fig, func=update_three, frames=frames)

                            for i, anim in enumerate([anim_one, anim_two, anim_three]):
                                anim.save(f'animation.gif', writer=animation.PillowWriter(fps=30), dpi=300)

                                clip = mp.VideoFileClip(f'animation.gif')
                                (w, h) = clip.size
                                clip = crop(clip, width=h, height=h, x_center=w/2, y_center=h/2)
                                clip.write_videofile(f'./animations/{file}_{plot_counter}_animation_{i}.mp4')

                            combined_clip = clips_array([[
                                mp.VideoFileClip(f'./animations/{file}_{plot_counter}_animation_0.mp4'),
                                mp.VideoFileClip(f'./animations/{file}_{plot_counter}_animation_1.mp4'),
                                mp.VideoFileClip(f'./animations/{file}_{plot_counter}_animation_2.mp4'),
                            ]])

                            combined_clip.write_videofile(f'./animations/{file}_{plot_counter}_animation.mp4')
                            plot_counter += 1

DatabaseHandler.disconnect()
