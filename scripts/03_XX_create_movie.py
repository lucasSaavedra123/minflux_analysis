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
from scipy.spatial.distance import cdist
from IPython import embed

APPLY_GS_CRITERIA = True

def smooth_square(x,reference,width):
    def gaussian(x, mu, sig):
        return (
            1.0 / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-np.power((x - mu) / sig, 2.0) / 2)
        )

    return gaussian(x, reference, width/2)

def clear_ax(ax):
    ax.cla()
    #ax.set_aspect('equal', adjustable='box')
    ax.xaxis.pane.fill = False # Left pane
    ax.yaxis.pane.fill = False # Right pane

    # Remove grid lines
    ax.grid(False)

    # Remove tick labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    # Transparent spines
    ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

    # Transparent panes
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # No ticks
    ax.set_xticks([]) 
    ax.set_yticks([]) 
    ax.set_zticks([])

DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)

CHOL_AND_BTX_DATASETS = [
    'Cholesterol and btx',
    'CK666-BTX680-CHOL',
    'BTX680-fPEG-CHOL-50-nM',
    'BTX680-fPEG-CHOL-100-nM',
]

files = [[info[1], info[2]] for info in extract_dataset_file_roi_file() if info[0] in CHOL_AND_BTX_DATASETS]


for file, roi in files:
    print(file,roi)
    plot_counter = 0
    trajectories = Trajectory.objects(info__file=file, info__roi=roi)
    for index, trajectory in enumerate(trajectories):
        if trajectory.info['classified_experimental_condition'] == BTX_NOMENCLATURE and trajectory.info['number_of_confinement_zones'] != 0:# and 0.30 <= trajectory.info[f'number_of_confinement_zones_with_{CHOL_NOMENCLATURE}']/trajectory.info['number_of_confinement_zones'] <= 0.40:
            video_found = False
            receptor_polygons = []
            for sub_t in trajectory.sub_trajectories_trajectories_from_confinement_states(v_th=33, use_info=True)[1]:
                xx, yy = MultiPoint(list(zip(sub_t.get_noisy_x(), sub_t.get_noisy_y()))).convex_hull.exterior.coords.xy
                xx, yy =  xx.tolist(), yy.tolist()
                receptor_polygons.append((sub_t, sort_vertices_anti_clockwise_and_remove_duplicates(list(zip(xx, yy)))))

            #for chol_trajectory_id in trajectory.info[f'{CHOL_NOMENCLATURE}_intersections']:
            for aux_t in [t for t in trajectories if t.info['classified_experimental_condition'] == CHOL_NOMENCLATURE]:
                chol_polygons = []
                #aux_t = Trajectory.objects(id=chol_trajectory_id)[0]
                try:
                    for sub_t in aux_t.sub_trajectories_trajectories_from_confinement_states(v_th=33, use_info=True)[1]:
                        try:
                            xx, yy = MultiPoint(list(zip(sub_t.get_noisy_x(), sub_t.get_noisy_y()))).convex_hull.exterior.coords.xy
                            xx, yy =  xx.tolist(), yy.tolist()
                            chol_polygons.append((sub_t, sort_vertices_anti_clockwise_and_remove_duplicates(list(zip(xx, yy)))))
                        except AttributeError:
                            pass
                except KeyError:
                    pass
                for chol_polygon in chol_polygons:
                    for receptor_polygon in receptor_polygons:
                        polygon_intersection = intersect(receptor_polygon[1], chol_polygon[1])
                        if len(polygon_intersection) != 0:
                            video_found = True
                            print("video_found = True")
                            #Determine time offset
                            chol_time_mean = np.mean(chol_polygon[0].get_time())
                            receptor_time_mean = np.mean(receptor_polygon[0].get_time())

                            if receptor_time_mean > chol_time_mean:
                                receptor_time_offset = 0
                                chol_time_offset = np.mean(receptor_time_mean) - np.mean(chol_time_mean)
                            else:
                                receptor_time_offset =  np.mean(chol_time_mean) - np.mean(receptor_time_mean)
                                chol_time_offset = 0

                            #Prepare plot
                            plt.style.use('dark_background')
                            fig = plt.figure()
                            ax = fig.add_subplot(projection='3d')
                            ax.set_aspect('equal', adjustable='box')

                            #Prepare receptor dict
                            receptor_trajectory_dict = {}
                            receptor_trajectory_dict['colors'] = ['#04b900', '#025500']
                            receptor_trajectory_dict['dataframe'] = pd.DataFrame({
                                'x': trajectory.get_noisy_x(),
                                'y': trajectory.get_noisy_y(),
                                't': trajectory.get_time() + receptor_time_offset,
                            })
                            receptor_trajectory_dict['polygon'] = np.array(receptor_polygon[1] + [receptor_polygon[1][0]])

                            #Prepare chol dict
                            chol_trajectory_dict = {}
                            chol_trajectory_dict['colors'] = ['#ff0000', '#740000']
                            chol_trajectory_dict['dataframe'] = pd.DataFrame({
                                'x': aux_t.get_noisy_x(),
                                'y': aux_t.get_noisy_y(),
                                't': aux_t.get_time() + chol_time_offset,
                            })
                            chol_trajectory_dict['polygon'] = np.array(chol_polygon[1] + [chol_polygon[1][0]])

                            all_info = pd.concat([chol_trajectory_dict['dataframe'], receptor_trajectory_dict['dataframe']])
                            all_info = all_info.sort_values('t')

                            #Overlap point and spatial offset
                            intersection_centroid = np.mean(np.array(polygon_intersection), axis=0)

                            edge_offset = 0.15

                            # Keep trajectory points within times
                            tiempo_min = max(chol_trajectory_dict['dataframe']['t'].min(), receptor_trajectory_dict['dataframe']['t'].min())
                            tiempo_max = min(chol_trajectory_dict['dataframe']['t'].max(), receptor_trajectory_dict['dataframe']['t'].max())

                            chol_trajectory_dict['dataframe'] = chol_trajectory_dict['dataframe'][(chol_trajectory_dict['dataframe']['t'] >= tiempo_min) & (chol_trajectory_dict['dataframe']['t'] <= tiempo_max)]
                            receptor_trajectory_dict['dataframe'] = receptor_trajectory_dict['dataframe'][(receptor_trajectory_dict['dataframe']['t'] >= tiempo_min) & (receptor_trajectory_dict['dataframe']['t'] <= tiempo_max)]

                            #Get moment when both particles are near
                            dist = cdist(
                                receptor_trajectory_dict['dataframe'][['x','y','t']].values,
                                chol_trajectory_dict['dataframe'][['x','y','t']].values
                            )
                            minimum_value = np.min(dist)
                            receptor_index, chol_index = np.where(dist == minimum_value)
                            receptor_index, chol_index = receptor_index[0], chol_index[0]

                            reference_time = np.mean([
                                receptor_trajectory_dict['dataframe'].iloc[receptor_index].t,
                                chol_trajectory_dict['dataframe'].iloc[chol_index].t,
                            ])

                            frames = np.unique((sorted(chol_trajectory_dict['dataframe']['t'].tolist() + receptor_trajectory_dict['dataframe']['t'].tolist())))
                            
                            """
                            fig = plt.figure()
                            ax = fig.add_subplot()
                            x = np.linspace(np.min(frames),np.max(frames),1000)
                            plt.plot(x, smooth_square(x,reference_time,0.01)/3)
                            plt.show()
                            exit()
                            """

                            # Create points which camera follows
                            tck,u = interpolate.splprep([receptor_trajectory_dict['dataframe']['x'].tolist(),receptor_trajectory_dict['dataframe']['y'].tolist()], s=10)
                            u=np.linspace(0,1,num=len(frames),endpoint=True)
                            camera_track_for_receptor = interpolate.splev(u,tck)

                            tck,u = interpolate.splprep([receptor_trajectory_dict['dataframe']['x'].tolist(),receptor_trajectory_dict['dataframe']['y'].tolist()], s=100)
                            u=np.linspace(0,1,num=len(frames),endpoint=True)
                            angle_track_for_receptor = interpolate.splev(u,tck)

                            tck,u = interpolate.splprep([chol_trajectory_dict['dataframe']['x'].tolist(),chol_trajectory_dict['dataframe']['y'].tolist()], s=10)
                            u=np.linspace(0,1,num=len(frames),endpoint=True)
                            camera_track_for_chol = interpolate.splev(u,tck)

                            tck,u = interpolate.splprep([chol_trajectory_dict['dataframe']['x'].tolist(),chol_trajectory_dict['dataframe']['y'].tolist()], s=100)
                            u=np.linspace(0,1,num=len(frames),endpoint=True)
                            angle_track_for_chol = interpolate.splev(u,tck)

                            receptor_trajectory_dict['camera_track'] = pd.DataFrame({
                                'x': camera_track_for_receptor[0],
                                'y': camera_track_for_receptor[1],
                                't': frames,
                            })

                            receptor_trajectory_dict['angle_track'] = pd.DataFrame({
                                'x': angle_track_for_receptor[0],
                                'y': angle_track_for_receptor[1],
                                't': frames,
                            })

                            chol_trajectory_dict['camera_track'] = pd.DataFrame({
                                'x': camera_track_for_chol[0],
                                'y': camera_track_for_chol[1],
                                't': frames,
                            })

                            chol_trajectory_dict['angle_track'] = pd.DataFrame({
                                'x': angle_track_for_chol[0],
                                'y': angle_track_for_chol[1],
                                't': frames,
                            })

                            def update_one(time):
                                clear_ax(ax)

                                ax.set(
                                    xlim=[intersection_centroid[0] - edge_offset, intersection_centroid[0]+edge_offset], 
                                    ylim=[intersection_centroid[1] - edge_offset, intersection_centroid[1]+edge_offset],
                                )

                                ax.view_init(elev=45, azim=np.sin(time*2)*360)
                                ax.set_box_aspect(None, zoom=(smooth_square(time,reference_time,0.01)/50)+1)

                                ticks_number = 100
                                x_grid = np.linspace(intersection_centroid[0]-edge_offset*10, intersection_centroid[0]+edge_offset*10, ticks_number)
                                y_grid = np.linspace(intersection_centroid[1]-edge_offset*10, intersection_centroid[1]+edge_offset*10, ticks_number)
                                x_grid = np.repeat(np.expand_dims(x_grid,0), ticks_number, axis=0)
                                y_grid = np.repeat(np.expand_dims(y_grid,1), ticks_number, axis=1)
                                z_grid = np.zeros_like(x_grid)

                                ax.plot_wireframe(x_grid,y_grid,z_grid, rstride=2, cstride=2,color='white', alpha=0.25)

                                for trajectory_line_dict in [chol_trajectory_dict, receptor_trajectory_dict]:
                                    dataframe = trajectory_line_dict['dataframe']
                                    dataframe = dataframe[dataframe['t'] <= time]

                                    ax.plot(trajectory_line_dict['polygon'][:,0], trajectory_line_dict['polygon'][:,1],color=trajectory_line_dict['colors'][0], zorder=100+1)

                                    if len(dataframe) != 0:
                                        if len(dataframe) <= 3:
                                            ax.plot(dataframe['x'], dataframe['y'], color=trajectory_line_dict['colors'][0], linewidth=4)
                                        else:
                                            df_i_1 = dataframe[-3:]
                                            df_i_2 = dataframe[:-2]

                                            ax.plot(df_i_2['x'], df_i_2['y'], color=trajectory_line_dict['colors'][1], linewidth=4, zorder=100)[0]
                                            ax.plot(df_i_1['x'], df_i_1['y'], color=trajectory_line_dict['colors'][0], linewidth=4, zorder=100)[0]

                                        ax.scatter(dataframe[-1:]['x'].values, dataframe[-1:]['y'].values, color=trajectory_line_dict['colors'][0], s=100, zorder=100+1)

                                ax.set_title(f'{np.round(time, 3)}s')
                                ax.scatter(
                                    [intersection_centroid[0]],
                                    [intersection_centroid[1]],
                                    color='white',
                                    marker='+',
                                    zorder=100+1
                                )

                            def update_two(time):
                                clear_ax(ax)

                                ax.set(
                                    xlim=[intersection_centroid[0] - edge_offset, intersection_centroid[0]+edge_offset], 
                                    ylim=[intersection_centroid[1] - edge_offset, intersection_centroid[1]+edge_offset],
                                )

                                ax.set_box_aspect(None, zoom=(smooth_square(time,reference_time,0.01)/50)+1)

                                ticks_number = 100
                                x_grid = np.linspace(intersection_centroid[0]-edge_offset*10, intersection_centroid[0]+edge_offset*10, ticks_number)
                                y_grid = np.linspace(intersection_centroid[1]-edge_offset*10, intersection_centroid[1]+edge_offset*10, ticks_number)
                                x_grid = np.repeat(np.expand_dims(x_grid,0), ticks_number, axis=0)
                                y_grid = np.repeat(np.expand_dims(y_grid,1), ticks_number, axis=1)
                                z_grid = np.zeros_like(x_grid)

                                ax.plot_wireframe(x_grid,y_grid,z_grid, rstride=2, cstride=2,color='white', alpha=0.25)

                                for trajectory_line_dict in [chol_trajectory_dict, receptor_trajectory_dict]:
                                    dataframe = trajectory_line_dict['dataframe']
                                    dataframe = dataframe[dataframe['t'] <= time]
                                    camera_track = trajectory_line_dict['camera_track'][trajectory_line_dict['camera_track']['t'] <= time]
                                    angle_track = trajectory_line_dict['angle_track'][trajectory_line_dict['angle_track']['t'] <= time]

                                    ax.plot(trajectory_line_dict['polygon'][:,0], trajectory_line_dict['polygon'][:,1],color=trajectory_line_dict['colors'][0], zorder=100+1)

                                    if len(dataframe) != 0:
                                        if len(dataframe) <= 3:
                                            ax.plot(dataframe['x'], dataframe['y'], color=trajectory_line_dict['colors'][0], linewidth=4)
                                        else:
                                            df_i_1 = dataframe[-3:]
                                            df_i_2 = dataframe[:-2]

                                            ax.plot(df_i_2['x'], df_i_2['y'], color=trajectory_line_dict['colors'][1], linewidth=4, zorder=100)[0]
                                            ax.plot(df_i_1['x'], df_i_1['y'], color=trajectory_line_dict['colors'][0], linewidth=4, zorder=100)[0]

                                        if trajectory_line_dict == receptor_trajectory_dict:
                                            ax.set(
                                                xlim=[camera_track['x'].tolist()[-1]-edge_offset, camera_track['x'].tolist()[-1]+edge_offset], 
                                                ylim=[camera_track['y'].tolist()[-1]-edge_offset, camera_track['y'].tolist()[-1]+edge_offset],
                                            )
                                            try:
                                                my = angle_track['y'].tolist()[-1] - angle_track['y'].tolist()[-2]
                                                mx = angle_track['x'].tolist()[-1] - angle_track['x'].tolist()[-2]
                                            except:
                                                my = trajectory_line_dict['angle_track']['y'].tolist()[1] - trajectory_line_dict['angle_track']['y'].tolist()[0]
                                                mx = trajectory_line_dict['angle_track']['x'].tolist()[1] - trajectory_line_dict['angle_track']['x'].tolist()[0]

                                            m = my/mx
                                            ax.view_init(elev=45, azim=np.rad2deg(np.abs(np.arctan(m))))

                                            #ax.plot(trajectory_line_dict['camera_track']['x'], trajectory_line_dict['camera_track']['y'], linewidth=1, color='blue')
                                            #ax.plot(trajectory_line_dict['angle_track']['x'], trajectory_line_dict['angle_track']['y'], linewidth=1, color='yellow')

                                        ax.scatter(dataframe[-1:]['x'].values, dataframe[-1:]['y'].values, color=trajectory_line_dict['colors'][0], s=100, zorder=100+1)

                                ax.set_title(f'{np.round(time, 3)}s')
                                ax.scatter(
                                    [intersection_centroid[0]],
                                    [intersection_centroid[1]],
                                    color='white',
                                    marker='+',
                                    zorder=100+1
                                )

                            def update_three(time):
                                clear_ax(ax)

                                ax.set(
                                    xlim=[intersection_centroid[0] - edge_offset, intersection_centroid[0]+edge_offset], 
                                    ylim=[intersection_centroid[1] - edge_offset, intersection_centroid[1]+edge_offset],
                                )

                                ax.set_box_aspect(None, zoom=(smooth_square(time,reference_time,0.01)/50)+1)

                                ticks_number = 100
                                x_grid = np.linspace(intersection_centroid[0]-edge_offset*10, intersection_centroid[0]+edge_offset*10, ticks_number)
                                y_grid = np.linspace(intersection_centroid[1]-edge_offset*10, intersection_centroid[1]+edge_offset*10, ticks_number)
                                x_grid = np.repeat(np.expand_dims(x_grid,0), ticks_number, axis=0)
                                y_grid = np.repeat(np.expand_dims(y_grid,1), ticks_number, axis=1)
                                z_grid = np.zeros_like(x_grid)

                                ax.plot_wireframe(x_grid,y_grid,z_grid, rstride=2, cstride=2,color='white', alpha=0.25)

                                for trajectory_line_dict in [chol_trajectory_dict, receptor_trajectory_dict]:
                                    dataframe = trajectory_line_dict['dataframe']
                                    dataframe = dataframe[dataframe['t'] <= time]
                                    camera_track = trajectory_line_dict['camera_track'][trajectory_line_dict['camera_track']['t'] <= time]
                                    angle_track = trajectory_line_dict['angle_track'][trajectory_line_dict['angle_track']['t'] <= time]

                                    ax.plot(trajectory_line_dict['polygon'][:,0], trajectory_line_dict['polygon'][:,1],color=trajectory_line_dict['colors'][0], zorder=100+1)

                                    if len(dataframe) != 0:
                                        if len(dataframe) <= 3:
                                            ax.plot(dataframe['x'], dataframe['y'], color=trajectory_line_dict['colors'][0], linewidth=4)
                                        else:
                                            df_i_1 = dataframe[-3:]
                                            df_i_2 = dataframe[:-2]

                                            ax.plot(df_i_2['x'], df_i_2['y'], color=trajectory_line_dict['colors'][1], linewidth=4, zorder=100)[0]
                                            ax.plot(df_i_1['x'], df_i_1['y'], color=trajectory_line_dict['colors'][0], linewidth=4, zorder=100)[0]

                                        if trajectory_line_dict == chol_trajectory_dict:
                                            ax.set(
                                                xlim=[camera_track['x'].tolist()[-1]-edge_offset, camera_track['x'].tolist()[-1]+edge_offset], 
                                                ylim=[camera_track['y'].tolist()[-1]-edge_offset, camera_track['y'].tolist()[-1]+edge_offset],
                                            )
                                            try:
                                                my = angle_track['y'].tolist()[-1] - angle_track['y'].tolist()[-2]
                                                mx = angle_track['x'].tolist()[-1] - angle_track['x'].tolist()[-2]
                                            except:
                                                my = trajectory_line_dict['angle_track']['y'].tolist()[1] - trajectory_line_dict['angle_track']['y'].tolist()[0]
                                                mx = trajectory_line_dict['angle_track']['x'].tolist()[1] - trajectory_line_dict['angle_track']['x'].tolist()[0]

                                            m = my/mx
                                            ax.view_init(elev=45, azim=np.rad2deg(np.abs(np.arctan(m))))

                                            #ax.plot(trajectory_line_dict['camera_track']['x'], trajectory_line_dict['camera_track']['y'], linewidth=1, color='blue')
                                            #ax.plot(trajectory_line_dict['angle_track']['x'], trajectory_line_dict['angle_track']['y'], linewidth=1, color='yellow')

                                        ax.scatter(dataframe[-1:]['x'].values, dataframe[-1:]['y'].values, color=trajectory_line_dict['colors'][0], s=100, zorder=100+1)

                                ax.set_title(f'{np.round(time, 3)}s')
                                ax.scatter(
                                    [intersection_centroid[0]],
                                    [intersection_centroid[1]],
                                    color='white',
                                    marker='+',
                                    zorder=100+1
                                )

                            anim_one = animation.FuncAnimation(fig=fig, func=update_one, frames=frames)
                            anim_two = animation.FuncAnimation(fig=fig, func=update_two, frames=frames)
                            anim_three = animation.FuncAnimation(fig=fig, func=update_three, frames=frames)
                            for i, anim in enumerate([anim_one, anim_two, anim_three]):
                                anim.save(f'animation.gif', writer=animation.PillowWriter(fps=60), dpi=300)
                                clip = mp.VideoFileClip(f'animation.gif')
                                (w, h) = clip.size
                                clip = crop(clip, width=2300//2, height=2300//2, x_center=w/2+(50//2), y_center=h/2+(12//2))
                                clip.write_videofile(f'./animations/{file}_{trajectory.info["trajectory_id"]}_{plot_counter}_animation_{i}.mp4')

                            combined_clip = clips_array([[
                                mp.VideoFileClip(f'./animations/{file}_{trajectory.info["trajectory_id"]}_{plot_counter}_animation_0.mp4'),
                                mp.VideoFileClip(f'./animations/{file}_{trajectory.info["trajectory_id"]}_{plot_counter}_animation_1.mp4'),
                                mp.VideoFileClip(f'./animations/{file}_{trajectory.info["trajectory_id"]}_{plot_counter}_animation_2.mp4'),
                            ]])

                            combined_clip.write_videofile(f'./animations/{file}_{trajectory.info["trajectory_id"]}_{plot_counter}_animation.mp4')
                            plot_counter += 1

                        if video_found:
                            break
                    if video_found:
                        break
                if video_found:
                    break

DatabaseHandler.disconnect()
