import math

import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.stats import chi2, bootstrap
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import ruptures as rpt
from mongoengine import Document, FloatField, ListField, DictField, BooleanField
from scipy.spatial import ConvexHull
import scipy.stats as st
import matplotlib.animation as animation
from collections import defaultdict
import moviepy.editor as mp
from moviepy.video.fx.all import crop
from moviepy.editor import *

#Example about how to read trajectories from .mat
"""
from scipy.io import loadmat
from Trajectory import Trajectory
mat_data = loadmat('data/all_tracks_thunder_localizer.mat')
# Orden en la struct [BTX|mAb] [CDx|Control|CDx-Chol]
dataset = []
# Add each label and condition to the dataset
dataset.append({'label': 'BTX',
                'exp_cond': 'CDx',
                'tracks': mat_data['tracks'][0][0]})
dataset.append({'label': 'BTX',
                'exp_cond': 'Control',
                'tracks': mat_data['tracks'][0][1]})
dataset.append({'label': 'BTX',
                'exp_cond': 'CDx-Chol',
                'tracks': mat_data['tracks'][0][2]})
dataset.append({'label': 'mAb',
                'exp_cond': 'CDx',
                'tracks': mat_data['tracks'][1][0]})
dataset.append({'label': 'mAb',
                'exp_cond': 'Control',
                'tracks': mat_data['tracks'][1][1]})
dataset.append({'label': 'mAb',
                'exp_cond': 'CDx-Chol',
                'tracks': mat_data['tracks'][1][2]})
for data in dataset:
    trajectories = Trajectory.from_mat_dataset(data['tracks'], data['label'], data['exp_cond'])
    for trajectory in trajectories:
        if not trajectory.is_immobile(1.8):
            trajectory.save()
"""


def turning_angles(length, x, y, steps_lag=1, normalized=False):
    #if length/steps_lag <= 1:
    #    return []

    X = np.zeros((length,2))
    X[:,0] = x
    X[:,1] = y
    X = X[::steps_lag,:]
    length = X.shape[0]

    steps_lag = 1
    U_t = X[:length-steps_lag]
    U_t_plus_delta = X[np.arange(0, length-steps_lag)+steps_lag]

    V_t = U_t_plus_delta - U_t
    V_t_plus_delta = V_t[np.arange(0,len(V_t)-steps_lag)+steps_lag]
    V_t = V_t[:len(V_t)-steps_lag]

    A = np.sum((V_t_plus_delta * V_t), axis=1)
    B = np.linalg.norm(V_t, axis=1) * np.linalg.norm(V_t_plus_delta, axis=1)

    #Some B values could be 0. These are removed
    values_to_keep = np.where(B != 0)
    A = A[values_to_keep]
    B = B[values_to_keep]

    angles = np.clip(A/B, -1, 1)

    if not normalized:
        angles = np.rad2deg(np.arccos(angles))

    return angles.tolist()

"""
This method is a Array-Oriented Python implementation of a similar algorithm proposed in the referenced
paper to how direction change in time.

Taylor, R. W., Holler, C., Mahmoodabadi, R. G., Küppers, M., Dastjerdi, H. M., Zaburdaev, V., . . . Sandoghdar, V. (2020). 
High-Precision Protein-Tracking With Interferometric Scattering Microscopy. 
Frontiers in Cell and Developmental Biology, 8. 
https://doi.org/10.3389/fcell.2020.590158
"""
def directional_correlation(length, x, y, steps_lag=1, window_size=9):
    assert window_size % 2 == 1, 'Window size has to be odd'
    angles = turning_angles(length, x, y, steps_lag=steps_lag, normalized=True)
    convolution_result = np.convolve(angles, np.ones(window_size), mode='same')/window_size
    return convolution_result[window_size//2:-window_size//2]

def directional_correlation_segmentation(length, x, y, steps_lag=1, window_size=9, pen=1, jump=1, min_size=3, return_break_points=False):
    result = []
    signal = directional_correlation(length, x, y, window_size=window_size, steps_lag=steps_lag)

    break_points = rpt.Pelt(
        model='l2',
        jump=jump,
        min_size=min_size,
        ).fit_predict(
            signal,
            pen=pen
            )

    initial_index = 0
    for break_point in break_points:
        result.append(np.mean(signal[initial_index:break_point]))
        initial_index = break_point

    assert len(result) == len(break_points)

    if return_break_points:
        return result, break_points
    else:
        return result


class Trajectory(Document):
    x = ListField(required=True)
    y = ListField(required=False)
    z = ListField(required=False)

    t = ListField(required=True)

    noise_x = ListField(required=False)
    noise_y = ListField(required=False)
    noise_z = ListField(required=False)

    noisy = BooleanField(required=True)

    info = DictField(required=False)

    @classmethod
    def from_mat_dataset(cls, dataset, label='no label', experimental_condition='no experimental condition', scale_factor=1000): # With 1000 we convert trajectories steps to nm
        trajectories = []
        number_of_tracks = len(dataset)
        for i in range(number_of_tracks):
            raw_trajectory = dataset[i][0]

            trajectory_time = raw_trajectory[:, 0]
            trayectory_x = raw_trajectory[:, 1] * scale_factor
            trayectory_y = raw_trajectory[:, 2] * scale_factor

            trajectories.append(Trajectory(trayectory_x, trayectory_y, t=trajectory_time, info={"label": label, "experimental_condition": experimental_condition}, noisy=True))

        return trajectories

    @classmethod
    def ensemble_average_mean_square_displacement(cls, trajectories, number_of_points_for_msd=50, bin_width=None, alpha=0.95):

        trajectories = [trajectory for trajectory in trajectories if trajectory.length > number_of_points_for_msd + 1]
        #print("len average ->", np.mean([t.length for t in trajectories]))
        ea_msd_dict = defaultdict(lambda: [])
        mu_t_dict = defaultdict(lambda: [])

        for j_index, trajectory in enumerate(trajectories):
            positions = np.zeros((trajectory.length,2))
            positions[:,0] = trajectory.get_noisy_x()
            positions[:,1] = trajectory.get_noisy_y()
            time_position = trajectory.get_time()

            for index in range(0, number_of_points_for_msd):
                ea_msd_dict[int(time_position[index+1] - time_position[0]/bin_width)].append(np.linalg.norm(positions[index+1]-positions[0]) ** 2)
                mu_t_dict[int(time_position[index+1] - time_position[0]/bin_width)].append(np.linalg.norm(positions[index+1]-positions[0]))
        
        t_lag = np.sort(np.array([bin_width * i for i in ea_msd_dict]))

        ea_msd, mu_t = [], []

        for t in t_lag:
            ea_msd.append(np.mean(ea_msd_dict[t]))
            mu_t.append(np.mean(mu_t_dict[t]))

        ea_msd = np.array(ea_msd)
        mu_t = np.array(mu_t)

        alpha_1 = chi2.ppf(alpha/2, len(trajectories))
        alpha_2 = chi2.ppf(1-(alpha/2), len(trajectories))

        A = (ea_msd-(mu_t**2))*len(trajectories)

        intervals = [
            (A/alpha_1)+(mu_t**2),
            (A/alpha_2)+(mu_t**2)
        ]

        return t_lag, ea_msd, intervals

        """
        #print("len average ->", np.mean([t.length for t in trajectories]))
        ea_msd = defaultdict(lambda: [])

        delta = np.min(np.diff(trajectories[0].get_time())) if bin_width is None else bin_width

        for trajectory in trajectories:
            positions = np.zeros((trajectory.length,2))
            positions[:,0] = trajectory.get_noisy_x()
            positions[:,1] = trajectory.get_noisy_y()

            for index in range(1, trajectory.length):
                interval = trajectory.get_time()[index] - trajectory.get_time()[0]
                displacement = np.sum(np.abs((positions[index] - positions[0]) ** 2))
                ea_msd[int(interval/delta)].append(displacement)

        intervals = [[], []]

        for i in ea_msd:
            res = bootstrap(ea_msd[i], np.mean, n_resamples=len(trajectories), confidence_level=alpha, method='percentile')
            ea_msd[i] = np.mean(ea_msd[i])
            intervals[0].append(res.confidence_interval.low)
            intervals[1].append(res.confidence_interval.high)

        aux = np.array(sorted(list(zip(list(ea_msd.keys()), list(ea_msd.values()))), key=lambda x: x[0]))
        t_vec, ea_msd = (aux[:,0] * delta) + delta, aux[:,1]

        return t_vec, ea_msd, [np.array(intervals[0]), np.array(intervals[1])]
        """

    def __init__(self, x, y=None, z=None, model_category=None, noise_x=None, noise_y=None, noise_z=None, noisy=False, t=None, exponent=None, exponent_type='anomalous', info={}, **kwargs):

        if exponent_type == "anomalous":
            self.anomalous_exponent = exponent
        elif exponent_type == "hurst":
            self.anomalous_exponent = exponent * 2
        elif exponent_type is None:
            self.anomalous_exponent = None
        else:
            raise Exception(
                f"{exponent_type} exponent type is not available. Use 'anomalous' or 'hurst'.")

        self.model_category = model_category
        
        super().__init__(
            x=x,
            y=y,
            z=z,
            t=t,
            noise_x=noise_x,
            noise_y=noise_y,
            noise_z=noise_z,
            noisy=noisy,
            info=info,
            **kwargs
        )

    """
    Below method only works for bidimension trajectories
    """
    def reconstructed_trajectory(self, delta_t, with_noise=True):
        if with_noise:
            x = self.get_noisy_x()
            y = self.get_noisy_y()
        else:
            x = self.get_x()
            y = self.get_y()

        t = self.get_time()

        new_x = []
        new_y = []
        new_t = []

        for i in range(len(x)):
            t_right = t >= t[i]
            t_left = t <= t[i]+delta_t
            result = np.logical_and(t_right, t_left)

            if np.sum(result) >= 0:
                new_x.append(np.mean(x[result]))
                new_y.append(np.mean(y[result]))
                new_t.append(delta_t*i)

        return Trajectory(
            x = new_x,
            y = new_y,
            t = new_t,
            noisy=True
        )

    def get_anomalous_exponent(self):
        if self.anomalous_exponent is None:
            return "Not available"
        else:
            return self.anomalous_exponent

    def get_model_category(self):
        if self.model_category is None:
            return "Not available"
        else:
            return self.model_category

    @property
    def centroid(self):
        return np.array([np.mean(self.get_noisy_x()), np.mean(self.get_noisy_y())])

    @property
    def length(self):
        return len(self.x)

    def get_x(self):
        if self.x is None:
            raise Exception("x was not given")
        return np.copy(np.reshape(self.x, (len(self.x))))

    def get_y(self):
        if self.y is None:
            raise Exception("y was not given")
        return np.copy(np.reshape(self.y, (len(self.y))))
    
    """
    3D Methods are ignored
    def get_z(self):
        if self.z is None:
            raise Exception("y was not given")
        return np.copy(np.reshape(self.z, (len(self.z))))
    """

    @property
    def duration(self):
        return self.get_time()[-1] - self.get_time()[0]

    def get_time(self):
        if self.t is None:
            raise Exception("Time was not given")
        return np.copy(np.copy(np.reshape(self.t, (len(self.t)))))

    def get_noise_x(self):   
        return np.copy(self.noise_x)

    def get_noise_y(self):
        return np.copy(self.noise_y)

    def get_noise_z(self):
        return np.copy(self.noise_z)

    def get_noisy_x(self):   
        if self.noisy:
            return self.get_x()
        
        if self.noise_x is None:
            raise Exception('no x noise was provided')
        else:
            return self.get_x() + np.array(self.noise_x)

    def get_noisy_y(self):
        if self.noisy:
            return self.get_y()
        
        if self.noise_y is None:
            raise Exception('no y noise was provided')
        else:
            return self.get_y() + np.array(self.noise_y)

    """
    3D Methods are ignored
    def get_noisy_z(self):
        if self.noisy:
            return self.get_z()
        
        if self.noise_z is None:
            raise Exception('no z noise was provided')
        else:
            return self.get_z() + np.array(self.noise_z)
    """

    def displacements_on_x(self, with_noise=False):
        if with_noise:
            return np.diff(self.get_noisy_x())
        else:
            return np.diff(self.get_x())

    def displacements_on_y(self, with_noise=False):
        if with_noise:
            return np.diff(self.get_noisy_y())
        else:
            return np.diff(self.get_y())

    """
    3D Methods are ignored
    def displacements_on_z(self, with_noise=False):
        if with_noise:
            return np.diff(self.get_noisy_z())
        else:
            return np.diff(self.get_z())
    """

    def hurst_exponent(self):
        return self.anomalous_exponent / 2

    def plot(self, axis='xy'):
        plt.title(self)
        if axis == 'x':
            plt.plot(self.get_x(), marker="X")
            plt.plot(self.get_noisy_x(), marker="X")
        elif axis == 'y':
            plt.plot(self.get_y(), marker="X")
            plt.plot(self.get_noisy_y(), marker="X")        
        elif axis == 'xy':
            plt.plot(self.get_x(), self.get_y(), marker="X")
            plt.plot(self.get_noisy_x(), self.get_noisy_y(), marker="X")

        plt.show()

    def animate_plot(self, roi_size=None, save_animation=False, title='animation'):
        fig, ax = plt.subplots()

        line = ax.plot(self.get_noisy_x()[0], self.get_noisy_y()[0])[0]

        if roi_size is None:
            ax.set(xlim=[np.min(self.get_noisy_x()), np.max(self.get_noisy_x())], ylim=[np.min(self.get_noisy_y()), np.max(self.get_noisy_y())], xlabel='X', ylabel='Y')
        else:
            xlim = [np.min(self.get_noisy_x()), np.max(self.get_noisy_x())]
            ylim = [np.min(self.get_noisy_y()), np.max(self.get_noisy_y())]
            x_difference = xlim[1]-xlim[0]
            y_difference = ylim[1]-ylim[0]
            x_offset = (roi_size - x_difference)/2
            y_offset = (roi_size - y_difference)/2
            xlim = [xlim[0]-x_offset, xlim[1]+x_offset]
            ylim = [ylim[0]-y_offset, ylim[1]+y_offset]
            ax.set(xlim=xlim, ylim=ylim, xlabel='X', ylabel='Y')
        def update(frame):
            # for each frame, update the data stored on each artist.
            x_f = self.get_noisy_x()[:frame]
            y_f = self.get_noisy_y()[:frame]

            if self.t is not None:
                time = (self.get_time() - self.get_time()[0])[frame]
                time = np.round(time, 6)
                ax.set_title(f'{time}s')

            # update the scatter plot:
            #data = np.stack([x, y]).T
            # update the line plot:
            line.set_xdata(x_f[:frame])
            line.set_ydata(y_f[:frame])
            plt.tight_layout()
            return (line)

        ani = animation.FuncAnimation(fig=fig, func=update, frames=self.length, interval=1)

        if not save_animation:
            plt.show()
        else:
            ani.save(f'DELETE.gif', writer=animation.PillowWriter(fps=30), dpi=300)
            clip = mp.VideoFileClip(f'DELETE.gif')
            clip.write_videofile(f'./animations_plus/{title}.mp4')

    def plot_confinement_states(
        self,
        v_th=11,
        window_size=3,
        transition_fix_threshold=9,
        non_confinement_color='black',
        confinement_color='green',
        show=True,
        alpha=1,
        plot_confinement_convex_hull=False,
        color_confinement_convex_hull='grey',
        alpha_confinement_convex_hull=0.5
    ):
        x = self.get_noisy_x().tolist()
        y = self.get_noisy_y().tolist()

        state_to_color = {1:confinement_color, 0:non_confinement_color}
        states_as_color = np.vectorize(state_to_color.get)(self.confinement_states(v_th=v_th, window_size=window_size, transition_fix_threshold=transition_fix_threshold))

        for i,(x1, x2, y1,y2) in enumerate(zip(x, x[1:], y, y[1:])):
            plt.plot([x1, x2], [y1, y2], states_as_color[i], alpha=alpha)

        confinement_sub_trajectories = self.sub_trajectories_trajectories_from_confinement_states(v_th=v_th, window_size=window_size, transition_fix_threshold=transition_fix_threshold)[1]

        if plot_confinement_convex_hull:
            for trajectory in confinement_sub_trajectories:
                points = np.zeros((trajectory.length, 2))
                points[:,0] = trajectory.get_noisy_x()
                points[:,1] = trajectory.get_noisy_y()
                hull = ConvexHull(points)

                plt.fill(points[hull.vertices, 0], points[hull.vertices, 1], color_confinement_convex_hull, alpha=alpha_confinement_convex_hull)
        
        if show:
            plt.show()

    def __str__(self):
        anomalous_exponent_string = "%.2f" % self.anomalous_exponent if self.anomalous_exponent is not None else None
        return f"Model: {self.model_category}, Anomalous Exponent: {anomalous_exponent_string}, Trajectory Length: {self.length}"

    @property
    def normalized_ratio(self):
        r = 0
        delta_r = []

        # Extract coordinate values from track
        data = np.zeros(shape=(2, self.length))
        data[0, :] = self.get_noisy_x()
        data[1, :] = self.get_noisy_y()

        for j in range(self.length):
            r = r + np.linalg.norm([data[0, j] - np.mean(data[0, :]), data[1, j] - np.mean(data[1, :])]) ** 2

        for j in range(self.length - 1):
            delta_r.append(np.linalg.norm([data[0, j + 1] - data[0, j], data[1, j + 1] - data[1, j]]))

        rad_gir = np.sqrt((1 / self.length) * r)
        mean_delta_r = np.mean(delta_r)
        criteria = (rad_gir / mean_delta_r) * np.sqrt(np.pi/2)
        return float(criteria)

    def is_immobile(self, threshold):
        return self.normalized_ratio <= threshold

    def sub_trajectories_trajectories_from_confinement_states(self, v_th=11, window_size=3, transition_fix_threshold=9, use_info=False, custom_states=None):
        if custom_states is None:
            confinement_states = self.confinement_states(return_intervals=False, v_th=v_th, transition_fix_threshold=transition_fix_threshold, window_size=window_size) if not use_info else self.info['analysis']['confinement-states']
        else:
            confinement_states = custom_states
        trajectories = {
            0: [],
            1: []
        }

        def divide_list(a_list):
            sublists = []
            current_sublist = []
            
            for element in a_list:
                if len(current_sublist) == 0 or element == current_sublist[0]:
                    current_sublist.append(element)
                else:
                    sublists.append(current_sublist)
                    current_sublist = [element]

            if len(current_sublist) > 0:
                sublists.append(current_sublist)

            return sublists

        index = 0

        for sublist in divide_list(confinement_states):
            trajectories[sublist[0]].append(self.build_noisy_subtrajectory_from_range(index, index+len(sublist)))
            index += len(sublist)
        
        return trajectories

    def build_noisy_subtrajectory_from_range(self, initial_index, final_index, noisy=True):
        new_trajectory = Trajectory(
                    x = self.get_noisy_x()[initial_index:final_index],
                    y = self.get_noisy_y()[initial_index:final_index],
                    t = self.get_time()[initial_index:final_index],
                    noisy=noisy
                )

        BTX_NOMENCLATURE = 'BTX680R'
        CHOL_NOMENCLATURE = 'fPEG-Chol'

        if 'dcr' in self.info:
            new_trajectory.info['dcr'] = self.info['dcr'][initial_index:final_index]
        if 'intensity' in self.info:
            new_trajectory.info['intensity'] = self.info['intensity'][initial_index:final_index]
        if 'dataset' in self.info:
            new_trajectory.info['dataset'] = self.info['dataset']
        if 'roi' in self.info:
            new_trajectory.info['roi'] = self.info['roi']
        if 'roi' in self.info:
            new_trajectory.info['roi'] = self.info['roi']
        if 'trajectory_id' in self.info:
            new_trajectory.info['trajectory_id'] = self.info['trajectory_id']
        if 'classified_experimental_condition' in self.info:
            new_trajectory.info['classified_experimental_condition'] = self.info['classified_experimental_condition']
        if f'{BTX_NOMENCLATURE}_single_intersections' in self.info:
            new_trajectory.info[f'{BTX_NOMENCLATURE}_single_intersections'] = self.info[f'{BTX_NOMENCLATURE}_single_intersections'][initial_index:final_index]
        if f'{CHOL_NOMENCLATURE}_single_intersections' in self.info:
            new_trajectory.info[f'{CHOL_NOMENCLATURE}_single_intersections'] = self.info[f'{CHOL_NOMENCLATURE}_single_intersections'][initial_index:final_index]
        if 'analysis' in self.info:
            new_trajectory.info['analysis'] = {}
            if 'confinement-states' in self.info['analysis']:
                new_trajectory.info['analysis']['confinement-states'] = self.info['analysis']['confinement-states'][initial_index:final_index]

        return new_trajectory

    def confinement_states(self,v_th=11, window_size=3, transition_fix_threshold=9, return_intervals=False):
        """
        This method is the Array-Oriented Python implementation of the algorithm proposed in the referenced
        paper to identify periods of transient confinement within individual trajectories.

        Sikora, G., Wyłomańska, A., Gajda, J., Solé, L., Akin, E. J., Tamkun, M. M., & Krapf, D. (2017).

        Elucidating distinct ion channel populations on the surface of hippocampal neurons via single-particle
        tracking recurrence analysis. Physical review. E, 96(6-1), 062404.
        https://doi.org/10.1103/PhysRevE.96.062404
        """
        if self.length == 1:
            if return_intervals:
                return [0], []
            else:
                return [0]

        C = self.length-1

        X = np.zeros((self.length,2))
        X[:,0] = self.get_noisy_x()
        X[:,1] = self.get_noisy_y()

        M = (X[:-1] + X[1:])/2
        R = np.linalg.norm(X[:-1] - X[1:], axis=1)/2

        S = scipy.sparse.lil_matrix(np.zeros((self.length, C)))

        for position_index in range(self.length):
            distances = scipy.spatial.distance_matrix(np.array([X[position_index]]), M)
            S[position_index, :] = (distances < R).astype(int)

        V = np.array(np.sum(S, axis=0))[0]
        V_convolved = np.convolve(V, np.ones(window_size))
        V = np.repeat(V_convolved[window_size-1::window_size], window_size)[:C]
        V = (V > v_th).astype(int)

        states = np.zeros(self.length)

        for position_index in range(self.length):
            states[position_index] = np.sum(S[position_index, :] * V)

        states = (states > 0).astype(int)

        #Spurious transitions are eliminated
        for window_index in range(0,len(states), transition_fix_threshold):
            states[window_index:window_index+transition_fix_threshold] = np.argmax(np.bincount(states[window_index:window_index+transition_fix_threshold]))

        if return_intervals:
            indices = np.nonzero(states[1:] != states[:-1])[0] + 1
            intervals = np.split(self.get_time(), indices)
            intervals = intervals[0::2] if states[0] else intervals[1::2]
            intervals = [interval for interval in intervals if interval[-1] - interval[0] != 0]

            return states, intervals
        else:
            return states

    def calculate_msd_curve(self, with_noise=True, bin_width=None, return_variances=False, limit_type=None, limit_value=None, time_start=None):
        """
        Code Obtained from https://github.com/Eggeling-Lab-Microscope-Software/TRAIT2D/blob/b51498b730140ffac5c0abfc5494ebfca25b445e/trait2d/analysis/__init__.py#L1061
        """
        if with_noise:
            x = self.get_noisy_x()
            y = self.get_noisy_y()
        else:
            x = self.get_x()
            y = self.get_y()

        N = len(x)
        assert N-3 > 0
        data_tmp = np.column_stack((x, y))
        data_t_tmp = self.get_time()

        msd_dict = defaultdict(lambda: [])
        msd_variances_dict = defaultdict(lambda: [])

        delta = np.min(np.diff(self.get_time())) if bin_width is None else bin_width

        for i in range(1,N-2):
            if limit_type == 'points' and (i-1) > limit_value:
                break
            if limit_type == 'time' and ((time_start is not None and time_start+((i-1)*delta) > limit_value) or (time_start is None and (i-1)*delta > limit_value)):
                break
            calc_tmp = np.sum(np.abs((data_tmp[i:N,:]-data_tmp[0:N-i,:]) ** 2), axis=1)
            calc_t_tmp = data_t_tmp[i:N] - data_t_tmp[0:N-i]
            #plt.scatter(calc_t_tmp, calc_tmp, color='blue', s=0.1)
            for interval, square_displacement in zip(calc_t_tmp, calc_tmp):
                if time_start is not None:
                    if time_start < interval:
                        msd_dict[int((interval-time_start)/delta)+1].append(square_displacement)
                else:
                    msd_dict[int(interval/delta)].append(square_displacement)

        for i in msd_dict:
            msd_variances_dict[i] = np.var(msd_dict[i])
            msd_dict[i] = np.mean(msd_dict[i])

        if time_start is not None:
            time_msd = [[time_start+(bin_width*(t-(1/2))), msd_dict[t]] for t in msd_dict]
        else:
            time_msd = [[bin_width*t, msd_dict[t]] for t in msd_dict]

        aux = np.array(sorted(time_msd, key=lambda x: x[0]))
        t_vec, msd = aux[:,0], aux[:,1]

        if time_start is not None:
            time_msd = [[time_start+(bin_width*(t-(1/2))), msd_variances_dict[t]] for t in msd_variances_dict]
        else:
            time_msd = [[bin_width*t, msd_variances_dict[t]] for t in msd_variances_dict]

        aux = np.array(sorted(time_msd, key=lambda x: x[0]))
        t_vec, msd_var = aux[:,0], aux[:,1]
        #plt.scatter(t_vec, np.zeros_like(t_vec))
        #plt.show()
        assert len(t_vec) == len(msd) == len(msd_var)

        if not return_variances:
            return t_vec, msd
        else:
            return t_vec, msd, msd_var

    def temporal_average_mean_squared_displacement(self, log_log_fit_limit=50, limit_type='points', with_noise=True, bin_width=None, time_start=None, with_corrections=False):
        t_vec, msd = self.calculate_msd_curve(with_noise=with_noise, bin_width=bin_width, limit_type=limit_type, limit_value=log_log_fit_limit, time_start=time_start)

        if limit_type == 'points':
            msd_fit = msd[0:log_log_fit_limit]
            t_vec_fit = t_vec[0:log_log_fit_limit]
            assert len(t_vec_fit) == log_log_fit_limit
            assert len(msd_fit) == log_log_fit_limit
        elif limit_type == 'time':
            msd_fit = msd[t_vec < log_log_fit_limit]
            t_vec_fit = t_vec[t_vec < log_log_fit_limit]
            enough_number_of_points = int((log_log_fit_limit/bin_width)*0.75)
            assert len(t_vec_fit) >= enough_number_of_points
            assert len(msd_fit) >= enough_number_of_points
        else:
            raise Exception(f'limit_type=={limit_type} is not accepted')

        if not with_corrections:
            def real_func(t, betha, k):
                return k * (t ** betha)

            def linear_func(t, betha, k):
                return np.log(k) + (np.log(t) * betha)

            popt, _ = curve_fit(real_func, t_vec_fit, msd_fit, bounds=((0, 0), (2, np.inf)), maxfev=2000)
            goodness_of_fit = r2_score(np.log(msd_fit), linear_func(t_vec_fit, popt[0], popt[1]))
            """
            fig, ax = plt.subplots(1,2)

            ax[0].set_title(f"betha={np.round(popt[0], 2)}, k={popt[1]}")
            ax[0].plot(t_vec_fit, real_func(t_vec_fit, popt[0], popt[1]), marker='X', color='black')
            ax[0].plot(t_vec_fit, msd_fit, marker='X', color='red')

            ax[1].set_title(f"betha={np.round(popt[0], 2)}, k={popt[1]}")
            ax[1].loglog(t_vec_fit, real_func(t_vec_fit, popt[0], popt[1]), marker='X', color='black')
            ax[1].loglog(t_vec_fit, msd_fit, marker='X', color='red')

            plt.show()
            """
            return t_vec, msd, popt[0], popt[1], goodness_of_fit
        else:
            R = 1/6
            DELTA_T = bin_width
            DIMENSION = 2

            def equation_anomalous(x, T, B, LOCALIZATION_PRECISION):
                TERM_1 = T*((x*DELTA_T)**(B-1))*2*DIMENSION*DELTA_T*x*(1-((2*R)/x))
                TERM_2 = 2*DIMENSION*(LOCALIZATION_PRECISION**2)
                return TERM_1 + TERM_2 

            def log_equation_anomalous(x, T, B, LOCALIZATION_PRECISION):
                return np.log10(equation_anomalous(10**x, T, B, LOCALIZATION_PRECISION))

            popt, _ = curve_fit(log_equation_anomalous, np.log10(t_vec_fit/DELTA_T), np.log10(msd_fit), bounds=((0, 0, 0), (np.inf, 2, np.inf)), maxfev=2000)
            goodness_of_fit = msd_fit - equation_anomalous(t_vec_fit/DELTA_T, popt[0], popt[1], popt[2])
            goodness_of_fit = np.sum(goodness_of_fit**2)/(len(t_vec_fit)-2)
            goodness_of_fit = np.sqrt(goodness_of_fit)
            """
            fig, ax = plt.subplots(1,2)
            print(goodness_of_fit) #1000 < goodness_of_fit * 1e6 fittings are ignored.
            ax[0].set_title(f"T={np.round(popt[0], 3)}, betha={np.round(popt[1], 2)}, loc_precision={np.round(popt[2], 3)}")
            ax[0].plot(t_vec_fit, equation_anomalous(t_vec_fit/DELTA_T, popt[0], popt[1], popt[2]), marker='X', color='black')
            ax[0].plot(t_vec_fit, msd_fit, marker='X', color='red')

            ax[1].set_title(f"T={np.round(popt[0], 3)}, betha={np.round(popt[1], 2)}, loc_precision={np.round(popt[2], 3)}")
            ax[1].loglog(t_vec_fit, equation_anomalous(t_vec_fit/DELTA_T, popt[0], popt[1], popt[2]), marker='X', color='black')
            ax[1].loglog(t_vec_fit, msd_fit, marker='X', color='red')

            plt.show()
            """
            return t_vec, msd, popt[0], popt[1], popt[2], goodness_of_fit

    def short_range_diffusion_coefficient_msd(self, with_noise=True, bin_width=None, time_start=None):
        def linear_func(t, d, sigma):
            return (4 * t * d) + (sigma**2)

        if with_noise:
            x = self.get_noisy_x()
            y = self.get_noisy_y()
        else:
            x = self.get_x()
            y = self.get_y()

        t_vec, msd = self.calculate_msd_curve(with_noise=with_noise, bin_width=bin_width, time_start=time_start)

        msd_fit = msd[1:4]
        t_vec_fit = t_vec[1:4]
        assert len(msd_fit) == 3
        assert len(t_vec_fit) == 3
        popt, _ = curve_fit(linear_func, t_vec_fit, msd_fit, bounds=((0, 0), (np.inf, np.inf)), maxfev=2000)
        goodness_of_fit = r2_score(msd_fit, linear_func(t_vec_fit, popt[0], popt[1]))

        return t_vec, msd, popt[0], popt[1], goodness_of_fit

    def turning_angles(self,steps_lag=1, normalized=False):
        return turning_angles(
            self.length,
            self.get_noisy_x(),
            self.get_noisy_y(),
            steps_lag=steps_lag,
            normalized=normalized
        )

    def directional_correlation(self, steps_lag=1, window_size=9):
        return directional_correlation(
            self.length,
            self.get_noisy_x(),
            self.get_noisy_y(),
            steps_lag=steps_lag,
            window_size=window_size
        )

    def directional_correlation_segmentation(self, steps_lag=1, window_size=9, pen=1, jump=1, min_size=3, return_break_points=False):
        return directional_correlation_segmentation(
            self.length,
            self.get_noisy_x(),
            self.get_noisy_y(),
            steps_lag=steps_lag,
            window_size=window_size,
            pen=pen,
            jump=jump,
            min_size=min_size,
            return_break_points=return_break_points
        )

    def mean_turning_angle(self):
        """
        This is meanDP in

        Deep learning assisted single particle tracking for
        automated correlation between diffusion and
        function
        """
        normalized_angles = turning_angles(
            self.length,
            self.get_noisy_x(),
            self.get_noisy_y(),
            normalized=True,
            steps_lag=1
        )
        return np.nanmean(normalized_angles)

    def correlated_turning_angle(self):
        """
        This is corrDP in

        Deep learning assisted single particle tracking for
        automated correlation between diffusion and
        function
        """
        normalized_angles = turning_angles(
            self.length,
            self.get_noisy_x(),
            self.get_noisy_y(),
            normalized=True,
            steps_lag=1
        )
        return np.nanmean(np.sign(normalized_angles[1:])==np.sign(normalized_angles[:-1]))

    def directional_persistance(self):
        """
        This is AvgSignDp in

        Deep learning assisted single particle tracking for
        automated correlation between diffusion and
        function
        """
        normalized_angles = turning_angles(
            self.length,
            self.get_noisy_x(),
            self.get_noisy_y(),
            normalized=True,
            steps_lag=1
        )
        return np.nanmean(np.sign(normalized_angles[1:])>0)

    def random_sample(self, roi_x, roi_y, in_place=False):
        """
        Only works with noisy trajectories
        """
        import math

        def rotate(origin, point, angle):
            """
            Rotate a point counterclockwise by a given angle around a given origin.

            The angle should be given in radians.
            """
            ox, oy = origin
            px, py = point

            qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
            qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
            return qx, qy


        x = self.get_noisy_x()
        y = self.get_noisy_y()

        x -= np.mean(x)
        y -= np.mean(y)

        #plt.plot(x, y)
        rotation_angle = np.random.uniform(0,180)
        rotated_points = np.array([rotate([0,0], point, math.radians(rotation_angle)) for point in zip(x, y)])
        rotated_x, rotated_y = rotated_points[:,0] * np.random.choice([1,-1]), rotated_points[:,1] * np.random.choice([1,-1])
        new_x, new_y = rotated_x, rotated_y
        #plt.plot(rotated_x, rotated_y)
        #plt.show()

        def is_correct_offset(i_x,i_y):
            condition_a = roi_x[0] < min(i_x) and max(i_x) < roi_x[1]
            condition_b = roi_y[0] < min(i_y) and max(i_y) < roi_y[1]
            return condition_a and condition_b

        while not is_correct_offset(new_x,new_y):
            offset_x = np.random.uniform(low=min(roi_x), high=max(roi_x))
            offset_y = np.random.uniform(low=min(roi_y), high=max(roi_y))
            new_x = rotated_x + offset_x
            new_y = rotated_y + offset_y

        #plt.plot(new_x, new_y)
        #plt.xlim(roi_x)
        #plt.ylim(roi_y)
        #plt.show()

        if in_place:
            self.x = new_x.tolist()
            self.y = new_y.tolist()
        else:
            return Trajectory(
                x=new_x.tolist(),
                y=new_y.tolist(),
                t=self.t,
                info=self.info,
                noisy=True
            )

    def copy(self):
        """
        Only works with noisy trajectories
        """
        return Trajectory(
            x=self.get_noisy_x().tolist(),
            y=self.get_noisy_y().tolist(),
            t=self.get_time(),
            info=self.info,
            noisy=True
        )
