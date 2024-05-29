import numpy as np
from Trajectory import Trajectory
from scipy.spatial import KDTree
from shapely.geometry import MultiPoint, LinearRing
import scipy
import numpy as np
import scipy
import pandas as pd
import EntropyHub as EH
from collections import defaultdict
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint
from CONSTANTS import *
import math
from scipy.spatial import distance_matrix


def remove_outliers_from_set_of_values_of_column(list_of_values, return_upper_and_lower=False):
    filtered_list_of_values = list_of_values[~np.isnan(list_of_values)]
    Q1 = np.percentile(filtered_list_of_values, 25, method='midpoint')
    Q3 = np.percentile(filtered_list_of_values, 75, method='midpoint')
    IQR = Q3 - Q1

    upper = Q3+1.5*IQR
    lower = Q1-1.5*IQR

    for i in range(len(list_of_values)):
        if list_of_values[i] is not None and not (lower <= list_of_values[i] <= upper):
            list_of_values[i] = None

    return list_of_values

def get_dataframe_of_trajectory_analysis_data(a_query):
    p = {
        'info.file':1,
        'info.roi':1,
        'info.analysis.confinement-states': 1,
        't': 1,
        f'info.number_of_confinement_zones_with_{CHOL_NOMENCLATURE}': 1,
        f'info.number_of_confinement_zones_with_{BTX_NOMENCLATURE}': 1,
        'info.number_of_confinement_zones': 1
    }

    dataframe = {}
    fields = ['k', 'betha', 'd_2_4', 'localization_precision', 'goodness_of_fit', 'meanDP', 'corrDP', 'AvgSignD', 'residence_time', 'inverse_residence_time']

    for field in fields:
        p[f'info.analysis.{field}'] = 1
        dataframe[field] = []

    dataframe['file'] = []
    dataframe['roi'] = []
    dataframe['change_rate'] = []
    dataframe[f'number_of_confinement_zones_with_{CHOL_NOMENCLATURE}'] = []
    dataframe[f'number_of_confinement_zones_with_{BTX_NOMENCLATURE}'] = []
    dataframe[f'number_of_confinement_zones'] = []

    documents = Trajectory._get_collection().find(a_query, p)

    for d in documents:
        if 'analysis' not in d['info']:
            continue
        dataframe['file'].append(d['info']['file'])
        dataframe['roi'].append(d['info']['roi'])
        for field in fields:
            try:
                dataframe[field].append(d['info']['analysis'][field])
            except KeyError:
                dataframe[field].append(None)

        if 'confinement-states' in d['info']['analysis']:
            rate = np.abs(np.diff(d['info']['analysis']['confinement-states'])!=0).sum() / (d['t'][-1] - d['t'][0])
            dataframe['change_rate'].append(rate)
        else:
            dataframe['change_rate'].append(None)

        dataframe[f'number_of_confinement_zones_with_{CHOL_NOMENCLATURE}'].append(0)
        dataframe[f'number_of_confinement_zones_with_{BTX_NOMENCLATURE}'].append(0)
        dataframe[f'number_of_confinement_zones'].append(0)

        if f'number_of_confinement_zones_with_{CHOL_NOMENCLATURE}' in d['info']:
            dataframe[f'number_of_confinement_zones_with_{CHOL_NOMENCLATURE}'][-1] = d['info'][f'number_of_confinement_zones_with_{CHOL_NOMENCLATURE}']

        if f'number_of_confinement_zones_with_{BTX_NOMENCLATURE}' in d['info']:
            dataframe[f'number_of_confinement_zones_with_{BTX_NOMENCLATURE}'][-1] = d['info'][f'number_of_confinement_zones_with_{BTX_NOMENCLATURE}']

        if f'number_of_confinement_zones' in d['info']:
            dataframe[f'number_of_confinement_zones'][-1] = d['info'][f'number_of_confinement_zones']

    dataframe = pd.DataFrame(dataframe)

    rows_to_eliminate = dataframe['goodness_of_fit'] < 0.8
    dataframe[rows_to_eliminate]['k'] = None
    dataframe[rows_to_eliminate]['betha'] = None
    dataframe[rows_to_eliminate]['goodness_of_fit'] = None
    return dataframe

def get_dataframe_of_portions_analysis_data(a_query):
    p = {'info.file':1,'info.roi':1}

    confinement_dataframe, non_confinement_dataframe = {}, {}
    confinement_fields = ['confinement-a', 'confinement-e','confinement-area', 'confinement-steps', 'confinement-betha', 'confinement-k', 'confinement-d_2_4', 'confinement-duration', 'confinement-goodness_of_fit']
    non_confinement_fields = ['non-confinement-steps', 'non-confinement-betha', 'non-confinement-k', 'non-confinement-d_2_4', 'non-confinement-duration', 'non-confinement-goodness_of_fit']

    confinement_dataframe['roi'], confinement_dataframe['file'] = [], []
    for field in confinement_fields:
        p[f'info.analysis.{field}'] = 1
        confinement_dataframe[field] = []

    non_confinement_dataframe['roi'], non_confinement_dataframe['file'] = [], []
    for field in non_confinement_fields:
        p[f'info.analysis.{field}'] = 1
        non_confinement_dataframe[field] = []

    documents = Trajectory._get_collection().find(a_query, p)

    for d in documents:
        if 'analysis' not in d['info']:
            continue
        confinement_values = [d['info']['analysis'][field] for field in confinement_fields]
        confinement_values = zip(*confinement_values)

        for row in confinement_values:
            confinement_dataframe['file'].append(d['info']['file'])
            confinement_dataframe['roi'].append(d['info']['roi'])

            for field, value in zip(confinement_fields, row):
                confinement_dataframe[field].append(value)

        non_confinement_values = [d['info']['analysis'][field] for field in non_confinement_fields]
        non_confinement_values = zip(*non_confinement_values)

        for row in non_confinement_values:
            non_confinement_dataframe['file'].append(d['info']['file'])
            non_confinement_dataframe['roi'].append(d['info']['roi'])

            for field, value in zip(non_confinement_fields, row):
                non_confinement_dataframe[field].append(value)

    confinement_dataframe, non_confinement_dataframe = pd.DataFrame(confinement_dataframe), pd.DataFrame(non_confinement_dataframe)

    rows_to_eliminate = confinement_dataframe['confinement-goodness_of_fit'] < 0.8
    confinement_dataframe[rows_to_eliminate]['confinement-k'] = None
    confinement_dataframe[rows_to_eliminate]['confinement-betha'] = None
    confinement_dataframe[rows_to_eliminate]['confinement-goodness_of_fit'] = None

    rows_to_eliminate = non_confinement_dataframe['non-confinement-goodness_of_fit'] < 0.8
    non_confinement_dataframe[rows_to_eliminate]['non-confinement-k'] = None
    non_confinement_dataframe[rows_to_eliminate]['non-confinement-betha'] = None
    non_confinement_dataframe[rows_to_eliminate]['non-confinement-goodness_of_fit'] = None

    return confinement_dataframe, non_confinement_dataframe

def logarithmic_sampling(N, k):
    # Calcula los intervalos logarítmicos
    intervals = [math.exp(i * math.log(N) / (k - 1)) for i in range(k)]
    # Muestrea los elementos más cercanos en la lista original
    samples = [round(interval) for interval in intervals]
    return samples

def custom_histogram(data, starting_x, final_x, x_step):
  bin_edges = [starting_x]
  frequency = []

  current_x = starting_x

  while current_x < final_x:
    left = data > current_x
    right = data < (current_x + x_step)

    frequency.append(np.sum(np.logical_and(left,right).astype(int)))
    current_x += x_step

    bin_edges.append(current_x)

  return frequency, bin_edges

def get_list_of_positions(filter_query):
    values = [np.vstack((document['x'], document['y'])) for document in Trajectory._get_collection().find(filter_query, {'x':1,'y':1})]
    values = [value for value in values if value.shape[1] > 1]
    return values

def get_list_of_values_of_field(filter_query, field_name, mean_by_roi=False):
    if not mean_by_roi:
        values = [document['info'].get(field_name, None) for document in Trajectory._get_collection().find(filter_query, {f'info.{field_name}':1})]
    else:
        documents = list(Trajectory._get_collection().find(filter_query, {f'info.{field_name}':1,'info.dataset':1,'info.file':1,'info.roi':1}))

        values_per_roi = defaultdict(lambda: [])
        for document in documents:
            value = document['info'].get(field_name, None)
            if value is not None:
                values_per_roi[document['info']['dataset']+document['info']['file']+str(document['info']['roi'])].append(value)

        for alias in values_per_roi:
            values_per_roi[alias] = np.mean(values_per_roi[alias])   
        values = list(values_per_roi.values())
    values = [value for value in values if value is not None]
    return values

def get_list_of_main_field(filter_query, field_name):
    list_of_list = [document.get(field_name, []) for document in Trajectory._get_collection().find(filter_query, {f'{field_name}':1})]
    list_of_list = [a_list for a_list in list_of_list if len(a_list) != 0]
    return list_of_list

def get_ids_of_trayectories_under_betha_limits(filter_query, betha_min, betha_max):
    filter_query['info.analysis.betha'] = {'$gt': betha_min, '$lte': betha_max}
    list_of_list = [str(document['_id']) for document in Trajectory._get_collection().find(filter_query, {f'_id':1})]
    return list_of_list

def irregular_brownian_motion(length, D, dim=1, dt=None, lower=100e-6, upper=50e-3, scale=500e-6):
    n_traj = 1
    assert lower < scale < upper, f"{lower } < {scale} < {upper}"
    return_intervals = False

    if dt is None:
        return_intervals = True
        dt = scipy.stats.truncexpon(b=(upper-lower)/scale, loc=lower, scale=scale).rvs(n_traj*1*length)
        dt = dt.reshape((n_traj, 1, length))

    bm = (np.sqrt(2*D*dt)*np.random.randn(n_traj, dim, length)).cumsum(-1)
    dt = dt.cumsum(-1)

    if return_intervals:
        return (bm - bm[0, :, 0, None])[0], (dt - dt[0, :, 0, None])[0]
    else:
        return (bm - bm[0, :, 0, None])[0]

def irregular_fractional_brownian_motion(length, alpha, dim=1, dt=None, lower=100e-6, upper=50e-3, scale=500e-6):
    assert lower < scale < upper, f"{lower } < {scale} < {upper}"
    return_intervals = False

    if dt is None:
        ts = np.cumsum(scipy.stats.truncexpon(b=(upper-lower)/scale, loc=lower, scale=scale).rvs(1*1*length))
        return_intervals = True
    else:
        ts = np.linspace(0,dt,length)

    gamma = np.sqrt(np.pi)
    number_of_steps = len(ts)
    H = alpha/2
    wxs = np.zeros((number_of_steps))
    increments = np.zeros((dim, number_of_steps))

    for i in range(dim):
        phases = np.random.uniform(0, np.pi*2, size=48+8+1)

        for t_index, t in enumerate(ts):
            tStar = 2*np.pi*t/np.max(ts) #Check this line
            wx = 0

            for n in range(-8, 48+1):
                phasex = phases[n+np.abs(8)]
                wx += (np.cos(phasex)-np.cos(np.power(gamma, n) * tStar + phasex))/np.power(gamma, n*H)

            prevwx = wxs[t_index-2] if t_index-2>=0 else 0
            wxs[t_index-1] = wx
            increments[i, t_index-1] = wxs[t_index-1]-prevwx; 

    if return_intervals:
        return increments.cumsum(-1), ts
    else:
        return increments.cumsum(-1)

#GeelsForGeeks script: https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/
def both_segments_intersect(segment_one, segment_two):
    """
    This code is the algorithm written in:
    
    Introduction to Algorithms,
    Cormen, Leiserson, Rivest, Stein
    """

    class Point: 
        def __init__(self, x, y): 
            self.x = x 
            self.y = y 

    def onSegment(p, q, r): 
        if ( (q.x <= max(p.x, r.x)) and (q.x >= min(p.x, r.x)) and 
            (q.y <= max(p.y, r.y)) and (q.y >= min(p.y, r.y))): 
            return True
        return False
    
    def orientation(p, q, r): 
        val = (float(q.y - p.y) * (r.x - q.x)) - (float(q.x - p.x) * (r.y - q.y)) 
        if (val > 0): 
            return 1
        elif (val < 0): 
            return 2
        else: 
            return 0
    
    def doIntersect(p1,q1,p2,q2): 
        o1 = orientation(p1, q1, p2) 
        o2 = orientation(p1, q1, q2) 
        o3 = orientation(p2, q2, p1) 
        o4 = orientation(p2, q2, q1) 
    
        # General case 
        if ((o1 != o2) and (o3 != o4)): 
            return True
    
        # Special Cases 
        if ((o1 == 0) and onSegment(p1, p2, q1)): 
            return True
    
        if ((o2 == 0) and onSegment(p1, q2, q1)): 
            return True
    
        if ((o3 == 0) and onSegment(p2, p1, q2)): 
            return True
    
        if ((o4 == 0) and onSegment(p2, q1, q2)): 
            return True
    
        return False

    p1 = Point(segment_one[0][0], segment_one[0][1]) 
    q1 = Point(segment_one[1][0], segment_one[1][1]) 
    p2 = Point(segment_two[0][0], segment_two[0][1]) 
    q2 = Point(segment_two[1][0], segment_two[1][1]) 

    return doIntersect(p1, q1, p2, q2)

def both_trajectories_intersect(trajectory_one, trajectory_two, via='kd-tree', radius_threshold=1, return_kd_tree_intersections=False, grid_size=1):
    """
    points_one = np.column_stack((trajectory_one.get_noisy_x(),trajectory_one.get_noisy_y()))
    points_two = np.column_stack((trajectory_two.get_noisy_x(),trajectory_two.get_noisy_y()))

    segments_one = [points_one[i:i+2] for i in range(trajectory_one.length - 1)]
    segments_two = [points_two[i:i+2] for i in range(trajectory_two.length - 1)]

    for segment_one in segments_one:
        for segment_two in segments_two:
            if both_segments_intersect(segment_one, segment_two):
                return True
    """

    if via=='hull':
        t_one = MultiPoint([point for point in zip(trajectory_one.get_noisy_x(), trajectory_one.get_noisy_y())]).convex_hull
        t_two = MultiPoint([point for point in zip(trajectory_two.get_noisy_x(), trajectory_two.get_noisy_y())]).convex_hull
        return t_one.intersects(t_two)
    if via=='grid':
        X1 = np.zeros((trajectory_one.length, 2))
        X1[:,0] = trajectory_one.get_noisy_x()
        X1[:,1] = trajectory_one.get_noisy_y()

        X2 = np.zeros((trajectory_two.length, 2))
        X2[:,0] = trajectory_two.get_noisy_x()
        X2[:,1] = trajectory_two.get_noisy_y()

        x_min = min(np.min(X1[:,0]), np.min(X2[:,0]))
        x_max = max(np.max(X1[:,0]), np.max(X2[:,0]))

        y_min = min(np.min(X1[:,1]), np.min(X2[:,1]))
        y_max = max(np.max(X1[:,1]), np.max(X2[:,1]))

        x_space = np.arange(x_min, x_max, grid_size)
        y_space = np.arange(y_min, y_max, grid_size)

        for x_i in x_space:
            for y_i in y_space:
                localization_of_one = len(X1[((x_i<X1[:,0]) & (X1[:,0]<(x_i+grid_size))) & ((y_i<X1[:,1]) & (X1[:,1]<(y_i+grid_size)))])
                localization_of_two = len(X2[((x_i<X2[:,0]) & (X2[:,0]<(x_i+grid_size))) & ((y_i<X2[:,1]) & (X2[:,1]<(y_i+grid_size)))])
                if localization_of_one > 0 and localization_of_two > 0:
                    return True

        return False

    elif via=='ellipse':
        X1 = np.zeros((trajectory_one.length, 2))
        X1[:,0] = trajectory_one.get_noisy_x()
        X1[:,1] = trajectory_one.get_noisy_y()

        X2 = np.zeros((trajectory_two.length, 2))
        X2[:,0] = trajectory_two.get_noisy_x()
        X2[:,1] = trajectory_two.get_noisy_y()

        ellipses = [get_elliptical_information_of_data_points(X, return_full_description=True)[:5] for X in [X1,X2]]
        a, b = ellipse_polyline(ellipses)

        ea = LinearRing(a)
        eb = LinearRing(b)

        return ea.intersects(eb)
    elif via=='rectangle':
        class Rectangle:
            def __init__(self, min_x, max_x, min_y, max_y):
                self.top_right_point = [max_x, max_y]
                self.top_left_point = [min_x, max_y]
                self.bottom_right_point = [max_x, min_y]
                self.bottom_left_point = [min_x, min_y]

            def intersects(self, other):
                return not (self.top_right_point[0] < other.bottom_left_point[0]
                            or self.bottom_left_point[0] > self.top_right_point[0]
                            or self.top_right_point[1] < other.bottom_left_point[1]
                            or self.bottom_left_point[1] > self.top_right_point[1])


        X1 = np.zeros((trajectory_one.length, 2))
        X1[:,0] = trajectory_one.get_noisy_x()
        X1[:,1] = trajectory_one.get_noisy_y()

        rectangle_one = Rectangle(np.min(X1[:,0]), np.max(X1[:,0]), np.min(X1[:,1]), np.max(X1[:,1]))

        X2 = np.zeros((trajectory_two.length, 2))
        X2[:,0] = trajectory_two.get_noisy_x()
        X2[:,1] = trajectory_two.get_noisy_y()

        rectangle_two = Rectangle(np.min(X2[:,0]), np.max(X2[:,0]), np.min(X2[:,1]), np.max(X2[:,1]))
        return rectangle_one.intersects(rectangle_two)
    elif via=='brute-force':
        points_one = np.column_stack((trajectory_one.get_noisy_x(),trajectory_one.get_noisy_y()))
        points_two = np.column_stack((trajectory_two.get_noisy_x(),trajectory_two.get_noisy_y()))

        segments_one = [points_one[i:i+2] for i in range(trajectory_one.length - 1)]
        segments_two = [points_two[i:i+2] for i in range(trajectory_two.length - 1)]

        for segment_one in segments_one:
            for segment_two in segments_two:
                if both_segments_intersect(segment_one, segment_two):
                    return True
    elif via=='kd-tree':
        # Example curves represented as arrays of points (x, y)
        curve1 = np.column_stack((trajectory_one.get_noisy_x(),trajectory_one.get_noisy_y()))
        curve2 = np.column_stack((trajectory_two.get_noisy_x(),trajectory_two.get_noisy_y()))

        # Create KD-trees
        tree1 = KDTree(curve1)
        tree2 = KDTree(curve2)

        # Query for intersections between curves
        intersections = tree1.query_ball_tree(tree2, r=radius_threshold)

        if return_kd_tree_intersections:
            return len([intersection for intersection in intersections if intersection != []]) > 0, intersections
        else:
            return len([intersection for intersection in intersections if intersection != []]) > 0
    else:
        raise ValueError(f"via={via} is not correct")

"""
THE FOLLOWING CODE IS FROM https://github.com/LazyVinh/convex-polygon-intersection/tree/master
"""
def get_edges(polygon):
    """
    :param polygon: a list of points (point = list or tuple holding two numbers)
    :return: the edges of the polygon, i.e. all pairs of points
    """
    for i in range(len(polygon)):
        yield Edge(polygon[i - 1], polygon[i])

class Edge:
    def __init__(self, point_a, point_b):
        self._support_vector = np.array(point_a)
        self._direction_vector = np.subtract(point_b, point_a)

    def get_intersection_point(self, other):
        t = self._get_intersection_parameter(other)
        return None if t is None else self._get_point(t)

    def _get_point(self, parameter):
        return self._support_vector + parameter * self._direction_vector

    def _get_intersection_parameter(self, other):
        A = np.array([-self._direction_vector, other._direction_vector]).T
        if np.linalg.matrix_rank(A) < 2:
            return None
        b = np.subtract(self._support_vector, other._support_vector)
        x = np.linalg.solve(A, b)
        return x[0] if 0 <= x[0] <= 1 and 0 <= x[1] <= 1 else None

def intersect(polygon1, polygon2):
    """
    The given polygons must be convex and their vertices must be in anti-clockwise order (this is not checked!)

    Example: polygon1 = [[0,0], [0,1], [1,1]]

    """
    polygon3 = list()
    polygon3 += _get_vertices_lying_in_the_other_polygon(polygon1, polygon2)
    polygon3 += _get_edge_intersection_points(polygon1, polygon2)
    return sort_vertices_anti_clockwise_and_remove_duplicates(polygon3)


def _get_vertices_lying_in_the_other_polygon(polygon1, polygon2):
    vertices = list()
    vertices += [vertex for vertex in polygon1 if _polygon_contains_point(polygon2, vertex)]
    vertices += [vertex for vertex in polygon2 if _polygon_contains_point(polygon1, vertex)]
    return vertices


def _get_edge_intersection_points(polygon1, polygon2):
    intersection_points = list()
    for edge1 in get_edges(polygon1):
        for edge2 in get_edges(polygon2):
            intersection_point = edge1.get_intersection_point(edge2)
            if intersection_point is not None:
                intersection_points.append(intersection_point)
    return intersection_points


def _polygon_contains_point(polygon, point):
    for i in range(len(polygon)):
        a = np.subtract(polygon[i], polygon[i - 1])
        b = np.subtract(point, polygon[i - 1])
        if np.cross(a, b) < 0:
            return False
    return True


def sort_vertices_anti_clockwise_and_remove_duplicates(polygon, tolerance=1e-7):
    polygon = sorted(polygon, key=lambda p: _get_angle_in_radians(_get_bounding_box_midpoint(polygon), p))

    def vertex_not_similar_to_previous(_polygon, i):
        diff = np.subtract(_polygon[i - 1], _polygon[i])
        return np.linalg.norm(diff, np.inf) > tolerance

    return [p for i, p in enumerate(polygon) if vertex_not_similar_to_previous(polygon, i)]


def _get_angle_in_radians(point1, point2):
    return np.arctan2(point2[1] - point1[1], point2[0] - point1[0])


def _get_bounding_box_midpoint(polygon):
    x = [p[0] for p in polygon]
    y = [p[1] for p in polygon]
    return [(np.max(x) + np.min(x)) / 2., (np.max(y) + np.min(y)) / 2.]

"""
The next part of utils.py comes from paper:

Title: Characterization of anomalous diffusion classical statistics powered by deep learning (CONDOR)
Authors: Alessia Gentili and Giorgio Volpe
"""
def average_statistics(vector):
    Mf = np.mean(vector)
    MDf = np.median(vector)
    SDf = np.std(vector)
    Sf = scipy.stats.skew(vector)
    Kf = scipy.stats.kurtosis(vector)
    Rf = np.sqrt(np.mean(vector**2))

    return [Mf, MDf, SDf, Sf, Kf, Rf]

def normalize_vector(vector):
    vectorn = vector - np.mean(vector)
    #Delta_x = np.diff(vectorn)
    #normalization = np.std(Delta_x)
    #Delta_x = Delta_x/normalization
    #vectorn = np.append(0, np.cumsum(Delta_x))
    #vectorn = vectorn - np.mean(vectorn)
    return vectorn

def ischange(xn, method='variance', threshold=0, window_size=1):
    """
    Determine if there are significant changes in a vector using a specific method.

    Parameters:
    - xn: array_like, input vector.
    - method: str, method to detect changes ('variance' by default).
    - threshold: float, threshold to consider a significant change (default value: 0).
    - window_size: int, window size for calculating the metric (default value: 1).

    Return:
    - result: ndarray, boolean vector indicating the presence of significant changes.
    """
    if method == 'variance':
        xn_series = pd.Series(xn)
        rolling_var = xn_series.rolling(window=window_size).var()
        diff_var = np.abs(rolling_var.diff())
        result = diff_var > threshold
        return result.fillna(False).values
    else:
        raise ValueError(f"Method '{method}' is not valid.")

def transform_traj_into_features(array):
    expected_q = 60*(array.shape[-1]-1)
    features = np.zeros((array.shape[0], expected_q))

    for trajectory_index in range(array.shape[0]):
        T_max = array.shape[1]

        q = 0

        #features[trajectory_index, q] = np.log(T_max)

        #q += 1
        
        for dimension_index in range(array.shape[-1]-1):
            xn = normalize_vector(array[trajectory_index, :, dimension_index])

            #Displacements
            v = np.diff(xn)
            Mf, MDf, SDf, Sf, Kf, Rf = average_statistics(v)
            assert not np.any(np.isnan([Mf, MDf, SDf, Sf, Kf, Rf]))
            new_statistics = [Mf, MDf, SDf, Sf, Kf, Rf]
            features[trajectory_index, q:q+len(new_statistics)] = new_statistics
            q += len(new_statistics)

            #Absolute displacements
            f = np.abs(v)
            Mf, MDf, SDf, Sf, Kf, Rf = average_statistics(f)
            assert not np.any(np.isnan([Mf, MDf, SDf, Sf, Kf, Rf]))
            new_statistics = [Mf, MDf, SDf, Sf, Kf, Rf]
            features[trajectory_index, q:q+len(new_statistics)] = new_statistics
            q += len(new_statistics)

            #Absolute displacements -> Sampled every step_offset time steps
            for step_offset in [2,3,4,5,6,7,8,9]:
                v_i = np.abs(xn[step_offset:] - xn[:-step_offset])
                Mf, MDf, SDf, Sf, Kf, Rf = average_statistics(v_i)
                assert not np.any(np.isnan([Mf, MDf, SDf, Sf, Kf, Rf]))
                new_statistics = [Mf, MDf, SDf, Sf, Kf, Rf]

                features[trajectory_index, q:q+len(new_statistics)] = new_statistics
                q += len(new_statistics)

        assert q == expected_q, f"{q} == {expected_q}"

    return features

def equation_free(x, D, LOCALIZATION_PRECISION):
    TERM_1 = 2*DIMENSION*D*DELTA_T*(x-(2*R))
    TERM_2 = 2*DIMENSION*(LOCALIZATION_PRECISION**2)
    return TERM_1 + TERM_2

def equation_hop(x, DM, DU, L_HOP, LOCALIZATION_PRECISION):
    TERM_1_1_1 = (DU-DM)/DU
    TERM_1_1_2 = (L_HOP**2)/(6*DIMENSION*x*DELTA_T)
    TERM_1_1_3 = 1 - (np.exp(-((12*DU*x*DELTA_T)/(L_HOP**2))))

    TERM_1_1 = 2*DIMENSION*DELTA_T
    TERM_1_2 = DM + (TERM_1_1_1*TERM_1_1_2*TERM_1_1_3)
    TERM_1_3 = (x-(2*R))
    TERM_1 = TERM_1_1 * TERM_1_2 * TERM_1_3

    TERM_2 = 2*DIMENSION*(LOCALIZATION_PRECISION**2)
    return TERM_1 + TERM_2

def equation_confined(x, DU, L_HOP, LOCALIZATION_PRECISION):
    return equation_hop(x, 0, DU, L_HOP, LOCALIZATION_PRECISION)

def free_fitting(X,Y):
    select_indexes = np.unique(np.geomspace(1,len(X),len(X)//2).astype(int))-1
    X = X[select_indexes]
    Y = Y[select_indexes]

    def eq_4_obj_raw(x, y, d, delta): return np.sum((y - equation_free(x, d, delta))**2)
    #def eq_4_obj_raw(x, y, d, delta): return np.sum((y - equation_free(x, d, delta))**2)

    eq_4_obj = lambda coeffs: eq_4_obj_raw(X, Y, *coeffs)
    res_eq_4s = []

    for _ in range(100):        
        x0=[np.random.uniform(100, 100000), np.random.uniform(1, 100)]
        res_eq_4 = minimize(eq_4_obj, x0=x0, bounds=[(100, None), (1, None)])
        res_eq_4s.append(res_eq_4)

    return min(res_eq_4s, key=lambda r: r.fun)

def hop_fitting(X,Y):
    select_indexes = np.unique(np.geomspace(1,len(X),len(X)//2).astype(int))-1
    X = X[select_indexes]
    Y = Y[select_indexes]

    def eq_9_obj_raw(x, y, dm, du, l_hop, delta): return np.sum((y - equation_hop(x, dm, du, l_hop, delta))**2)
    eq_9_obj = lambda coeffs: eq_9_obj_raw(X, Y, *coeffs)
    res_eq_9s = []

    for _ in range(100):        
        x0=[np.random.uniform(100, 100000), np.random.uniform(100, 100000), np.random.uniform(1, 1000), np.random.uniform(1, 100)]
        res_eq_9 = minimize(eq_9_obj, x0=x0, bounds=[(100, None), (100, None), (1, None), (1, None)], constraints=[LinearConstraint([-1,1,0,0], lb=0, ub=np.inf)])
        res_eq_9s.append(res_eq_9)

    return min(res_eq_9s, key=lambda r: r.fun)

def confined_fitting(X,Y):
    select_indexes = np.unique(np.geomspace(1,len(X),len(X)//2).astype(int))-1
    X = X[select_indexes]
    Y = Y[select_indexes]

    def eq_9_obj_raw(x, y, du, l, delta): return np.sum((y - equation_confined(x, du, l, delta))**2)
    eq_9_obj = lambda coeffs: eq_9_obj_raw(X, Y, *coeffs)
    res_eq_9s = []

    for _ in range(100):        
        x0=[np.random.uniform(100, 100000), np.random.uniform(1, 1000), np.random.uniform(1, 100)]
        res_eq_9 = minimize(eq_9_obj, x0=x0, bounds=[(100, None), (1, None), (1, None)])
        res_eq_9s.append(res_eq_9)

    return min(res_eq_9s, key=lambda r: r.fun)

def extract_dataset_file_roi_file():
    file_id_and_roi_list = []
    a_file = open(FILE_AND_ROI_FILE_CACHE, 'r')
    for line in a_file.readlines():
        line = line.strip()
        line = line.split(',')
        file_id_and_roi_list.append([line[0], line[1],int(line[2])])
    a_file.close()
    return file_id_and_roi_list

def measure_overlap(trajectories_by_label, chol_confinement_to_chol_trajectory, chol_confinements):
    for btx_trajectory in trajectories_by_label[BTX_NOMENCLATURE]:
        btx_confinements = btx_trajectory.sub_trajectories_trajectories_from_confinement_states(v_th=33, transition_fix_threshold=5, use_info=True)[1]

        btx_trajectory.info['number_of_confinement_zones'] = len(btx_confinements)
        btx_trajectory.info[f'number_of_confinement_zones_with_{CHOL_NOMENCLATURE}'] = 0

        for btx_confinement in btx_confinements:
            already_overlap = False
            for chol_confinement in chol_confinements:
                if np.linalg.norm(chol_confinement.centroid-btx_confinement.centroid) < 1:#um
                    there_is_overlap = both_trajectories_intersect(chol_confinement, btx_confinement, via='hull')
                    btx_trajectory.info[f'number_of_confinement_zones_with_{CHOL_NOMENCLATURE}'] += 1 if there_is_overlap and not already_overlap else 0
                    chol_confinement_to_chol_trajectory[chol_confinement].info[f'number_of_confinement_zones_with_{BTX_NOMENCLATURE}'] += 1
                    if there_is_overlap:
                        already_overlap = True

def ellipse_polyline(ellipses, n=100):
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    st = np.sin(t)
    ct = np.cos(t)
    result = []
    for x0, y0, a, b, angle in ellipses:
        #angle = np.deg2rad(angle)
        sa = np.sin(angle)
        ca = np.cos(angle)
        p = np.empty((n, 2))
        p[:, 0] = x0 + a * ca * ct - b * sa * st
        p[:, 1] = y0 + a * sa * ct + b * ca * st
        result.append(p)
    return result

def intersections_between_ellipses(a, b):
    ea = LinearRing(a)
    eb = LinearRing(b)
    mp = ea.intersection(eb)

    x = [p.x for p in mp]
    y = [p.y for p in mp]
    return x, y

def get_elliptical_information_of_data_points(X, return_full_description=False):
    def cart2pol(numpy_point):
        rho = np.linalg.norm(numpy_point)
        phi = np.arctan2(numpy_point[1], numpy_point[0])
        return rho, phi

    #Displacement Process
    distances = distance_matrix(X, X)
    point_a_index, point_b_index = np.unravel_index(np.argmax(distances, axis=None), distances.shape)
    direction_vector = X[point_a_index] - X[point_b_index]

    middle_point = (X[point_a_index] + X[point_b_index])/2

    displacement = (direction_vector/2) + X[point_b_index]

    displaced_X = X - displacement

    #Rotation Process

    _, phi = cart2pol(direction_vector)

    if phi <= 0:
        direction_vector = displaced_X[point_b_index] - displaced_X[point_a_index]
        _, phi = cart2pol(direction_vector)

    rotation_matrix = np.array([
        [np.cos(np.pi - phi), -np.sin(np.pi - phi)],
        [np.sin(np.pi - phi), np.cos(np.pi - phi)]
    ])

    rotated_X = np.dot(rotation_matrix, X.T).T

    a = (np.max(rotated_X[:,0]) - np.min(rotated_X[:,0]))/2 # Semi-Major Axis
    b = (np.max(rotated_X[:,1]) - np.min(rotated_X[:,1]))/2 # Semi-Minor Axis
    e = np.sqrt(1-(np.power(b,2)/np.power(a,2)))#Eccentricity

    if not return_full_description:
        return a,b,e
    else:
        return float(middle_point[0]),float(middle_point[1]),a,b,phi,e
