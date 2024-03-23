import numpy as np
from Trajectory import Trajectory
from scipy.spatial import KDTree
from shapely.geometry import MultiPoint
import scipy
import numpy as np
import scipy
import pandas as pd
import EntropyHub as EH


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

def get_list_of_values_of_analysis_field(filter_query, field_name):
    values = [document['info'].get('analysis', {}).get(field_name, None) for document in Trajectory._get_collection().find(filter_query, {f'info.analysis.{field_name}':1})]
    values = [value for value in values if value is not None]
    return values

def get_list_of_values_of_field(filter_query, field_name):
    values = [document['info'].get(field_name, None) for document in Trajectory._get_collection().find(filter_query, {f'info.{field_name}':1})]
    values = [value for value in values if value is not None]
    return values

def get_list_of_analysis_field(filter_query, field_name):
    list_of_list = [document['info'].get('analysis', {}).get(field_name, []) for document in Trajectory._get_collection().find(filter_query, {f'info.{field_name}':1})]
    list_of_list = [a_list for a_list in list_of_list if len(a_list) != 0]
    return list_of_list

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

def both_trajectories_intersect(trajectory_one, trajectory_two, via='kd-tree', radius_threshold=1, return_kd_tree_intersections=False):
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
    """
    """
    if via=='hull':
        t_one = MultiPoint([point for point in zip(trajectory_one.get_noisy_x(), trajectory_one.get_noisy_y())]).convex_hull
        t_two = MultiPoint([point for point in zip(trajectory_two.get_noisy_x(), trajectory_two.get_noisy_y())]).convex_hull
        return t_one.intersects(t_two)
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
    """
    if len(vector) > 2:
        Ap, Phi = EH.ApEn(vector, 2, 1)
        Ef = Ap[-1]
    else:
        Ef = 0
    """
    Ef = 0
    Rf = np.sqrt(np.mean(vector**2))

    return [Mf, MDf, SDf, Sf, Kf, Ef, Rf]

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
    expected_q = 1+(57*(array.shape[-1]-1))
    features = np.zeros((array.shape[0], expected_q))
    delta = 3

    for trajectory_index in range(array.shape[0]):
        T_max = array.shape[1]

        q = 0

        features[trajectory_index, q] = np.log(T_max)

        q += 1
        
        for dimension_index in range(array.shape[-1]-1):
            xn = array[trajectory_index, :, dimension_index]#normalize_vector(array[trajectory_index, :, dimension_index])

            #Displacements
            v = np.diff(xn)
            Mf, MDf, SDf, Sf, Kf, Ef, Rf = average_statistics(v)
            new_statistics = [Mf, MDf, Sf, Kf, Ef]
            features[trajectory_index, q:q+len(new_statistics)] = new_statistics
            q += len(new_statistics)

            #Absolute displacements
            f = np.abs(v)
            Mf, MDf, SDf, Sf, Kf, Ef, Rf = average_statistics(f)
            new_statistics = [Mf, MDf, SDf, Sf, Kf, Ef]
            features[trajectory_index, q:q+len(new_statistics)] = new_statistics
            q += len(new_statistics)

            #Absolute displacements -> Sampled every step_offset time steps
            for step_offset in [2,3,4,5,6,7,8,9]:
                v_i = np.abs(xn[step_offset:] - xn[:-step_offset])
                Mf, MDf, SDf, Sf, Kf, Ef, Rf = average_statistics(v_i)

                if step_offset < 8:
                    new_statistics = [Mf, MDf, SDf, Sf, Kf, Ef]
                else:
                    new_statistics = [Mf, MDf, SDf, Sf, Kf]

                features[trajectory_index, q:q+len(new_statistics)] = new_statistics
                q += len(new_statistics)

            """
            #Displacement relative change
            f = v[1:]/v[:-1]
            Mf, MDf, SDf, Sf, Kf, Ef, Rf = average_statistics(f)
            new_statistics = [MDf]
            features[trajectory_index, q:q+len(new_statistics)] = new_statistics
            q += len(new_statistics)

            #Fourier Transform
            fv = np.abs(np.fft.fftshift(np.fft.fft(v)))
            f = fv[int(np.ceil(T_max/2)):] / T_max
            Mf, MDf, SDf, Sf, Kf, Ef, Rf = average_statistics(f)
            new_statistics = [Mf, MDf, SDf, Sf, Kf, Ef]
            features[trajectory_index, q:q+len(new_statistics)] = new_statistics
            q += len(new_statistics)

            #Normalized power spectral density (PSD)
            f = (fv[int(np.ceil(T_max/2)):]**2) / (T_max**2)
            h = np.ones(len(f))
            h[:int(np.round(T_max/4))] = -1
            f = f * h
            Pf = np.sum(f)
            new_statistics = [Pf]
            features[trajectory_index, q:q+len(new_statistics)] = new_statistics
            q += len(new_statistics)

            Pt = np.zeros(int(np.ceil(T_max/delta))-1)
            for m in range(0, len(Pt)):
                index = m*delta
                f = v[int(index):int(index+delta)]
                fv_time = np.abs(np.fft.fftshift(np.fft.fft(f)))
                fv_time = (fv_time[int(np.ceil(delta/2))-1:]**2) / (delta**2)
                Pt[m] = np.sum(fv_time)

            h = np.ones_like(Pt)
            h[:int(np.round(Pt.shape[0]/2))] = -1
            dPf = np.sum(Pt * h)
            new_statistics = [dPf]
            features[trajectory_index, q:q+len(new_statistics)] = new_statistics
            q += len(new_statistics)

            f = np.abs(Pt)
            Mf, MDf, SDf, Sf, Kf, Ef, Rf = average_statistics(f)
            new_statistics = [Mf, MDf, SDf, Sf, Kf, Ef]
            features[trajectory_index, q:q+len(new_statistics)] = new_statistics
            q += len(new_statistics)

            #Signal rate of variation
            LT = ischange(xn, 'variance', threshold=20, window_size=3)
            LT = np.sum(LT==0) / T_max
            new_statistics = [LT]
            features[trajectory_index, q:q+len(new_statistics)] = new_statistics
            q += 1

            MSD = np.zeros(int(np.floor(T_max/2)))
            for n in range(0, len(MSD)):
                MSD[n] = np.mean((xn[1+n:] - xn[:-1-n])**2)

            t = np.arange(1, np.floor(T_max/2).astype(int))
            v2 = (MSD[1:] - MSD[:-1])/t
            f = np.abs(v2)
            Mf, MDf, SDf, Sf, Kf, Ef, Rf = average_statistics(f)
            new_statistics = [Mf, MDf, SDf, Sf, Kf, Ef]
            features[trajectory_index, q:q+len(new_statistics)] = new_statistics
            q += len(new_statistics)

            #Autocorrelation function of the displacement 
            rv = np.correlate(v, v, mode='full') / T_max
            new_statistics = [np.sum(rv[T_max - 1 + np.arange(delta)])]
            features[trajectory_index, q:q+len(new_statistics)] = new_statistics
            q += len(new_statistics)

            #Wavelet transform
            wt = scipy.signal.cwt(v, scipy.signal.morlet, widths=np.arange(1, 3))

            f = np.abs(wt[0])
            Mf, MDf, SDf, Sf, Kf, Ef, Rf = average_statistics(f)
            new_statistics = [Mf, MDf, SDf, Sf, Kf, Ef]
            features[trajectory_index, q:q+len(new_statistics)] = new_statistics
            q += len(new_statistics)

            f = np.abs(wt[1])
            Mf, MDf, SDf, Sf, Kf, Ef, Rf = average_statistics(f)
            new_statistics = [Mf, MDf, SDf, Sf, Kf, Ef]
            features[trajectory_index, q:q+len(new_statistics)] = new_statistics
            q += len(new_statistics)
            """

        assert q == expected_q, f"{q} == {expected_q}"

    return features
