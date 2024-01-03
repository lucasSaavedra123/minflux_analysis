import numpy as np
from Trajectory import Trajectory
from scipy.spatial import KDTree
from shapely.geometry import MultiPoint, Point


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
