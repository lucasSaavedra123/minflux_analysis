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
