import numpy as np
from sklearn.neighbors import KDTree


def rearange_mesh_faces(faces):
    faces = faces[np.arange(len(faces)) % 4 != 0]
    faces = faces.reshape(-1, 3)
    return faces

def calc_dk1_1(points, v1, k1):
    delta = 0.1
    output = np.zeros((len(points), 1))
    A = KDTree(points)
    for i in range(len(points)):
        p = points[i]
        p_delta = p + delta * v1[i]
        p_delta = p_delta[np.newaxis, :]
        p_minus_delta = p - delta * v1[i]
        p_minus_delta = p_minus_delta[np.newaxis, :]
        # calculate closest point to p_delta using kd-tree
        closest_point_p_delta = A.query(p_delta,k=2)
        dist_p_delta = closest_point_p_delta[0][0][0]
        p_delta = closest_point_p_delta[1][0][0]
        # calculate closest point to p_minus_delta
        closest_point_p_minus_delta = A.query(p_minus_delta, k=2)
        dist_p_minus_delta = closest_point_p_minus_delta[0][0][0]
        p_minus_delta = closest_point_p_minus_delta[1][0][0]
        # check if the closest point is the same as the point we are calculating the curvature for
        if closest_point_p_delta[1][0][0] == i:
            # dist_p_delta = closest_point_p_delta[0][0][1]
            p_delta = closest_point_p_delta[1][0][1]
        if closest_point_p_minus_delta[1][0][0] == i:
            # dist_p_minus_delta = closest_point_p_minus_delta[0][0][1]
            p_minus_delta = closest_point_p_minus_delta[1][0][1]



        output[i] = (k1[p_delta] - k1[p_minus_delta]) / (np.linalg.norm(points[p_delta] - points[p_minus_delta]))

    return output
