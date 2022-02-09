import math
import sys

import numpy as np



def calc_TFL_dist(prev_container, curr_container, focal, pp):
    norm_prev_pts, norm_curr_pts, R, foe, tZ =  \
        prepare_3D_data(prev_container, curr_container, focal, pp)
    if(abs(tZ) < 10e-6):
        print('tz = ', tZ)
    elif norm_prev_pts.size == 0:
        print('no prev points')
    elif norm_prev_pts.size == 0:
        print('no curr points')
    else:
        curr_container.corresponding_ind, curr_container.traffic_lights_3d_location, \
        curr_container.valid = calc_3D_data(norm_prev_pts, norm_curr_pts, R, foe, tZ)
    return curr_container


def prepare_3D_data(prev_container, curr_container, focal, pp):
    norm_prev_pts = normalize(prev_container.traffic_light, focal, pp)
    norm_curr_pts = normalize(curr_container.traffic_light, focal, pp)
    R, foe, tZ = decompose(np.array(curr_container.EM))
    return norm_prev_pts, norm_curr_pts, R, foe, tZ


def calc_3D_data(norm_prev_pts, norm_curr_pts, R, foe, tZ):
    norm_rot_pts = rotate(norm_prev_pts, R)
    pts_3D = []
    corresponding_ind = []
    validVec = []
    for p_curr in norm_curr_pts:
        corresponding_p_ind, corresponding_p_rot = find_corresponding_points(p_curr, norm_rot_pts, foe)
        Z = calc_dist(p_curr, corresponding_p_rot, foe, tZ)
        valid = (Z > 0)
        if not valid:
            Z = 0
        validVec.append(valid)
        P = Z * np.array([p_curr[0], p_curr[1], 1])
        pts_3D.append((P[0], P[1], P[2]))
        corresponding_ind.append(corresponding_p_ind)
    return corresponding_ind, np.array(pts_3D), validVec


def normalize(pts, focal, pp):
    # transform pixels into normalized pixels using the focal length and principle point
    new_pts = [None] * len(pts)
    for i, point in enumerate(pts):
        new_pts[i] = [(point[0] - pp[0]) / focal, (point[1] - pp[1]) / focal, 1]
    return np.array(new_pts).reshape((len(pts), 3))


def unnormalize(pts, focal, pp):
    # transform normalized pixels into pixels using the focal length and principle point
    new_pts = [None] * len(pts)
    for i, point in enumerate(pts):
        new_pts[i] = [(point[0] * focal) + pp[0], (point[1] * focal) + pp[1], focal]
    return np.array(new_pts).reshape((len(pts), 3))


def decompose(EM):
    # extract R, foe and tZ from the Ego Motion
    R = EM[:3, :3]
    t = EM[:3, 3]
    foe = [t[0]/t[2], t[1]/t[2]]
    tZ = t[2]
    return R, foe, tZ


def rotate(pts, R):
    # rotate the points - pts using R
    for i, point in enumerate(pts):
        pts[i] = R.dot(point)
    return pts





def find_corresponding_points(p, norm_pts_rot, foe):
    # compute the epipolar line between p and foe
    # run over all norm_pts_rot and find the one closest to the epipolar line
    # return the closest point and its index
    minimum = sys.maxsize
    index = None
    coord = [-1, -1]
    norm_pts_rot = norm_pts_rot.tolist()
    m = (foe[1] - p[1]) / (foe[0] - p[0])
    n = (p[1] * foe[0] - foe[1] * p[0]) / (foe[0] - p[0])
    for point in norm_pts_rot:
        dist = abs((m * point[0] + n - point[1]) / math.sqrt(m ** 2 + 1))
        if dist < minimum:
            minimum = dist
            index = norm_pts_rot.index(point)
            coord = point
    return index, coord



def calc_dist(p_curr, p_rot, foe, tZ):
    # calculate the distance of p_curr using x_curr, x_rot, foe_x and tZ
    # calculate the distance of p_curr using y_curr, y_rot, foe_y and tZ
    # combine the two estimations and return estimated Z
    x = (tZ * (foe[0] - p_rot[0])) / (p_curr[0] - p_rot[0])
    y = (tZ * (foe[1] - p_rot[1])) / (p_curr[1] - p_rot[1])
    return (x + y) / 2


