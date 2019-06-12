#!python2
# encoding: UTF-8

from orb import ORBDetector
import numpy as np
from scipy.optimize import least_squares
from scipy.linalg import norm, expm
import time
import math
from math import cos, sin


def rotate_matrix(axis, radian):
    return expm(np.cross(np.eye(3), axis / norm(axis) * radian))


def getT(pp, three_d=False):
    """
    get the displace matrix of a given pp (position and posture)
    :param pp: np.array([<x>, <y>, <z>, <theta_x>, <theta_y>, <theta_z>]
    :param three_d: bool, whether to calculate 3-d coordinates
    :return: displace matrix: 4-by-4 ndarray
    """
    if three_d:
        c1 = cos(pp[3])
        s1 = sin(pp[3])
        c2 = cos(pp[4])
        s2 = sin(pp[4])
        c3 = cos(pp[5])
        s3 = sin(pp[5])

        return np.array([[c3*c2, c3*s2*s1-c1*s3, c3*s2*c1+s3*s1, pp[0]],
                         [s3*c2, s3*s2*s1+c3*c1, s3*s2*c1-c3*s1, pp[1]],
                         [-s2,   c2*s1,          c2*c1,          pp[2]],
                         [0,     0,              0,                 1]])
    else:
        return np.array([[cos(pp[2]), 0, sin(pp[2]), pp[0]],
                         [0, 1, 0, 0],
                         [-sin(pp[2]), 0, cos(pp[2]), pp[1]],
                         [0, 0, 0, 1]])


if __name__ == "__main__":
    three_d = True
    pp = np.array([1, 2, 3, math.pi/2, math.pi/4, math.pi/6])
    T = ORBDetector.getT(pp, three_d)

    p_b = np.array([[1, 2, 3, 1],
                    [2, 4, 3, 1],
                    [1, 4, 3, 1],
                    [5, 6, 7, 1],
                    [3, 4, 5, 1],
                    [1, 0, 0, 1]]).astype(float)
    p_a = p_b.dot(T.T)
    # add some noise:
    p_a[:, :3] += np.random.randn(6, 3) * 0.1
    p_b[:, :3] += np.random.randn(6, 3)*0.1

    p_b = list(p_b[:, :3])
    p_a = list(p_a[:, :3])

    cord_list = list(map(list, zip(p_a, p_b)))

    t0 = time.clock()
    res = least_squares(ORBDetector.ransac_residual_func, np.zeros(6), method='lm',
                       kwargs={'cord_list': cord_list, 'is_lm': True, 'three_d': three_d})
    print("elapsed time: {}".format(time.clock()-t0))
    pp_calculated = res.x

    t0 = time.clock()
    T1 = getT(pp, True)
    print("elapsed time: {}".format(time.clock() - t0))

    t0 = time.clock()
    T2 = ORBDetector.getT(pp, True)
    print("elapsed time: {}".format(time.clock() - t0))

    assert(np.allclose(T1, T2))

"""
1. 至少lm的least squares是正确的。 
2. 事实证明，矩阵相乘计算和直接利用公式计算的计算复杂度相差了100倍的效率。。。
"""