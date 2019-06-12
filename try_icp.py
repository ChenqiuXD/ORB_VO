from icp.icp import icp, best_fit_transform
import numpy as np
from math import cos,sin
from scipy.optimize import least_squares
# print(np.array([[1] for _ in range(3)]))
# initial_matrix = np.array()
def cal_matrix_T(x):
    "x: 6d vector : x , y, z"
    c1 = cos(x[3])
    s1 = sin(x[3])
    c2 = cos(x[4])
    s2 = sin(x[4])
    c3 = cos(x[5])
    s3 = sin(x[5])

    mat = np.zeros((4,4))

    mat[0,0] = (c1 * c3) - (s1 * c2 * s3)
    mat[0,1] = (-c1 * s3) - (s1 * c2 * c3)
    mat[0,2] = (s1 * s2)
    mat[0,3] = x[0]

    mat[1,0] = (s1 * c3) + (c1 * c2 * s3)
    mat[1,1] = (-s1 * s3) + (c1 * c2 * c3)
    mat[1,2] = (-c1 * s2)
    mat[1, 3] = x[1]

    mat[2,0] = (s2 * s3)
    mat[2,1] = (s2 * c3)
    mat[2,2] = c2
    mat[2, 3] = x[2]

    mat[3,0] = mat[3,1] = mat[3,2] = 0
    mat[3,3] =1

    return mat
def fun(result):
    num_points = A.shape[0]
    dimension = A.shape[1]
    matrix = cal_matrix_T(result)
    A_ = np.hstack((A,np.array([[1] for _ in range(num_points)])))
    # print(A_.T)
    B_ = np.hstack((B,np.array([[1] for _ in range(num_points)])))
    # print(B_.T)
    error = np.square(A_.T - matrix.dot(B_.T)).flatten()
    # print(error)
    return error
theta = 0.01*np.pi
rotation = np.array([[cos(theta),sin(theta),0,0],
                     [-sin(theta),cos(theta),0,0],
                     [0,0,1,0],
                     [0,0,0,1]])
# print(rotation)
A = np.array([[1,0,1],[2,0,1],[3,0,1],[4,0,1]])
B = np.array([[1,2,1],[2,2,1],[3,2,1],[4,2,1],[6,2,1],[14,2,1],[12,3.2,1],[10,7.7,1],[3.8,2,1],[5.6,86.7,1],[4.5,16.4,1],[8.4,2,1],[5.2,2,1],[11.2,2,1],[25.3,2,1],[33.4,51,1],[45.9,2,1],[11.7,1.4,1]])
C = np.array([[1,2.01,1],[2,2.01,1],[3,2.01,1],[4,2.01,1],[6,2.01,1],[14,2.01,1],[12,3.21,1],[10,7.71,1],[3.81,2.01,1],[5.6,86.71,1],[4.5,16.41,1],[8.4,2.01,1],[5.2,2.01,1],[11.2,2.01,1],[25.3,2.01,1],[33.4,52,1],[45.9,2.01,1],[11.7,1.41,1]])
def rotate(C):
    D = np.hstack((C,np.array([[1] for _ in range(18)])))
    # print(D)
    D =D.T
    D = rotation.dot(D)
    # print(D)
    D = D[:3].T
    return D
for i in range(1):
    result = icp(C,B,max_iterations=1000,tolerance=1e-10)
    print(result)
    C = np.hstack((C,np.array([[1] for _ in range(18)])))
    # print(D)
    C =C.T
    print(result.dot(C))





        # error =

# result1 = least_squares(fun,x0=[0,0,0,0],method='lm')



