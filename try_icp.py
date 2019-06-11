from icp.icp import icp, best_fit_transform
import numpy as np
from math import cos,sin
from scipy.optimize import least_squares
# print(np.array([[1] for _ in range(3)]))
# initial_matrix = np.array()
def cal_matrix_T(x):
    "x 4d vector: x y z theta"
    return np.array([[cos(x[3]), sin(x[3]), 0,x[0]],
                        [-sin(x[3]), cos(x[3]),0, x[1]],
                        [0,0,1,x[2]],
                        [0, 0,0, 1]])
def fun(result):
    num_points = A.shape[0]
    dimension = A.shape[1]
    matrix = cal_matrix_T(result)
    A_ = np.hstack((A,np.array([[1] for _ in range(num_points)])))
    # print(A_.T)
    B_ = np.hstack((B,np.array([[1] for _ in range(num_points)])))
    # print(B_.T)
    error = np.sum(np.square(A_.T - matrix.dot(B_.T)),axis=0)
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
C = np.array([[1,2.01,1],[2,2.01,1],[3,2.01,1],[4,2.01,1],[6,2.01,1],[14,2.01,1],[12,3.21,1],[10,7.73,1],[3.81,2,1],[5.6,86.73,1],[4.5,16.47,1],[8.4,26,1],[5.2,2.1,1],[11.2,2.3,1],[25.3,2.07,1],[33.4,55,1],[45.9,2.01,1],[11.7,1.41,1]])
A = B
B = C
def rotate(C):
    D = np.hstack((C,np.array([[1] for _ in range(18)])))
    # print(D)
    D =D.T
    D = rotation.dot(D)
    # print(D)
    D = D[:3].T
    return D
for i in range(100):
    C = rotate(C)
    B = C
    result = least_squares(fun, x0=[0, 0, 0, 0], method='lm')
    print(result.x)





        # error =

# result1 = least_squares(fun,x0=[0,0,0,0],method='lm')



