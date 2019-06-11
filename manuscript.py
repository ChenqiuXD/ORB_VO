# encoding: UTF-8
"""
This file stores my manuscripts while doing this project.
"""

"""
When assigning primitive type value to a variable, it just points to it, and changes its type, as is the feature of 
dynamic language, even for static attributes inside instances (if assigned with a new value, then it changes without 
changing the __class__ static variable, which means it no longer points to the __class__ static variable).
"""
# class A:
#     attr_static = 0
#
#     def __init__(self, value):
#         self.attr_none_static = value
#
#
# if __name__ == "__main__":
#     a = A(0)
#     a1 = A(1)
#     a1.attr_static = 1
#     print(a1.attr_static)
#     a2 = A(2)
#     a2.attr_static = 2
#     print(a2.attr_static)
#     a3 = A(3)
#     a3.attr_static = 3
#     print(a3.attr_static)
#     A.attr_static = 9
#     print(a.attr_static)

from icp.icp import best_fit_transform
import numpy as np

p0 = np.array([[1, 0, 0],
               [0, 1, 0],
               []])
p1 = np.array([0, 1, 0]).reshape(1, 3)
T, R, t = best_fit_transform(p1, p0)
print(T)
print(R)
print(t)


if __name__ == "__main__":
    print('manuscript.py')
