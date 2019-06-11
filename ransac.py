#!python2
# encoding: UTF-8
import numpy as np


def my_ransac(data, model_func, min_points, max_iteration, threshold, min_number_to_assert, **kwargs):
    iterations = 0
    best_fit = None
    best_err = float('inf')

    while iterations < max_iteration:

        maybe_inliers = np.random.choice(data, )