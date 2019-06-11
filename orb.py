# encoding: UTF-8
import numpy as np
import pyrealsense2 as rs
import cv2
from math import sin, cos
import math
from pso import PSO
import icp.icp as icp
from scipy.optimize import least_squares
import copy
import random

USE_LM = True
THRESHHOLD = 30
FEATUREMAX = 200
INLIER_DIST_THRE = 1
MAX_DIS = 4
MIN_DIS = 0.5


class ORBDetector:
    pp = np.array([0.0, 0.0, 0.0])  # The initial position and posture of the
    tm = np.eye(4)

    def __init__(self, depth_intrin, use_lm=USE_LM, inlier_thre=INLIER_DIST_THRE, max_dis=MAX_DIS, min_dis=MIN_DIS):
        # Every frame has four attribute : color_frame, depth_frame, features, feature_descriptorss.
        self.first_color_frame = []
        self.second_color_frame = []
        self.first_depth_frame = []
        self.second_depth_frame = []
        self.featureFrame_first = []
        self.featureFrame_second = []
        self.featureDes_first = []
        self.featureDes_second = []

        self.depth_intrin = depth_intrin
        self.orb = cv2.ORB_create(nfeatures=FEATUREMAX, fastThreshold=THRESHHOLD)
        self.USE_LM = use_lm
        self.INLIER_DIST_THRE = inlier_thre
        self.min_dis =min_dis
        self.max_dis = max_dis
        self.score = []
        self.bfMatcher = cv2.BFMatcher_create(normType=cv2.NORM_HAMMING, crossCheck=True)
        self.match = []
        self.W = []
        self.best_matches = []

        # The following part stores the coordinate of the features
        # self.world_coordinate_first = []
        self.camera_coordinate_first = []
        # self.world_coordinate_second = []
        self.camera_coordinate_second = []
        self.camera_pixel_second = []

        # self.res is the brief for result, displace_mat is a 4*4 matrix representing the homogeneous transform matrix
        self.res = [0, 0, 0]
        self.displace_mat = []

    def set_first_frame(self, color_frame, depth_frame):
        self.first_color_frame = np.asanyarray(color_frame.get_data())
        self.first_depth_frame = depth_frame

    def set_second_frame(self, color_frame, depth_frame):
        self.second_color_frame = np.asanyarray(color_frame.get_data())
        self.second_depth_frame = depth_frame

    def reset_frame(self, color_frame_next, depth_frame_next):
        """This method is applied after each frame is processed intending for reduce the calculation cost
        Refer to the jounal : A2 step last paragraph"""
        self.featureDes_first = self.featureDes_second
        self.featureFrame_first = self.featureFrame_second
        self.first_color_frame = self.second_color_frame
        self.first_depth_frame = self.second_depth_frame

        self.second_color_frame = np.asanyarray(color_frame_next.get_data())
        self.second_depth_frame = depth_frame_next
        self.featureFrame_second, self.featureDes_second = self.orb.detectAndCompute(self.second_color_frame, None)

    def detect_all_features(self):
        """For debugging in test.py, would not be called in read_from_bag or example_run"""
        self.featureFrame_first, self.featureDes_first = self.orb.detectAndCompute(self.first_color_frame, None)
        self.featureFrame_second, self.featureDes_second = self.orb.detectAndCompute(self.second_color_frame, None)

    def detect_second_features(self):
        """Detect features and calculate the descriptors"""
        # P.S. the features and descriptors of frame A are calculated beforehand
        # self.featureFrame_first, self.featureDes_first = self.orb.detectAndCompute(self.first_color_frame, None)
        self.featureFrame_second, self.featureDes_second = self.orb.detectAndCompute(self.second_color_frame, None)

    def match_features(self):
        """This method match the features using BrutalForce and sort them by similarity
         and only take the strongest 50"""
        if self.featureDes_first is not None and self.featureDes_second is not None:
            # IMPORTANT : match(queryDescriptors, trainDescriptors)
            matches = self.bfMatcher.match(self.featureDes_first, self.featureDes_second)
            self.match = sorted(matches, key=lambda x: x.distance)
            self.match = self.match[:50]
        else:
            self.match = []

    def find_most_compatible_match(self, candidate):
        """This method loop through candidate to find matches which has most compatible number"""
        best_matchIdx = best_matchVal = None
        len_of_match = len(self.match)
        if not candidate.any():
            return None
        for i in candidate:
            if self.W[len_of_match][i] > best_matchVal:
                best_matchVal = self.W[len_of_match][i]
                best_matchIdx = i
        return best_matchIdx

    def find_inlier_without_depth(self):
        """This method execute the A4 step of the journal"""
        len_of_matches = len(self.match)
        # The last line of W stores the whole number of consistency of this match
        self.W = np.zeros((len_of_matches + 1, len_of_matches))
        for i in np.arange(len_of_matches):
            for j in np.arange(len_of_matches):
                if i >= j:
                    continue

                # ASSUMPTION : the index of descriptor is the same with the index of image
                wa = self.featureFrame_first[self.match[i].queryIdx].pt[0] - \
                     self.featureFrame_first[self.match[j].queryIdx].pt[0]
                wb = self.featureFrame_first[self.match[i].queryIdx].pt[1] - \
                     self.featureFrame_first[self.match[j].queryIdx].pt[1]
                wa_ = self.featureFrame_second[self.match[i].trainIdx].pt[0] - \
                      self.featureFrame_second[self.match[j].trainIdx].pt[0]
                wb_ = self.featureFrame_second[self.match[i].trainIdx].pt[1] - \
                      self.featureFrame_second[self.match[j].trainIdx].pt[1]

                # Compare and complete the matrix W
                if abs(wa - wa_) + abs(wb - wb_) <= self.INLIER_DIST_THRE:
                    self.W[i, j] = 1
                    self.W[j, i] = 1
                    self.W[len_of_matches, j] += 1

        # Choose the best inlier features
        self.best_matches = []
        candidate = np.arange(len_of_matches)
        while True:
            best_matchIdx = self.find_most_compatible_match(candidate)
            if not best_matchIdx or best_matchIdx == -1:  # in case no best match is found
                break
            else:
                self.best_matches.append(self.match[best_matchIdx])
                candidate = np.delete(candidate, np.where(candidate == best_matchIdx), axis=0)

    def find_inlier(self):
        """This method execute the A4 step of the journal"""
        len_of_matches = len(self.match)
        # The last line of W stores the whole number of consistency of this match
        self.W = np.zeros((len_of_matches + 1, len_of_matches))
        for i in np.arange(len_of_matches):
            for j in np.arange(len_of_matches):
                if i >= j:
                    continue

                # ASSUMPTION : the index of descriptor is the same with the index of image
                wa = self.featureFrame_first[self.match[i].queryIdx].pt[0] - \
                     self.featureFrame_first[self.match[j].queryIdx].pt[0]
                wb = self.featureFrame_first[self.match[i].queryIdx].pt[1] - \
                     self.featureFrame_first[self.match[j].queryIdx].pt[1]
                img_pixel1 = [int(self.featureFrame_first[self.match[i].queryIdx].pt[0]),
                              int(self.featureFrame_first[self.match[i].queryIdx].pt[1])]
                img_pixel2 = [int(self.featureFrame_first[self.match[j].queryIdx].pt[0]),
                              int(self.featureFrame_first[self.match[j].queryIdx].pt[1])]
                depth1 = self.first_depth_frame.get_distance(img_pixel1[0], img_pixel1[1])
                depth2 = self.first_depth_frame.get_distance(img_pixel2[0], img_pixel2[1])
                wc = depth1 - depth2
                wa_ = self.featureFrame_second[self.match[i].trainIdx].pt[0] - \
                      self.featureFrame_second[self.match[j].trainIdx].pt[0]
                wb_ = self.featureFrame_second[self.match[i].trainIdx].pt[1] - \
                      self.featureFrame_second[self.match[j].trainIdx].pt[1]
                img_pixel1_ = [int(self.featureFrame_second[self.match[i].trainIdx].pt[0]),
                               int(self.featureFrame_second[self.match[i].trainIdx].pt[1])]
                img_pixel2_ = [int(self.featureFrame_second[self.match[j].trainIdx].pt[0]),
                               int(self.featureFrame_second[self.match[j].trainIdx].pt[1])]
                depth1 = self.second_depth_frame.get_distance(img_pixel1_[0], img_pixel1_[1])
                depth2 = self.second_depth_frame.get_distance(img_pixel2_[0], img_pixel2_[1])
                wc_ = depth1 - depth2
                # Todo: based on the three dimension imformation

                # Compare and complete the matrix W
                if abs(wa - wa_) + abs(wb - wb_) + abs(wc - wc_) <= self.INLIER_DIST_THRE:
                    self.W[i, j] = 1
                    self.W[j, i] = 1
                    self.W[len_of_matches, j] += 1

        # Choose the best inlier features
        self.best_matches = []
        candidate = np.arange(len_of_matches)
        while True:
            best_matchIdx = self.find_most_compatible_match(candidate)
            if not best_matchIdx or best_matchIdx == -1:  # in case no best match is found
                break
            else:
                self.best_matches.append(self.match[best_matchIdx])
                candidate = np.delete(candidate, np.where(candidate == best_matchIdx), axis=0)

    def calculate_camera_coordinates(self):
        """This method get the list A and B by rs.deproject function"""
        for match in self.best_matches:
            img_pixel = [int(self.featureFrame_first[match.queryIdx].pt[0]),
                         int(self.featureFrame_first[match.queryIdx].pt[1])]
            # !!!!!!!!!!!!!!!!!!!!!!!!! CAUTIOUS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
            # The camera coordinate is different from ndarray coordinate from that their x and y axis are reversed
            depth = self.first_depth_frame.get_distance(img_pixel[1], img_pixel[0])
            if depth >= self.max_dis or depth <= self.min_dis:
                # print(depth)
                continue
            # print(depth)
            point_a = rs.rs2_deproject_pixel_to_point(self.depth_intrin, img_pixel, depth)
            # threeD_file.write(str(point_a[1]))
            # threeD_file.write("\n")
            # point_a = [point_a[0], point_a[2], 1]

            img_pixel = [int(self.featureFrame_second[match.trainIdx].pt[0]),
                         int(self.featureFrame_second[match.trainIdx].pt[1])]
            depth = self.second_depth_frame.get_distance(img_pixel[1], img_pixel[0])
            if depth >= self.max_dis or depth <= self.min_dis:
                # print(depth)
                continue
            point_b_pixel = img_pixel
            # print(depth)
            point_b = rs.rs2_deproject_pixel_to_point(self.depth_intrin, img_pixel, depth)
            # threeD_file.write(str(point_b[1]))
            # threeD_file.write("\n")
            # point_b = [point_b[0], point_b[2], 1]
            self.camera_coordinate_first.append(point_a)
            self.camera_pixel_second.append(point_b_pixel)
            self.camera_coordinate_second.append(point_b)

    def func(self, x):
        """Cost function used for optimization. Variables are defined as follows:
        x = [delta_x, delta_z, delta_θ] the sign of delta should refer to the camera coordinate
        self.camera_coordinate_first = [x_a, y_a, z_a]
        self.camera_coordinate_second = [x_b, y_b, z_b]
        P.S. the y_a should approximately equal to y_b coz the height of camera remains the same
        The function could be illustrate as follows:
            [x_b]   [cos(d_θ)   0  sin(d_θ) d_x] [x_a]
            |y_b| = |     0     1       0   d_y| |y_a|
            |z_b| = |-sin(d_θ)  0  cos(d_θ) d_z| |z_a|
            [1  ]   [     0     0       0    1 ] [ 1 ]
        d_y approximately equals to 0
        """
        result = []
        for j in range(len(self.camera_coordinate_first)):
            result.append(self.camera_coordinate_second[j][0] - (
                        cos(x[2]) * self.camera_coordinate_first[j][0] +
                        sin(x[2]) * self.camera_coordinate_first[j][2] + x[0]))
            result.append(self.camera_coordinate_second[j][2] - (
                        - sin(x[2]) * self.camera_coordinate_first[j][0] +
                        cos(x[2]) * self.camera_coordinate_first[j][2] + x[1]))
        return np.asarray(result)

    @staticmethod
    def ransac_cost_func(x, cord_list=None, is_lm=False):
        result = []
        for cord in cord_list:
            result.append(cord[1][0] - (cos(x[2])*cord[0][0] + sin(x[2])*cord[0][2] + x[0]))
            result.append(cord[1][2] - (-sin(x[2])*cord[0][0] + cos(x[2])*cord[0][2] + x[1]))
        if is_lm:
            return np.array(result)
        else:
            return np.max(np.fabs(result))

    def optimize(self):
        """LM method by scipy"""
        if self.USE_LM:
            # Use opencv function solvePnPRansac to get translational and rotational movement
            # list_a = np.array(self.camera_coordinate_first, dtype=np.float32).reshape((len(
            #     self.camera_coordinate_first), 1, 3))
            # list_b = np.array(self.camera_pixel_second, dtype=np.float32).reshape((len(self.camera_pixel_second),
            # 1, 2))
            # camera_mat = np.array([[self.depth_intrin.fx, 0, self.depth_intrin.ppx],
            #                        [0, self.depth_intrin.fy, self.depth_intrin.ppy],
            #                        [0, 0, 1]])
            # dist = np.zeros(5)
            # retval, rvec, tvec, _ = cv2.solvePnPRansac(list_a, list_b, camera_mat, distCoeffs=dist)
            # rvec, _ = cv2.Rodrigues(rvec)

            A = np.array(self.camera_coordinate_first, dtype=np.float32)
            B = np.array(self.camera_coordinate_second, dtype=np.float32)
            T, distances = icp.icp(B, A, tolerance=1e-6)[:2]

            # Use the result above as initial value for further optimize to get delta_x, delta_y and delta_theta
            # x0 = [tvec[0], tvec[2], math.atan2(rvec[0, 2], rvec[0, 0])]
            # self.res = least_squares(self.func, x0, method='lm')
            # self.res.x *= 100

            # Calculate the displacement matrix
            # temp = np.hstack((rvec, tvec))
            # self.displace_mat = np.vstack((temp, [0, 0, 0, 1]))
            self.displace_mat = T
        else:
            pso_ORBDetector = PSO(population_size=10, max_steps=50, pA=self.camera_coordinate_first, pB=self.listB)
            self.optimized_result = pso_ORBDetector.evolve()
            self.optimized_result[1:3] = 100 * self.optimized_result[1:3]

    def optimize_ransac(self):
        """

        :return:
        """
        # pre-process:
        cord_list_a = copy.deepcopy(self.camera_coordinate_first)
        cord_list_b = copy.deepcopy(self.camera_coordinate_second)
        cord_list = map(list, tuple(zip(cord_list_a, cord_list_b)))
        all_cord_indices = list(range(len(cord_list)))
        for pair in cord_list:
            pair[0] = np.array(pair[0])
            pair[1] = np.array(pair[1])
        """
        cord_list: [[array(), array()], [array(), array()], ...]
        """
        # hyper-parameters:
        min_points = 3
        max_iteration = 100
        threshold = 1e-2
        # min_number_to_assert = 0.8*len(cord_list)
        min_number_to_assert = 1
        # initialize
        iteration = 0
        best_pp = None
        best_err = float('inf')

        while iteration < max_iteration:
            maybe_inliers_indices = random.sample(all_cord_indices, min_points)
            other_indices = list(set(all_cord_indices)-set(maybe_inliers_indices))
            other_cord_list = [cord_list[i] for i in other_indices]
            maybe_inliers = [cord_list[i] for i in maybe_inliers_indices]
            pp = least_squares(self.ransac_cost_func, np.array([0, 0, 0]), method='lm',
                               kwargs={'cord_list': maybe_inliers, 'is_lm': True}).x
            also_inliers = []

            for sample in other_cord_list:
                err = self.ransac_cost_func(pp, cord_list=[sample], is_lm=False)
                if np.fabs(err) < threshold:
                    also_inliers.append(sample)
            if len(also_inliers) > min_number_to_assert:
                better_pp = least_squares(self.ransac_cost_func, np.array([0, 0, 0]), method='lm',
                                          kwargs={'cord_list': maybe_inliers+also_inliers, 'is_lm': True}).x
                this_err = self.ransac_cost_func(better_pp, cord_list=cord_list, is_lm=False)
                if this_err < best_err:
                    best_pp = better_pp
                    best_err = this_err

            iteration += 1
        self.displace_mat = np.array([[cos(best_pp[2]), 0, sin(best_pp[2]), best_pp[0]],
                                      [0, 1, 0, 0],
                                      [-sin(best_pp[2]), 0, cos(best_pp[2]), best_pp[1]],
                                      [0, 0, 0, 1]])
        return best_pp

    def get_new_pp(self):
        # cam_displace = self.optimized_result[[1, 2, 0]]
        # if self.USE_LM:
        #     ORBDetector.pp[2] += self.res.x[2]
        #     tm = np.array([[np.cos(ORBDetector.pp[2]), -np.sin(ORBDetector.pp[2])],
        #                    [np.sin(ORBDetector.pp[2]), np.cos(ORBDetector.pp[2])]])
        #     ORBDetector.pp[:2] += tm.dot(self.res.x[:2])
        # else:
        #     ORBDetector.pp[2] += self.optimized_result[0]
        #     tm = np.array([[np.cos(ORBDetector.pp[2]), -np.sin(ORBDetector.pp[2])],
        #                    [np.sin(ORBDetector.pp[2]), np.cos(ORBDetector.pp[2])]])
        #     ORBDetector.pp[:2] += tm.dot(self.optimized_result[1:3])

        ORBDetector.tm = np.dot(ORBDetector.tm, self.displace_mat)
        ORBDetector.pp = np.array([ORBDetector.tm[0, 3], ORBDetector.tm[2, 3], math.atan2(
            ORBDetector.tm[1, 0], ORBDetector.tm[0, 0])])
