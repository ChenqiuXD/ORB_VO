# -*- coding:utf-8 -*-
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
from scipy.linalg import expm, norm

USE_LM = True
THRESHHOLD = 30
FEATUREMAX = 200
INLIER_DIST_THRE = 0.2
MAX_DIS = 10
MIN_DIS = 0.2
FOUR = True

def cal_matrix_T(x):
    "x: 6d vector : x , y, z"
    c1 = cos(x[2])
    s1 = sin(x[2])
    c2 = cos(0)
    s2 = sin(0)
    c3 = cos(0)
    s3 = sin(0)
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
    mat[2, 3] = 0

    mat[3,0] = mat[3,1] = mat[3,2] = 0
    mat[3,3] =1

    return mat




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
        self.camera_pixel_first = []


        # self.res is the brief for result, displace_mat is a 4*4 matrix representing the homogeneous transform matrix
        self.res = [0, 0, 0]
        self.displace_mat = np.eye(4)

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


    def calc_w(self):
        len_of_matches = len(self.match)
        self.W = np.zeros((len_of_matches+1, len_of_matches))
        # The last line of W stores the whole number of consistency of this match
        for i in range(len_of_matches):
            for j in range(len_of_matches):
                if i >= j:
                    continue

                # ASSUMPTION : the order of self.camera_coordinate_first is the same with self.match
                wa = self.camera_coordinate_first[i][0] - self.camera_coordinate_first[j][0]
                wb = self.camera_coordinate_first[i][1] - self.camera_coordinate_first[j][1]
                wc = self.camera_coordinate_first[i][2] - self.camera_coordinate_first[j][2]

                wa_ = self.camera_coordinate_second[i][0] - self.camera_coordinate_second[j][0]
                wb_ = self.camera_coordinate_second[i][1] - self.camera_coordinate_second[j][1]
                wc_ = self.camera_coordinate_second[i][2] - self.camera_coordinate_second[j][2]

                if abs(wa-wa_) + abs(wb-wb_) + abs(wc-wc_) <= self.INLIER_DIST_THRE:
                    self.W[i, j] = self.W[j, i] = 1

        for i in range(len_of_matches):
            # Sum up all the row values
            self.W[len_of_matches, i] = np.sum(self.W[:, i])

    def find_most_compatible_match(self, candidate):
        """This method loop through candidate to find matches which has most compatible number"""
        best_matchIdx = -1
        best_matchVal = 0
        len_of_match = len(self.match)
        if not any(candidate):
            return -1
        for i in candidate:
            if self.W[len_of_match][i] > best_matchVal:
                best_matchVal = self.W[len_of_match][i]
                best_matchIdx = i
        return best_matchIdx

    def find_inlier_3d(self):
        # Calculate the W matrix
        self.calc_w()

        # Find the most compatible index until no match is compatible
        self.best_matches = []
        new_matches = []
        candidate = np.array(range(len(self.match)))

        new_camera_coor_first = []
        new_camera_coor_second = []
        new_camera_pixel_first = []
        new_camera_pixel_second = []
        self.index_result = []
        while True:
            index = self.find_most_compatible_match(candidate)
            if index == -1:
                break
            self.best_matches.append(self.match[index])
            new_matches.append(self.match[index])
            self.index_result.append(index)

            new_camera_coor_first.append(self.camera_coordinate_first[index])
            new_camera_coor_second.append(self.camera_coordinate_second[index])
            new_camera_pixel_first.append(self.camera_pixel_first[index])
            new_camera_pixel_second.append(self.camera_pixel_second[index])

            # candidate = np.delete(candidate, np.argwhere(candidate == index), axis=0)
            for i in candidate:
                if self.W[index, i] == 0:
                    candidate = np.delete(candidate, np.argwhere(candidate == i))

        self.camera_coordinate_first = new_camera_coor_first.copy()
        self.camera_coordinate_second = new_camera_coor_second.copy()
        self.camera_pixel_first = new_camera_pixel_first.copy()
        self.camera_pixel_second = new_camera_pixel_second.copy()
        self.match = new_matches.copy()

    def simple_match_filter(self,threshhold):
        assert len(self.camera_coordinate_second) == len(self.camera_coordinate_first)==len(self.match)
        original_match_len = len(self.camera_coordinate_first)
        after_filter_first = []
        after_filter_second = []
        max_deletax = 0
        max_deletay = 0
        max_deletaz = 0
        new_matches = []
        for i in range(original_match_len):

            delta_x = abs(self.camera_coordinate_first[i][0] - self.camera_coordinate_second[i][0])
            delta_y = abs(self.camera_coordinate_first[i][1] - self.camera_coordinate_second[i][1])
            delta_z = abs(self.camera_coordinate_first[i][2] - self.camera_coordinate_second[i][2])
            if delta_x<threshhold and delta_y<threshhold and delta_z < threshhold:
                if delta_x > max_deletax :
                    max_deletax = delta_x
                if delta_y > max_deletay :
                    max_deletay = delta_y
                if delta_z > max_deletaz :
                    max_deletaz = delta_z

                after_filter_first.append(self.camera_coordinate_first[i])
                after_filter_second.append(self.camera_coordinate_second[i])
                self.best_matches.append(self.match[i])
                new_matches.append(self.match[i])
        self.camera_coordinate_first = after_filter_first.copy()
        self.camera_coordinate_second = after_filter_second.copy()
        self.match = new_matches
        print('过滤后的关键点数量:'+str(len(self.camera_coordinate_first)))
        print('过滤后的delta最大值'+str([max_deletax,max_deletay,max_deletaz]))
        if len(self.camera_coordinate_first):
            print('过滤后的最远点:'+str(np.sort(np.asanyarray(self.camera_coordinate_first),axis=0)[-1][1]))
            print('过滤后的最近点:'+str(np.sort(np.asanyarray(self.camera_coordinate_first),axis=0)[0][1]))

    def calculate_camera_coordinates(self,depth_to_color_extrin):
        """This method get the list A and B by rs.deproject function"""
        self.camera_coordinate_first = []
        self.camera_coordinate_second = []
        self.camera_pixel_second = []
        self.camera_pixel_first = []
        match = []
        for pair in self.match:
            img_pixel = [int(self.featureFrame_first[pair.queryIdx].pt[1]),
                         int(self.featureFrame_first[pair.queryIdx].pt[0])]
            # !!!!!!!!!!!!!!!!!!!!!!!!! CAUTIOUS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
            # The camera coordinate is different from ndarray coordinate from that their x and y axis are reversed
            depth = self.first_depth_frame.get_distance(img_pixel[1], img_pixel[0])
            if depth >= self.max_dis or depth <= self.min_dis:
                # print(depth)
                continue
            # print(depth)
            depth_point_a = rs.rs2_deproject_pixel_to_point(self.depth_intrin, img_pixel[::-1], depth)
            point_a = rs.rs2_transform_point_to_point(depth_to_color_extrin, depth_point_a)
            point_a_pixel = img_pixel
            # threeD_file.write(str(point_a[1]))
            # threeD_file.write("\n")
            # point_a = [point_a[0], point_a[2], 1]

            img_pixel = [int(self.featureFrame_second[pair.trainIdx].pt[1]),
                         int(self.featureFrame_second[pair.trainIdx].pt[0])]
            depth = self.second_depth_frame.get_distance(img_pixel[1], img_pixel[0])
            if depth >= self.max_dis or depth <= self.min_dis:
                # print(depth)
                continue
            point_b_pixel = img_pixel
            # print(depth)
            depth_point_b = rs.rs2_deproject_pixel_to_point(self.depth_intrin, img_pixel[::-1], depth)
            point_b = rs.rs2_transform_point_to_point(depth_to_color_extrin, depth_point_b)
            # threeD_file.write(str(point_b[1]))
            # threeD_file.write("\n")
            # point_b = [point_b[0], point_b[2], 1]
            match.append(pair)
            self.camera_coordinate_first.append([point_a[0],point_a[2],point_a[1]])
            self.camera_pixel_second.append(point_b_pixel)
            self.camera_pixel_first.append(point_a_pixel)
            self.camera_coordinate_second.append([point_b[0],point_b[2],point_b[1]])
        self.match = match.copy()

    def func(self,result):
        num_points = self.A.shape[0]
        dimension = self.A.shape[1]
        matrix = cal_matrix_T(result)
        A_ = np.hstack((self.A, np.array([[1] for _ in range(num_points)])))
        # print(A_.T)
        B_ = np.hstack((self.B, np.array([[1] for _ in range(num_points)])))
        # print(B_.T)
        if not FOUR:
            error = np.square(A_.T - matrix.dot(B_.T)).flatten()
        else:
            error = np.sum(np.square(A_.T - matrix.dot(B_.T)),axis=1)

        # print(error)
        return error

    @staticmethod
    def rotate_matrix(axis, radian):
        """
        deprecated for the low efficiency.
        :param axis: rotation axis
        :param radian: radian
        :return: 3-by-3 rotation matrix
        """
        return expm(np.cross(np.eye(3), axis / norm(axis) * radian))

    @staticmethod
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

            return np.array([[c3 * c2, c3 * s2 * s1 - c1 * s3, c3 * s2 * c1 + s3 * s1, pp[0]],
                             [s3 * c2, s3 * s2 * s1 + c3 * c1, s3 * s2 * c1 - c3 * s1, pp[1]],
                             [-s2, c2 * s1, c2 * c1, pp[2]],
                             [0, 0, 0, 1]])
        else:
            # return np.array([[cos(pp[2]), 0, sin(pp[2]), pp[0]],
            #                  [0, 1, 0, 0],
            #                  [-sin(pp[2]), 0, cos(pp[2]), pp[1]],
            #                  [0, 0, 0, 1]])
            return np.array([[cos(pp[2]), -sin(pp[2]), 0, pp[0]],
                                  [sin(pp[2]), cos(pp[2]), 0, pp[1]],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 1]])

    @staticmethod
    def ransac_residual_func(x, cord_list=None, is_lm=False, three_d=False):
        """
        calculate the 3-dimensional residuals for a given model parameters x.
        If is_lm is True, this can be used in LM Least Squared method.
        :param x: np.array([<x>, <y>, <z>, <theta_x>, <theta_y>, <theta_z>]
        :param cord_list: a list of tuples of cords.
        :param is_lm: bool.
        :param three_d: bool.
        :return: residuals. If is_lm is true, it is an array of residuals.
        """
        result = []
        for cord in cord_list:
            if three_d:
                result.extend((ORBDetector.getT(x, three_d=True).dot(np.append(cord[1], 1)) - np.append(cord[0],
                                                                                                        1))[:3])
            else:
                result.extend([cord[0][0] - (cos(x[2]) * cord[1][0] + sin(x[2]) * cord[1][2] + x[0]),
                               cord[0][2] - (-sin(x[2]) * cord[1][0] + cos(x[2]) * cord[1][2] + x[1])])
                # result.append(cord[1][0] - (cos(x[2])*cord[0][0] + sin(x[2])*cord[0][2] + x[0]))
                # result.append(cord[1][2] - (-sin(x[2])*cord[0][0] + cos(x[2])*cord[0][2] + x[1]))

        if is_lm:  # return errors for all coordinates and points for the call of least_squares.
            return np.array(result)
        else:  # get the maximum absolute error on all coordinates
            return np.max(np.fabs(result))

    def optimize_ransac(self, three_d=False):
        """
        use lm (a sort of least squares) inside ransac each loop to solve the problem of outliers in matches.
        :return: The best position-posture variable: np.array([x, z, \theta])
        """
        # pre-process:
        cord_list_a = copy.deepcopy(self.camera_coordinate_first)
        cord_list_b = copy.deepcopy(self.camera_coordinate_second)
        cord_list = list(map(list, tuple(zip(cord_list_a, cord_list_b))))
        all_cord_indices = range(len(cord_list))
        # for debug:
        # d_z_list = [cord[1][2]-cord[0][2] for cord in cord_list]
        for pair in cord_list:  # Convert the coordinates from a list to a numpy array object
            pair[0] = np.array(pair[0])[[0, 2, 1]]
            pair[1] = np.array(pair[1])[[0, 2, 1]]
        """
        After convertion:
        cord_list: [[array(), array()], [array(), array()], ...]
        """
        # hyper-parameters:
        min_points = 6  # I think more points will add the possibility to obtain a good model.
        if min_points > len(cord_list):
            min_points = len(cord_list) - 1
        max_iteration = 20
        threshold = 0.015
        min_number_to_assert = 0.8 * len(cord_list)
        # if min_points != 20:
        #     min_number_to_assert = 0  # If too few matches, just accept it
        # initialize
        iteration = 0
        best_pp = None
        best_err = float('inf')
        if three_d:
            x0 = np.zeros(6)
        else:
            x0 = np.zeros(3)

        while iteration < max_iteration:
            maybe_inliers_indices = random.sample(all_cord_indices, min_points)
            other_indices = list(set(all_cord_indices) - set(maybe_inliers_indices))
            other_cord_list = [cord_list[i] for i in other_indices]
            maybe_inliers = [cord_list[i] for i in maybe_inliers_indices]
            # Calculate the pp with selected (maybe) inliers
            pp = least_squares(self.ransac_residual_func, x0, method='lm',
                               kwargs={'cord_list': maybe_inliers, 'is_lm': True, 'three_d': three_d}).x
            also_inliers = []

            # From other instances calculate the also inliers (maybe)
            for sample in other_cord_list:
                err = self.ransac_residual_func(pp, cord_list=[sample], is_lm=False, three_d=three_d)
                if np.fabs(err) < threshold:
                    also_inliers.append(sample)

            if len(also_inliers + maybe_inliers) > min_number_to_assert:
                best_pp = pp
                break
            else:
                # candidate_pp = least_squares(self.ransac_residual_func, x0, method='lm',
                #                              kwargs={'cord_list': maybe_inliers+also_inliers, 'is_lm': True,
                #                                      'three_d': three_d}).x
                this_err = self.ransac_residual_func(pp, cord_list=cord_list, is_lm=False, three_d=three_d)
                if this_err < best_err:
                    best_pp = pp
                    best_err = this_err
            # print(best_err)
            iteration += 1

        self.displace_mat = ORBDetector.getT(best_pp, three_d=three_d)
        return best_pp

    def optimize(self):
        """LM method by scipy"""
        if self.USE_LM:

            self.A = np.array(self.camera_coordinate_first, dtype=np.float32)
            self.B = np.array(self.camera_coordinate_second, dtype=np.float32)
            if USE_LM:
                self.optimized_result = least_squares(self.func,x0=[0,0,0],method='lm').x
                T = cal_matrix_T(self.optimized_result)
            else:
                result,_,_ = icp.icp(A = self.A,B=self.B)

            self.displace_mat = T

    def get_new_pp(self):

        ORBDetector.tm = np.dot(ORBDetector.tm,self.displace_mat)
        ORBDetector.pp = np.array([ORBDetector.tm[0, 3], ORBDetector.tm[1, 3], math.atan2(
            ORBDetector.tm[0, 1], ORBDetector.tm[1, 1])])

    def check_estimate(self,threshhold_coord,threshhold_theta,):
        delta_x = self.optimized_result[0]
        delta_y = self.optimized_result[1]
        delta_theta = self.optimized_result[2]
        if abs(delta_x)+abs(delta_y) < threshhold_coord or abs(delta_theta)<threshhold_theta:
            return False


