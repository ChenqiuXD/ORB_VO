# -*- coding:utf-8 -*-
import numpy as np
import pyrealsense2 as rs
import cv2
from math import sin, cos
import math
from pso import PSO
import icp.icp as icp
from scipy.optimize import least_squares

USE_LM = True
THRESHHOLD = 30
FEATUREMAX = 200
INLIER_DIST_THRE = 0.05
MAX_DIS = 6
MIN_DIS = 0.2

def cal_matrix_T(x):
    "x 4d vector: x y z theta"
    return np.array([[cos(x[3]), sin(x[3]), 0,x[0]],
                        [-sin(x[3]), cos(x[3]),0, x[1]],
                        [0,0,1,x[2]],
                        [0, 0,0, 1]])

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
        best_matchIdx = -1
        best_matchVal = 0

        len_of_match = len(self.match)
        if not candidate.any():
            return None
        for i in candidate:
            if self.W[len_of_match][i] > best_matchVal:
                best_matchVal = self.W[len_of_match][i]
                best_matchIdx = i
        return best_matchIdx

    def simple_match_filter(self,threshhold):
        assert len(self.camera_coordinate_second) == len(self.camera_coordinate_first)
        original_match_len = len(self.camera_coordinate_first)
        after_filter_first = []
        after_filter_second = []
        for i in range(original_match_len):
            delta_x = abs(self.camera_coordinate_first[i][0] - self.camera_coordinate_second[i][0])
            delta_y = abs(self.camera_coordinate_first[i][1] - self.camera_coordinate_second[i][1])
            delta_z = abs(self.camera_coordinate_first[i][2] - self.camera_coordinate_second[i][2])
            if delta_x<threshhold and delta_y<threshhold and delta_z<threshhold:
                after_filter_first.append(self.camera_coordinate_first[i])
                after_filter_second.append(self.camera_coordinate_second[i])
        self.camera_coordinate_first = after_filter_first.copy()
        self.camera_coordinate_second = after_filter_second.copy()




    def calculate_camera_coordinates(self):
        """This method get the list A and B by rs.deproject function"""
        self.camera_coordinate_first = []
        self.camera_coordinate_second = []
        self.camera_pixel_second = []
        match = []
        for pair in self.match:
            img_pixel = [int(self.featureFrame_first[pair.queryIdx].pt[0]),
                         int(self.featureFrame_first[pair.queryIdx].pt[1])]
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

            img_pixel = [int(self.featureFrame_second[pair.trainIdx].pt[0]),
                         int(self.featureFrame_second[pair.trainIdx].pt[1])]
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
            match.append(pair)
            self.camera_coordinate_first.append([point_a[0],point_a[2],point_a[1]])
            self.camera_pixel_second.append(point_b_pixel)
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
        error = np.sum(np.square(A_.T - matrix.dot(B_.T)), axis=1)
        # print(error)
        return error

        # error =

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

            self.A = np.array(self.camera_coordinate_first, dtype=np.float32).round(3)
            self.B = np.array(self.camera_coordinate_second, dtype=np.float32).round(3)
            result = least_squares(self.func,x0=[0,0,0,0],method='lm').x.round(2)
            T = cal_matrix_T(result).round(2)
            # print(result)

            # Use the result above as initial value for further optimize to get delta_x, delta_y and delta_theta
            # x0 = [tvec[0], tvec[2], math.atan2(rvec[0, 2], rvec[0, 0])]
            # self.res = least_squares(self.func, x0, method='lm')
            # self.res.x *= 100

            # Calculate the displacement matrix
            # temp = np.hstack((rvec, tvec))
            # self.displace_mat = np.vstack((temp, [0, 0, 0, 1]))
            self.displace_mat = T

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

        ORBDetector.tm = np.dot(ORBDetector.tm,self.displace_mat).round(2)
        ORBDetector.pp = np.array([ORBDetector.tm[0, 3], ORBDetector.tm[1, 3], math.atan2(
            ORBDetector.tm[0, 1], ORBDetector.tm[1, 1])])
