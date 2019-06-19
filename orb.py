# encoding: UTF-8
import numpy as np
import pyrealsense2 as rs
import cv2
from math import sin, cos
import math
# from pso import PSO
# import icp.icp as icp
from scipy.optimize import least_squares
import copy
import random
# from scipy.linalg import expm, norm

USE_LM = True
THRESHHOLD = 30
FEATUREMAX = 1000
INLIER_DIST_THRE = 0.2
MAX_DIS = 10
MIN_DIS = 0.2
FOUR = True

WIN_H = 480
WIN_W = 640
TILE_H = int(WIN_H/3)
TILE_W = int(WIN_W/4)

RANSAC_MIN_ERROR = 1
MAX_FEATURE = 250


def optimize_after():
    file_path = "result_new.txt"
    file_path_w = "result_op.txt"
    f = open(file_path, 'r')
    f_w = open(file_path_w, 'w')
    last_pp = []
    new_pp = []
    curr_pp = []
    lines = f.readlines()
    for i, line in enumerate(lines):
        if i == 0:
            curr_data = line.strip().split(' ')
            curr_pp = [curr_data[1], curr_data[2]]
            new_data = lines[i+1].strip().split(' ')
            new_pp = [new_data[1], new_data[2]]
            f_w.write(line)
        elif i == len(lines)-1:
            f_w.write(line)
        else:
            last_pp = curr_pp
            curr_data = new_data
            curr_pp = new_pp
            new_data = lines[i+1].strip().split(' ')
            new_pp = [new_data[1], new_data[2]]
            curr_pp[0] = (float(last_pp[0]) + float(curr_pp[0]) + float(new_pp[0])) / 3
            curr_pp[1] = (float(last_pp[1]) + float(curr_pp[1]) + float(new_pp[1])) / 3
            f_w.write(str(curr_data[0]) + ' ' + str(curr_pp[0]) + ' ' + str(curr_pp[1]) + '\n')
    f_w.close()
    f.close()


def cal_matrix_T(x):
    "x: 6d vector : x , y, z"
    c1 = cos(x[2])
    s1 = sin(x[2])
    c2 = cos(0)
    s2 = sin(0)
    c3 = cos(0)
    s3 = sin(0)
    mat = np.zeros((4, 4))

    mat[0, 0] = (c1 * c3) - (s1 * c2 * s3)
    mat[0, 1] = (-c1 * s3) - (s1 * c2 * c3)
    mat[0, 2] = (s1 * s2)
    mat[0, 3] = x[0]

    mat[1, 0] = (s1 * c3) + (c1 * c2 * s3)
    mat[1, 1] = (-s1 * s3) + (c1 * c2 * c3)
    mat[1, 2] = (-c1 * s2)
    mat[1, 3] = x[1]

    mat[2, 0] = (s2 * s3)
    mat[2, 1] = (s2 * c3)
    mat[2, 2] = c2
    mat[2, 3] = 0

    mat[3, 0] = mat[3, 1] = mat[3, 2] = 0
    mat[3, 3] = 1

    return mat


class ORBDetector:
    pp = np.array([0.0, 0.0, 0.0])  # The initial position and posture of the
    tm = np.eye(4)

    def __init__(self, depth_intrin, use_lm=USE_LM, inlier_thre=INLIER_DIST_THRE, max_dis=MAX_DIS,
                 min_dis=MIN_DIS, use_patch=True, use_blur=False):
        # Every frame has four attribute : color_frame, depth_frame, features, feature_descriptorss.
        self.first_color_frame = []
        self.second_color_frame = []
        self.first_depth_frame = []
        self.second_depth_frame = []
        self.featureFrame_first = []
        self.featureFrame_second = []
        self.featureDes_first = []
        self.featureDes_second = []
        self.feature_index_1 = []     # This variable stores the number of features in specific img_patch
        self.feature_index_2 = []

        self.depth_intrin = depth_intrin
        self.orb = cv2.ORB_create(nfeatures=FEATUREMAX, fastThreshold=THRESHHOLD,
                                  nlevels=8, scaleFactor=1.2)
        self.USE_LM = use_lm
        self.USE_PATCH = use_patch
        self.INLIER_DIST_THRE = inlier_thre
        self.USE_BLUR = use_blur
        self.min_dis = min_dis
        self.max_dis = max_dis
        self.score = []
        self.bfMatcher = cv2.BFMatcher_create(normType=cv2.NORM_HAMMING, crossCheck=True)
        self.match = []
        self.W = []
        self.best_matches = []

        # The following part stores the coordinate of the features
        self.camera_coordinate_first = []
        self.camera_coordinate_second = []
        self.camera_pixel_second = []
        self.camera_pixel_first = []

        # self.res is the brief for result, displace_mat is a 4*4 matrix representing the homogeneous transform matrix
        self.res = [0, 0, 0]
        self.displace_mat = np.eye(4)

        self.A = self.B = None
        self.optimized_result = None
        self.index_result = None

    def set_first_frame(self, color_frame, depth_frame):
        self.first_color_frame = np.asanyarray(color_frame.get_data())
        self.first_depth_frame = depth_frame

    def set_second_frame(self, color_frame, depth_frame):
        self.second_color_frame = np.asanyarray(color_frame.get_data())
        if self.USE_BLUR:
            self.second_color_frame = cv2.blur(self.second_color_frame, (3,3))
        self.second_depth_frame = depth_frame

    def reset_frame(self, color_frame_next, depth_frame_next):
        """This method is applied after each frame is processed intending for reduce the calculation cost
        Refer to the jounal : A2 step last paragraph"""
        self.featureDes_first = self.featureDes_second
        self.featureFrame_first = self.featureFrame_second
        self.first_color_frame = self.second_color_frame
        self.first_depth_frame = self.second_depth_frame
        self.feature_index_1 = self.feature_index_2

        self.second_color_frame = np.asanyarray(color_frame_next.get_data())
        if self.USE_BLUR:
            self.second_color_frame = cv2.blur(self.second_color_frame, (3,3))
        self.second_depth_frame = depth_frame_next
        # self.featureFrame_second, self.featureDes_second = self.orb.detectAndCompute(self.second_color_frame, None)
        self.detect_second_features()

    def detect_all_features(self):
        """For debugging in test.py, would not be called in read_from_bag or example_run"""
        self.featureFrame_first, self.featureDes_first = self.orb.detectAndCompute(self.first_color_frame, None)
        self.featureFrame_second, self.featureDes_second = self.orb.detectAndCompute(self.second_color_frame, None)

    def detect_second_features(self):
        """Detect features and calculate the descriptors"""
        # P.S. the features and descriptors of frame A are calculated beforehand
        # self.featureFrame_first, self.featureDes_first = self.orb.detectAndCompute(self.first_color_frame, None)
        if not self.USE_PATCH:
            self.featureFrame_second, self.featureDes_second = self.orb.detectAndCompute(self.second_color_frame, None)
        else:
            self.feature_index_2 = []
            self.featureFrame_second = []
            self.featureDes_second = []
            for y in range(0, WIN_H, TILE_H):
                for x in range(0, WIN_W, TILE_W):
                    img_patch = self.second_color_frame[y:y+TILE_H, x:x+TILE_W]
                    kpts = self.orb.detect(img_patch)
                    kpts, des = self.orb.compute(img_patch, kpts)
                    for pt in kpts:
                        pt.pt = (pt.pt[0] + x, pt.pt[1] + y)
                    if des is not None:
                        for i in range(len(des)):
                            self.featureDes_second.append(des[i])
                            self.featureFrame_second.append(kpts[i])
                        num = len(des)
                        self.feature_index_2.append(num)
                    else:
                        self.feature_index_2.append(0)
            self.featureDes_second = np.array(self.featureDes_second)

    def match_features(self):
        """This method match the features using BrutalForce and sort them by similarity
         and only take the strongest 50"""
        if self.featureDes_first is not None and self.featureDes_second is not None:
            if self.USE_PATCH:
                # IMPORTANT : match(queryDescriptors, trainDescriptors)
                matches = []
                index_1 = 0
                index_2 = 0
                for i in range(len(self.feature_index_2)):
                    if self.feature_index_1[i] == 0 or self.feature_index_2[i] == 0:
                        index_1 += self.feature_index_1[i]
                        index_2 += self.feature_index_2[i]
                        continue
                    if i == 0:
                        match = self.bfMatcher.match(np.array(self.featureDes_first[:self.feature_index_1[i]]),
                                                     np.array(self.featureDes_second[:self.feature_index_2[i]]))
                    else:
                        match = self.bfMatcher.match(np.array(self.featureDes_first[
                                                              index_1:index_1 + self.feature_index_1[i]]),
                                                     np.array(self.featureDes_second[
                                                              index_2:index_2 + self.feature_index_2[i]]))
                    for match_ in match:
                        if i == 0:
                            matches.append(match_)
                        else:
                            match_.queryIdx += index_1
                            match_.trainIdx += index_2
                            matches.append(match_)
                    index_1 += self.feature_index_1[i]
                    index_2 += self.feature_index_2[i]
                self.match = matches
            else:
                matches = self.bfMatcher.match(self.featureDes_first, self.featureDes_second)
                self.match = sorted(matches, key=lambda x: x.distance)
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

        self.camera_coordinate_first = new_camera_coor_first
        self.camera_coordinate_second = new_camera_coor_second
        self.camera_pixel_first = new_camera_pixel_first
        self.camera_pixel_second = new_camera_pixel_second
        self.match = new_matches

    def calculate_camera_coordinates(self, depth_to_color_extrin):
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
            self.camera_coordinate_first.append([point_a[0], point_a[2], point_a[1]])
            self.camera_pixel_second.append(point_b_pixel)
            self.camera_pixel_first.append(point_a_pixel)
            self.camera_coordinate_second.append([point_b[0], point_b[2], point_b[1]])
        self.match = match.copy()

        # Take the first 300 features to accelerate the algorithm
        if len(self.camera_coordinate_first) >= MAX_FEATURE:
            self.camera_coordinate_first = self.camera_coordinate_first[:MAX_FEATURE]
            self.camera_coordinate_second = self.camera_coordinate_second[:MAX_FEATURE]
            self.camera_pixel_first = self.camera_pixel_first[:MAX_FEATURE]
            self.camera_pixel_second = self.camera_pixel_second[:MAX_FEATURE]
            self.match = self.match[:MAX_FEATURE]

    def func(self, result):
        num_points = self.A.shape[0]
        # dimension = self.A.shape[1]
        matrix = cal_matrix_T(result)
        A_ = np.hstack((self.A, np.array([[1] for _ in range(num_points)])))
        # print(A_.T)
        B_ = np.hstack((self.B, np.array([[1] for _ in range(num_points)])))
        # print(B_.T)
        if not FOUR:
            error = np.square(A_.T - matrix.dot(B_.T)).flatten()
        else:
            error = np.sum(np.square(A_.T - matrix.dot(B_.T)), axis=1)

        # print(error)
        return error

    # @staticmethod
    # def rotate_matrix(axis, radian):
    #     """
    #     this function is deprecated for the low efficiency.
    #     :param axis: rotation axis
    #     :param radian: radian
    #     :return: 3-by-3 rotation matrix
    #     """
    #     return expm(np.cross(np.eye(3), axis / norm(axis) * radian))

    @staticmethod
    def getT(pp, three_d=False):
        """
        get the displace matrix of a given pp (position and posture)
        warning: this matrix can only be used when the coordinates y and z are reversed (for 2-d situations)
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
            return np.array([[cos(pp[2]), -sin(pp[2]), 0, pp[0]],
                             [sin(pp[2]), cos(pp[2]), 0, pp[1]],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]])

    @staticmethod
    def ransac_residual_func(x, cords=None, is_lm=False, three_d=False):
        """
        calculate the 3-dimensional residuals for a given model parameters x.
        If is_lm is True, this can be used in LM Least Squared method.
        :param x: np.array([<x>, <y>, <z>, <theta_x>, <theta_y>, <theta_z>]
        :param cords: np.array(<x1>, <y1>, <z1>, <x2>, <y2>, <z2>)
        :param is_lm: bool.
        :param three_d: bool.
        :return: residuals. If is_lm is true, it is an array of residuals.
        """
        result = []
        if is_lm:
            for cord in cords:
                if three_d:
                    result.extend((ORBDetector.getT(x, three_d=True).dot(np.append(cord[3:], 1)) - np.append(cord[:3],
                                                                                                             1))[:3])
                else:
                    result.extend([(cord[0] - (cos(x[2]) * cord[3] - sin(x[2]) * cord[5] + x[0])),
                                   (cord[2] - (sin(x[2]) * cord[3] + cos(x[2]) * cord[5] + x[1]))])
                """
                np.array([[c, 0, -s, d_x],
                          [0, 1, 0,  d_y],
                          [s, 0, c,  d_z],
                          [0, 0, 0,    1]])
                this might be different from right-hand axes, but here we consider x --> new_x, z --> new_y (2-d 
                situation), which equals:
                    np.array([[c, -s, d_x],
                              [s, c,  d_y],
                              [0, 0,    1]])
                """
            return np.array(result)
        else:
            for cord in cords:
                if three_d:
                    # TODO: add relative error to three_d.
                    result.extend((ORBDetector.getT(x, three_d=True).dot(np.append(cord[3:], 1)) - np.append(cord[:3],
                                                                                                             1))[:3])
                else:
                    result.extend([(cord[0] - (cos(x[2]) * cord[3] - sin(x[2]) * cord[5] + x[0]))/cord[0],
                                   (cord[2] - (sin(x[2]) * cord[3] + cos(x[2]) * cord[5] + x[1]))/cord[2]])
            return 0.5*np.sum(np.square(result))  # squares of relative error

    def optimize_ransac(self, three_d=False):
        """
        use lm (a sort of least squares) inside ransac each loop to solve the problem of outliers in matches.
        :return: The best position-posture variable: np.array([x, z, \theta])
        """
        cords = np.hstack((self.camera_coordinate_first, self.camera_coordinate_second))
        cords = cords[:, [0, 2, 1, 3, 5, 4]]  # revert the y, z to real camera axes
        """
        After convertion:
        cord_list: [<6-array>, 
                    <6-array>,
                    ...]
        """
        all_cord_indices = np.arange((len(cords)))
        # hyper-parameters:
        min_points = 4  # 4 points is enough to derive a unique model.
        if min_points > len(cords):
            print("Not enough model to derive a precise model")  # possible risk of wrong calculated pp
        max_iteration = 15
        threshold = 5e-5
        min_number_to_assert = int(0.7 * len(cords))
        iteration = 0
        best_pp = None
        # best_err = float('inf')
        best_num = 0

        x0 = np.zeros(3) if not three_d else np.zeros(6)
        while iteration < max_iteration:
            np.random.shuffle(all_cord_indices)
            maybe_inliers_indices = all_cord_indices[:min_points]
            other_indices = all_cord_indices[min_points:]
            maybe_inliers = cords[maybe_inliers_indices]
            other_cords = cords[other_indices]

            num_also_inliers = 0
            # Calculate the pp with selected (maybe) inliers
            pp = least_squares(self.ransac_residual_func, x0, method='lm',
                               kwargs={'cords': maybe_inliers, 'is_lm': True, 'three_d': three_d}).x

            # From other instances calculate the also inliers (maybe)
            for sample in other_cords:
                err = self.ransac_residual_func(pp, sample.reshape(1, 6), is_lm=False, three_d=three_d)
                if np.fabs(err) < threshold:
                    num_also_inliers += 1
                    # if also_inliers.size == 0:
                    #     also_inliers = sample.reshape(1, 6)
                    # else:
                    #     also_inliers = np.vstack((also_inliers, sample))

            # if also_inliers.shape[0] + min_points >= min_number_to_assert:
            if num_also_inliers + min_points >= min_number_to_assert:
                best_pp = pp
                print("assert.")
                break
            else:
                # this_err = self.ransac_residual_func(pp, cords=cords, is_lm=False, three_d=three_d)
                this_num = num_also_inliers
                if this_num > best_num:
                    best_pp = pp
                    best_num = this_num
            # print(best_err)
            iteration += 1

        self.displace_mat = ORBDetector.getT(best_pp, three_d=three_d)
        self.optimized_result = best_pp
        return best_pp

    def optimize(self):
        """LM method by scipy"""
        if self.USE_LM:
            self.A = np.array(self.camera_coordinate_first, dtype=np.float32)
            self.B = np.array(self.camera_coordinate_second, dtype=np.float32)
            if USE_LM:
                self.optimized_result = least_squares(self.func, x0=[0, 0, 0], method='lm').x
                T = cal_matrix_T(self.optimized_result)
            self.displace_mat = T

    def get_new_pp(self):
        ORBDetector.tm = np.dot(ORBDetector.tm, self.displace_mat)
        ORBDetector.pp = np.array([ORBDetector.tm[0, 3], ORBDetector.tm[1, 3], math.atan2(
            -ORBDetector.tm[0, 1], ORBDetector.tm[1, 1])])

    def check_estimate(self, threshhold_coord, threshhold_theta):
        delta_x = self.optimized_result[0]
        delta_y = self.optimized_result[1]
        delta_theta = self.optimized_result[2]
        if abs(delta_x) > threshhold_coord or abs(delta_y) > threshhold_coord or abs(delta_theta) > threshhold_theta:
            return False
        else:
            return True
