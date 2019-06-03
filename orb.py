import numpy as np
import pyrealsense2 as rs
import cv2
from math import sin, cos
from ORB_VO.pso import PSO
from scipy.optimize import least_squares

USE_LM = True
THRESHHOLD = 30
FEATUREMAX = 200
INLIER_DIST_THRE = 3
MAX_DIS = 4
MIN_DIS = 0.5

class ORBDetector:
    pp = np.array([0.0, 0.0, 0.0])  # The initial position and posture of the

    def __init__(self, depth_intrin, use_lm=USE_LM,inlier_thre = INLIER_DIST_THRE,max_dis=MAX_DIS,min_dis=MIN_DIS):
        self.first_color_frame = []
        self.second_color_frame = []
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
        self.world_coordinate_first = []
        self.camera_coordinate_first = []
        self.world_coordinate_second = []
        self.camera_coordinate_second = []
        self.res = [0, 0, 0]

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

    def detect_second_features(self):
        """Detect features and calculate the descriptors"""
        # P.S. the features and descriptors of frame A are calculated beforehand
        # self.featureFrame_first, self.featureDes_first = self.orb.detectAndCompute(self.first_color_frame, None)
        self.featureFrame_second, self.featureDes_second = self.orb.detectAndCompute(self.second_color_frame, None)

    def match_features(self):
        """This method match the features using BrutalForce and sort them by similarity
         and only take the strongest 50"""
        type_of_None = type(None)
        if type(self.featureDes_first) != type_of_None and type(self.featureDes_second) != type_of_None:
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
            depth = self.first_depth_frame.get_distance(img_pixel[0], img_pixel[1])
            if depth >= self.max_dis or depth <= self.min_dis:
                print(depth)
                continue
            # print(depth)
            point_a = rs.rs2_deproject_pixel_to_point(self.depth_intrin, img_pixel, depth)
            # threeD_file.write(str(point_a[1]))
            # threeD_file.write("\n")
            point_a = [point_a[0], point_a[2], 1]
            img_pixel = [int(self.featureFrame_second[match.trainIdx].pt[0]),
                         int(self.featureFrame_second[match.trainIdx].pt[1])]
            if depth >= self.max_dis or depth <= self.min_dis:
                print(depth)
                continue
            # print(depth)
            depth = self.second_depth_frame.get_distance(img_pixel[0], img_pixel[1])
            point_b = rs.rs2_deproject_pixel_to_point(self.depth_intrin, img_pixel, depth)
            # threeD_file.write(str(point_b[1]))
            # threeD_file.write("\n")
            point_b = [point_b[0], point_b[2], 1]
            self.camera_coordinate_first.append(point_a)
            self.camera_coordinate_second.append(point_b)

    def func(self, x):
        """Cost function used for optimization. Variables are defined as follows:
        x = [delta_x, delta_y, delta_theta]
        self.camera_coordinate_first = [x, y, 1]
        self.listB = [x, y, 1]"""
        result = []
        for j in np.arange(self.camera_coordinate_first.__len__()):
            result.append(self.camera_coordinate_second[j][0] - (
                        cos(x[2]) * self.camera_coordinate_first[j][0] - sin(x[2]) * self.camera_coordinate_first[j][
                    1] + x[0]))
            result.append(self.camera_coordinate_second[j][1] - (
                        sin(x[2]) * self.camera_coordinate_first[j][0] + cos(x[2]) * self.camera_coordinate_first[j][
                    1] + x[1]))
        return np.asarray(result)

    def optimize(self):
        """LM method by scipy"""
        if self.USE_LM:
            x0 = [0, 0, 0]
            self.res = least_squares(self.func, x0, method='lm')
            self.res.x *= 100
        else:
            pso_ORBDetector = PSO(population_size=10, max_steps=50, pA=self.camera_coordinate_first, pB=self.listB)
            self.optimized_result = pso_ORBDetector.evolve()
            self.optimized_result[1:3] = 100 * self.optimized_result[1:3]

    def get_new_pp(self):
        # cam_displace = self.optimized_result[[1, 2, 0]]
        if self.USE_LM:
            ORBDetector.pp[2] += self.res.x[2]
            tm = np.array([[np.cos(ORBDetector.pp[2]), -np.sin(ORBDetector.pp[2])],
                           [np.sin(ORBDetector.pp[2]), np.cos(ORBDetector.pp[2])]])
            ORBDetector.pp[:2] += tm.dot(self.res.x[:2])
        else:
            ORBDetector.pp[2] += self.optimized_result[0]
            tm = np.array([[np.cos(ORBDetector.pp[2]), -np.sin(ORBDetector.pp[2])],
                           [np.sin(ORBDetector.pp[2]), np.cos(ORBDetector.pp[2])]])
            ORBDetector.pp[:2] += tm.dot(self.optimized_result[1:3])