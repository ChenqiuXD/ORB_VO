# -*- coding: utf-8 -*-

import numpy as np
import cv2
import pyrealsense2 as rs
from pso import PSO


THRESHHOLD = 30
FEATUREMAX = 200
INLIER_DIST_THRE = 10


class Optimizer:
    def __init__(self, featureA, featureB, matches, intrin):
        self.featureA = featureA
        self.featureB = featureB
        self.matches = matches
        self.listA = []
        self.listB = []
        self.intrin = intrin
        # Added by RK
        self.optimized_result = None
        self.pp = np.array([197.176, 162.371, 0])  # The initial position and posture of the
        # cam, with original theta being zero

    def get_list(self):
        """This method get the list A and B by rs.deproject function"""
        for match in self.matches:
            img_pixel = [int(self.featureA[match.queryIdx].pt[0]), int(self.featureA[
                                                                           match.queryIdx].pt[1])]
            depth = aligned_depth_frame.get_distance(img_pixel[0], img_pixel[1])
            point_a = rs.rs2_deproject_pixel_to_point(self.intrin, img_pixel, depth)
            point_a = [point_a[0], point_a[2], 1]
            img_pixel = [int(self.featureB[match.trainIdx].pt[0]), int(self.featureB[
                                                                           match.trainIdx].pt[1])]
            depth = aligned_depth_frame.get_distance(img_pixel[0], img_pixel[1])
            point_b = rs.rs2_deproject_pixel_to_point(self.intrin, img_pixel, depth)
            point_b = [point_b[0], point_b[2], 1]
            self.listA.append(point_a)
            self.listB.append(point_b)

    def optimize(self):
        """PSO method"""
        self.optimized_result = PSO(population_size=100,max_steps=10000,pA=self.listA,
                                    pB=self.listB).evolve()

    def get_new_pp(self):
        cam_displace = self.optimized_result[[1, 2, 0]]
        self.pp[2] += cam_displace[2]
        tm = np.array([[np.cos(self.pp[2]), -np.sin(self.pp[2])],
                      [np.sin(self.pp[2]), np.cos(self.pp[2])]])
        self.pp[:2] = tm.dot(cam_displace[:2])


class ORBDetector:
    def __init__(self, frame):
        self.featureFrameA = []
        self.featureFrameB = []
        self.featureDesA = []
        self.featureDesB = []
        self.frameA = []
        self.frameB = frame
        self.orb = cv2.ORB_create(nfeatures=FEATUREMAX, fastThreshold=THRESHHOLD)
        self.score = []
        self.bfMatcher = cv2.BFMatcher_create(normType=cv2.NORM_HAMMING, crossCheck=True)
        self.match = []
        self.W = []
        self.best_matches = []

    def detect_features(self):
        """Detect features and calculate the descriptors"""
        # P.S. the features and descriptors of frame A are calculated beforehand
        self.featureFrameB, self.featureDesB = self.orb.detectAndCompute(self.frameB, None)

    def match_features(self):
        """This method match the features using BrutalForce and sort them by similarity
         and only take the strongest 50"""
        type_of_None = type(None)
        if type(self.featureDesA) != type_of_None and type(self.featureDesB) != type_of_None:
            matches = self.bfMatcher.match(self.featureDesA, self.featureDesB)
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
        self.W = np.zeros((len_of_matches+1, len_of_matches))
        for i in np.arange(len_of_matches):
            for j in np.arange(len_of_matches):
                if i >= j:
                    continue

                # ASSUMPTION : the index of descriptor is the same with the index of image
                wa = self.featureFrameA[self.match[i].queryIdx].pt[0]-self.featureFrameA[self.match[j].queryIdx].pt[0]
                wb = self.featureFrameA[self.match[i].queryIdx].pt[1]-self.featureFrameA[self.match[j].queryIdx].pt[1]
                wa_ = self.featureFrameB[self.match[i].trainIdx].pt[0]-self.featureFrameB[self.match[j].trainIdx].pt[0]
                wb_ = self.featureFrameB[self.match[i].trainIdx].pt[1]-self.featureFrameB[self.match[j].trainIdx].pt[1]

                # Compare and complete the matrix W
                if abs(wa-wa_) + abs(wb-wb_) <= INLIER_DIST_THRE:
                    self.W[i, j] = 1
                    self.W[j, i] = 1
                    self.W[len_of_matches, j] += 1

        # Choose the best inlier features
        self.best_matches = []
        candidate = np.arange(len_of_matches)
        while True:
            best_matchIdx = self.find_most_compatible_match(candidate)
            if not best_matchIdx or best_matchIdx == -1:    # in case no best match is found
                break
            else:
                self.best_matches.append(self.match[best_matchIdx])
                candidate = np.delete(candidate, np.where(candidate == best_matchIdx), axis=0)

    def set_frame(self, frame_next):
        """This method is applied after each frame is processed intending for reduce the calculation cost
        Refer to the jounal : A2 step last paragraph"""
        self.featureDesA = self.featureDesB
        self.featureFrameA = self.featureFrameB
        self.frameA = self.frameB
        self.frameB = frame_next


if __name__ == "__main__":
    pipe = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipe.start(config)

    # Unused line, intending for access to the intrinsic parameter of the camera
    profile = pipe.get_active_profile()

    # Getting the depth sensor's depth scale. Real dist / scale = depth_frame_dist
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    # Align object creation
    align_to = rs.stream.color
    align = rs.align(align_to)

    # Skip the first five frame for stable usage.
    for i in np.arange(5):
        frames = pipe.wait_for_frames()

    iterCount = 0
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipe.wait_for_frames()

        # Align the depth frame and color frame
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not aligned_depth_frame or not color_frame:
            continue

        # Intrinsics & Extrinsics
        depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
        color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
        depth_to_color_extrin = aligned_depth_frame.profile.get_extrinsics_to(
            color_frame.profile)

        # Convert images to numpy arrays
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Detect the ORB features by Opencv2 using orb method and match them
        # Corresponding to the A2-A3 steps in the journal
        if iterCount == 0:
            orb_detector = ORBDetector(color_image)
            orb_detector.detect_features()
        else:
            # Update a new frame by set_frame()
            orb_detector.set_frame(color_image)
            orb_detector.detect_features()
            orb_detector.match_features()
            if orb_detector.match.__len__() != 0:
                orb_detector.find_inlier()

        # Draw the features on the image for debugging
        # image = cv2.drawKeypoints(color_image, orb_detector.featureFrameB, color_image, color=(255, 0, 0))
        if iterCount != 0:
            image = cv2.drawMatches(orb_detector.frameA, orb_detector.featureFrameA,
                                    orb_detector.frameB, orb_detector.featureFrameB,
                                    orb_detector.best_matches, orb_detector.frameA)
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', image)
            cv2.waitKey(10)

        # Optimize to calculate the transition matrix
        optimizer = Optimizer(orb_detector.featureFrameA, orb_detector.featureFrameB, orb_detector.best_matches,
                              depth_scale, depth_intrin)
        if iterCount != 0:
            optimizer.get_list()
            optimizer.optimize()

        # Update the iterCount
        # print(orb_detector.best_matches)
        # print(orb_detector.featureFrameA)
        if iterCount <= 1000:
            iterCount += 1
        orb_detector.best_matches = []

