import cv2
import pyrealsense2 as rs
import numpy as np
from ORB_VO.main import ORBDetector
from math import sin,cos
from ORB_VO.pso import PSO

from scipy.optimize import least_squares

USE_LM =True
BAG_NAME = '20190601_230135.bag'
MAX_DIS = 2.5


class Optimizer:
    pp = np.array([0.0, 0.0, 0.0])  # The initial position and posture of the
    def __init__(self, featureA, featureB, matches, intrinsic, depth_frameA,depth_frameB,use_lm = USE_LM):
        self.USE_LM = use_lm
        self.featureA = featureA
        self.featureB = featureB
        self.matches = matches
        self.depth_frameA = depth_frameA
        self.depth_frameB = depth_frameB
        self.listA = []
        self.listB = []
        # Added by RK
        self.optimized_result = None
        # cam, with original theta being zero
        self.intrin = intrinsic
        self.res = [0, 0, 0]

    def get_list(self):
        """This method get the list A and B by rs.deproject function"""
        for match in self.matches:
            img_pixel = [int(self.featureA[match.queryIdx].pt[0]), int(self.featureA[match.queryIdx].pt[1])]
            depth = self.depth_frameA.get_distance(img_pixel[0], img_pixel[1])
            if depth >=6 or depth<=0.1:
                continue
            # print(depth)
            point_a = rs.rs2_deproject_pixel_to_point(self.intrin, img_pixel, depth)
            point_a = [point_a[0], point_a[2], 1]
            img_pixel = [int(self.featureB[match.trainIdx].pt[0]), int(self.featureB[match.trainIdx].pt[1])]
            if depth >=6 or depth<=0.1:
                continue
            # print(depth)
            depth = self.depth_frameB.get_distance(img_pixel[0], img_pixel[1])
            point_b = rs.rs2_deproject_pixel_to_point(self.intrin, img_pixel, depth)
            point_b = [point_b[0], point_b[2], 1]
            self.listA.append(point_a)
            self.listB.append(point_b)

    def get_new_pp(self):
        # cam_displace = self.optimized_result[[1, 2, 0]]
        if USE_LM:
            Optimizer.pp[2] += self.res.x[2]
            tm = np.array([[np.cos(Optimizer.pp[2]), -np.sin(Optimizer.pp[2])],
                          [np.sin(Optimizer.pp[2]), np.cos(Optimizer.pp[2])]])
            Optimizer.pp[:2] += tm.dot(self.res.x[:2])
        else:
            Optimizer.pp[2] += self.optimized_result[0]
            tm = np.array([[np.cos(Optimizer.pp[2]), -np.sin(Optimizer.pp[2])],
                           [np.sin(Optimizer.pp[2]), np.cos(Optimizer.pp[2])]])
            Optimizer.pp[:2] += tm.dot(self.optimized_result[1:3])

    def func(self, x):
        """Cost function used for optimization. Variables are defined as follows:
        x = [delta_x, delta_y, delta_theta]
        self.listA = [x, y, 1]
        self.listB = [x, y, 1]"""
        result = []
        for j in np.arange(self.listA.__len__()):
            result.append(self.listB[j][0] - (cos(x[2])*self.listA[j][0] - sin(x[2])*self.listA[j][1] + x[0]))
            result.append(self.listB[j][1] - (sin(x[2])*self.listA[j][0] + cos(x[2])*self.listA[j][1] + x[1]))
        return np.asarray(result)

    def optimize(self):
        """LM method by scipy"""
        if self.USE_LM:
            x0 = [0, 0, 0]
            self.res = least_squares(self.func, x0, method='lm')
            self.res.x *= 100
        else:
            pso_optimizer = PSO( population_size=10, max_steps=50, pA=self.listA, pB=self.listB)
            self.optimized_result = pso_optimizer.evolve()
            self.optimized_result[1:3] = 100*self.optimized_result[1:3]

if __name__ == "__main__":
    p = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device_from_file(BAG_NAME)
    prof = p.start(cfg)

    prof.get_device().as_playback().set_real_time(False)
    depth_sensor = prof.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    # Align object creation
    align_to = rs.stream.color
    align = rs.align(align_to)

    f = open('result_new.txt','w+')
    # Skip the first five frame for stable usage.
    for i in np.arange(5):
        frames = p.wait_for_frames()

    iterCount = 0
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = p.wait_for_frames()
        if iterCount%3 !=0:
            iterCount += 1
            continue


        # Align the depth frame and color frame
        aligned_frames = align.process(frames)
        second_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not second_depth_frame or not color_frame:
            iterCount += 1
            continue



        # Intrinsics & Extrinsics
        depth_intrin = second_depth_frame.profile.as_video_stream_profile().intrinsics
        color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
        depth_to_color_extrin = second_depth_frame.profile.get_extrinsics_to(
            color_frame.profile)

        # Convert images to numpy arrays
        depth_image = np.asanyarray(second_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        if iterCount == 0:
            orb_detector = ORBDetector(color_image)
            orb_detector.detect_features()
            first_depth_frame = second_depth_frame
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
            cv2.waitKey(0)

            # Optimize to calculate the transition matrix
            optimizer = Optimizer(orb_detector.featureFrameA, orb_detector.featureFrameB
                              , orb_detector.best_matches, depth_intrin,depth_frameA=first_depth_frame,depth_frameB=second_depth_frame)
        if iterCount != 0:
            optimizer.get_list()
            if len(optimizer.listA) >= 3:
                optimizer.optimize()
                optimizer.get_new_pp()
                print(iterCount,optimizer.pp)
                result = str(optimizer.res.x[0] )+ ' ' + str(optimizer.res.x[1])
                f.write(result)
                f.write("\n")

        # Update the iterCount
        if iterCount <= 10000:
            iterCount += 1
            first_depth_frame = second_depth_frame
        orb_detector.best_matches = []


        # cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        # cv2.imshow('RealSense', depth_image)
        # cv2.waitKey(10)
    p.stop()
