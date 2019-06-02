# -*- coding: utf-8 -*-

import numpy as np
import cv2
import pyrealsense2 as rs
from ORB_VO.orb import ORBDetector

THRESHHOLD = 30
FEATUREMAX = 200
INLIER_DIST_THRE = 3
USE_LM =True
BAG_NAME = '20190602_095040.bag'
MAX_DIS = 4
MIN_DIS = 0.5
GAP = 60
PLOT_TREJACTORY = True
# threeD_file = open('3D_file.txt','a')





if __name__ == "__main__":
    pipe = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipe.start(config)

    # Getting the depth sensor's depth scale. Real dist / scale = depth_frame_dist
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    # Align object creation
    align_to = rs.stream.color
    align = rs.align(align_to)

    # Skip the first five frame for stable usage.
    for i in np.arange(5):
        frames = pipe.wait_for_frames()

    # Open a file record.txt for the recording of result
    f = open("record.txt", "w")

    iterCount = 0
    while iterCount <= 100:
        # Wait for a coherent pair of frames: depth and color
        frames = pipe.wait_for_frames()

        # Align the depth frame and color frame
        aligned_frames = align.process(frames)
        second_depth_frame = aligned_frames.get_depth_frame()
        second_color_frame = aligned_frames.get_color_frame()
        if not second_depth_frame or not second_color_frame:
            continue

        # Intrinsics & Extrinsics
        depth_intrin = second_depth_frame.profile.as_video_stream_profile().intrinsics
        color_intrin = second_color_frame.profile.as_video_stream_profile().intrinsics
        depth_to_color_extrin = second_depth_frame.profile.get_extrinsics_to(
            second_color_frame.profile)

        # Detect the ORB features by Opencv2 using orb method and match them
        # Corresponding to the A2-A3 steps in the journal
        if iterCount == 0:
            orb_detector = ORBDetector(depth_intrin=depth_intrin, use_lm=USE_LM,inlier_thre=INLIER_DIST_THRE,max_dis=MAX_DIS,min_dis=MIN_DIS)
            orb_detector.set_second_frame(color_frame=second_color_frame,depth_frame=second_depth_frame)
            orb_detector.detect_second_features()

        # Draw the features on the image for debugging
        # image = cv2.drawKeypoints(color_image, orb_detector.featureFrameB, color_image, color=(255, 0, 0))
        else:
            # Update a new frame by set_frame()
            orb_detector.reset_frame(color_frame_next=second_color_frame,depth_frame_next=second_depth_frame)
            orb_detector.match_features()
            if orb_detector.match.__len__() != 0:
                orb_detector.find_inlier()

            #     visualize the color frame with best_maches
            image = cv2.drawMatches(orb_detector.first_color_frame, orb_detector.featureFrame_first,
                                    orb_detector.second_color_frame, orb_detector.featureFrame_second,
                                    orb_detector.best_matches, orb_detector.first_color_frame)
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', image)
            cv2.waitKey(10)

        # Optimize to calculate the transition matrix
            orb_detector.calculate_camera_coordinates()
            if orb_detector.camera_coordinate_first.__len__() >= 3:
                orb_detector.optimize()
                result = str(orb_detector.res.x)
                f.write(result)
                f.write("\n")

        # Update the iterCount
        if iterCount <= 10000:
            iterCount += 1
        orb_detector.best_matches = []
