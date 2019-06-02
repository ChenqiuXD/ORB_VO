import cv2
import pyrealsense2 as rs
import numpy as np
from ORB_VO.main import ORBDetector
from math import sin,cos
from ORB_VO.pso import PSO
from scipy.optimize import least_squares
USE_LM =True
BAG_NAME = '20190602_095040.bag'
MAX_DIS = 4
MIN_DIS = 0.5
GAP = 20
PLOT_TREJACTORY = True

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

    # f = open('result_new.txt','w+')
    # Skip the first five frame for stable usage.
    for i in np.arange(20):
        frames = p.wait_for_frames()
    f = open('result_new.txt', 'w')
    iterCount = 0
    while iterCount<2000:
        # Wait for a coherent pair of frames: depth and color
        frames = p.wait_for_frames()
        if iterCount%GAP !=0:
            iterCount += 1
            continue
            
        # Align the depth frame and color frame
        aligned_frames = align.process(frames)
        second_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not second_depth_frame or not color_frame:
            continue

        # Intrinsics & Extrinsics
        depth_intrin = second_depth_frame.profile.as_video_stream_profile().intrinsics
        color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
        depth_to_color_extrin = second_depth_frame.profile.get_extrinsics_to(
            color_frame.profile)

        if iterCount == 0:
            orb_detector = ORBDetector(depth_intrin=depth_intrin,use_lm=USE_LM)
            orb_detector.set_second_frame(color_frame=color_frame,depth_frame=second_depth_frame)
            orb_detector.detect_second_features()
        else:
            # Update a new frame by set_frame()
            orb_detector.reset_frame(color_frame_next=color_frame,depth_frame_next=second_depth_frame)
            orb_detector.match_features()
            if orb_detector.match.__len__() != 0:
                orb_detector.find_inlier()

            # Draw the features on the image for debugging
            # image = cv2.drawKeypoints(color_image, orb_detector.featureFrameB, color_image, color=(255, 0, 0))
            image = cv2.drawMatches(orb_detector.first_color_frame, orb_detector.featureFrame_first,
                                    orb_detector.second_color_frame, orb_detector.featureFrame_second,
                                    orb_detector.best_matches, orb_detector.first_color_frame)
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', image)
            cv2.waitKey(10)

            # Optimize to calculate the transition matrix
            orb_detector.calculate_camera_coordinates()
            if len(orb_detector.camera_coordinate_first) >= 3:
                orb_detector.optimize()
                orb_detector.get_new_pp()
                print(iterCount,ORBDetector.pp)
                if USE_LM:
                    if not PLOT_TREJACTORY:
                        result = str(orb_detector.res.x[0] )+ ' ' + str(orb_detector.res.x[1])
                    else:
                        result = str(ORBDetector.pp[0]) + ' ' + str(ORBDetector.pp[1])

                else:
                    if not PLOT_TREJACTORY:
                        result = str(orb_detector.optimized_result[1]) + ' ' + str(orb_detector.optimized_result[2])
                    else:
                        result = str(ORBDetector.pp[0]) + ' ' + str(ORBDetector.pp[1])
                f.write(result)
                f.write("\n")


        # Update the iterCount
        if iterCount <= 10000:
            iterCount += 1
        orb_detector.best_matches = []


        # cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        # cv2.imshow('RealSense', depth_image)
        # cv2.waitKey(10)
    f.close()
    p.stop()
