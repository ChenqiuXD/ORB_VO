import numpy as np
import cv2
from ORB_VO.main import Optimizer, ORBDetector

IS_CAMERA_CONNECTED = False

if __name__ == "__main__":
    pic1 = cv2.imread("pic1.jpg")
    pic1 = cv2.resize(pic1, (640, 480))
    pic2 = cv2.imread("pic2.jpg")
    pic2 = cv2.resize(pic2, (640, 480))

    orb_detector = ORBDetector(pic1)
    orb_detector.detect_features()
    orb_detector.set_frame(pic2)
    orb_detector.detect_features()
    orb_detector.match_features()
    if orb_detector.match.__len__() != 0:
        orb_detector.find_inlier()

    # image = cv2.drawMatches(orb_detector.frameA, orb_detector.featureFrameA,
    #                         orb_detector.frameB, orb_detector.featureFrameB,
    #                         orb_detector.best_matches, orb_detector.frameA)
    # cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    # cv2.imshow('RealSense', image)
    # cv2.waitKey(0)

    # The acquirement of depth_scale and depth_intrin is by debugging main.py
    depth_intrin = [1, 1, 1]

    # Create a optimizer and find the displacement
    optimizer = Optimizer(orb_detector.featureFrameA, orb_detector.featureFrameB
                          , orb_detector.best_matches, depth_intrin)
    optimizer.get_list()
    optimizer.optimize()

