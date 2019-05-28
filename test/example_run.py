import numpy as np
import cv2
import pyrealsense2 as rs
from main import Optimizer, ORBDetector

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

