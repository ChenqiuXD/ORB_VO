import cv2
from ORB_VO.orb import ORBDetector
import numpy as np
import math
from scipy.optimize import least_squares

PLOT_TRAJECTORY = True


def project_pixel_to_point(w, intrin, pixel, depth):
    """Rewritten deprojection function by me.
     reference : https://github.com/IntelRealSense/librealsense/blob/5e73f7bb906a3cbec8ae43e888f182cc56c18692/include/librealsense2/rsutil.h#L15"""
    x = (pixel[0] - intrin.ppx) / intrin.fx
    y = (pixel[1] - intrin.ppy) / intrin.fy

    w[0] = depth * x
    w[1] = depth * y
    w[2] = depth


class Intrinsic:
    def __init__(self):
        self.fx = 616.54541015625
        self.fy = 616.6361694335938
        self.height = 480
        self.width = 640
        self.ppx = 323.82830810546875
        self.ppy = 230.9412078857422
        self.model = -1


if __name__ == "__main__":
    pic1 = cv2.imread("pic3.jpg")
    pic1 = cv2.resize(pic1, (640, 480))
    pic2 = cv2.imread("pic4.jpg")
    pic2 = cv2.resize(pic2, (640, 480))

    intrin = Intrinsic()
    orb_detector = ORBDetector(intrin, use_lm=True, min_dis=0)

    orb_detector.first_color_frame = pic1
    orb_detector.second_color_frame = pic2
    orb_detector.detect_all_features()
    orb_detector.match_features()
    orb_detector.find_inlier_without_depth()

    list_a = []
    list_b = []
    for match in orb_detector.match:
        img_pixel = [int(orb_detector.featureFrame_first[match.queryIdx].pt[0]),
                     int(orb_detector.featureFrame_first[match.queryIdx].pt[1])]
        depth = 0.2
        point_a = [0, 0, 0]
        project_pixel_to_point(point_a, orb_detector.depth_intrin, img_pixel, depth)
        # threeD_file.write(str(point_a[1]))
        # threeD_file.write("\n")
        point_a = [point_a[0], point_a[1], point_a[2]]

        point_b_pixel = [int(orb_detector.featureFrame_second[match.trainIdx].pt[0]),
                         int(orb_detector.featureFrame_second[match.trainIdx].pt[1])]
        depth = 0.2
        point_b = [0, 0, 0]
        project_pixel_to_point(point_b, orb_detector.depth_intrin, point_b_pixel, depth)
        list_a.append(point_a)
        list_b.append(point_b_pixel)

        orb_detector.camera_coordinate_first.append(point_a)
        orb_detector.camera_coordinate_second.append(point_b)

    list_a = np.array(list_a, dtype=np.float32).reshape((50, 1, 3))
    list_b = np.array(list_b, dtype=np.float32).reshape((50, 1, 2))
    camera_mat = np.array([[intrin.fx, 0, intrin.ppx],
                           [0, intrin.fy, intrin.ppy],
                           [0, 0, 1]])
    dist = np.zeros(5)
    retval, rvec, tvec, _ = cv2.solvePnPRansac(list_a, list_b, camera_mat, distCoeffs=dist)
    rvec, _ = cv2.Rodrigues(rvec)
    temp = np.hstack((rvec, tvec))
    orb_detector.displace_mat = np.vstack((temp, [0, 0, 0, 1]))
    orb_detector.get_new_pp()

    x0 = [tvec[0], tvec[2], math.atan2(rvec[0, 2], rvec[0, 0])]
    orb_detector.res = least_squares(orb_detector.func, x0, method='lm')
    print(orb_detector.res.x)

    if len(orb_detector.best_matches) != 0:
        # Visualize the result of best matches
        image = cv2.drawMatches(orb_detector.first_color_frame, orb_detector.featureFrame_first,
                                orb_detector.second_color_frame, orb_detector.featureFrame_second,
                                orb_detector.best_matches, orb_detector.first_color_frame)
        if PLOT_TRAJECTORY:
            for i in range(4):
                text = str(orb_detector.tm[i, :])
                cv2.putText(image, text, (40, 50 + 20 * i), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 0, 255), 2)
        else:
            for i in range(4):
                text = str(orb_detector.displace_mat[i, :])
                cv2.putText(image, text, (40, 50+20*i), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 0, 255), 2)
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', image)
        cv2.waitKey(0)
