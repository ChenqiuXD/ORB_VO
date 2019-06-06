import numpy as np
import cv2
import os
from main import ORBDetector
import matplotlib.pyplot as plt

IS_CAMERA_CONNECTED = False
MAX_LENGTH = 10
MAX_ITER = 2000
USE_LM = True
GAP = 5
# DEPTH_SCALE = 0.001
PLOT_TRAJECTORY = False


def change_format(value):
    return ".%3f" % value


def project_pixel_to_point(w, intrin, pixel, depth):
    """Rewritten deprojection function by me.
     reference : https://github.com/IntelRealSense/librealsense/blob/5e73f7bb906a3cbec8ae43e888f182cc56c18692/include/librealsense2/rsutil.h#L15"""
    x = (pixel[0] - intrin.ppx) / intrin.fx
    y = (pixel[1] - intrin.ppy) / intrin.fy

    w[0] = depth*x
    w[1] = depth*y
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
    i = 0
    result = open("result.txt", "w")
    for pic_name in os.listdir('../data/rgb'):
        # Read in the picture, GAP pictures per read. Resize to promote calculation speed
        if i % GAP != 0:
            i += 1
            continue
        if i == 0:
            first_pic = cv2.imread('../data/rgb/' + pic_name)
            first_pic = cv2.resize(first_pic, (640, 480))
            depth_pic = cv2.imread('../data/depth/' + pic_name)
            depth_pic = cv2.resize(depth_pic, (640, 480))
        else:
            second_pic = cv2.imread('../data/rgb/' + pic_name)
            second_pic = cv2.resize(second_pic, (640, 480))
            depth_pic = cv2.imread('../data/depth/' + pic_name)
            depth_pic = cv2.resize(depth_pic, (640, 480))
        if i > MAX_ITER:
            break

        # Initialize the orbDetector and detect and match the features
        if i == 0:
            depth_intrin = Intrinsic()
            orb_detector = ORBDetector(depth_intrin, use_lm=True, max_dis=10, min_dis=0.05)
            orb_detector.second_color_frame = first_pic
            orb_detector.second_depth_frame = depth_pic
            orb_detector.detect_second_features()
            i += 1
            continue
        else:
            # This section corresponds to the self.reset_frame() method in orb.py
            orb_detector.first_color_frame = orb_detector.second_color_frame
            orb_detector.first_depth_frame = orb_detector.second_depth_frame
            orb_detector.featureDes_first = orb_detector.featureDes_second
            orb_detector.featureFrame_first = orb_detector.featureFrame_second

            orb_detector.second_color_frame = second_pic
            orb_detector.second_depth_frame = depth_pic
            orb_detector.detect_second_features()
            orb_detector.match_features()
            i += 1

        if orb_detector.match.__len__():
            orb_detector.find_inlier_without_depth()

            # The acquirement of depth_scale and depth_intrin is by debugging main.py
            depth_intrin = Intrinsic()

            # A changed implementation of ORBdetector.calculate_camera_coordinate with changes:
            # 1-calculate depth with depth pic
            for match in orb_detector.best_matches:
                img_pixel = [int(orb_detector.featureFrame_first[match.queryIdx].pt[0]),
                             int(orb_detector.featureFrame_first[match.queryIdx].pt[1])]
                depth = orb_detector.first_depth_frame[img_pixel[1], img_pixel[0]] / 255.0 * MAX_LENGTH
                depth = depth[0]    # For unknown reason, the depth pic is 3 channel which all have same value
                if depth >= orb_detector.max_dis or depth <= orb_detector.min_dis:
                    continue
                point_a = [0, 0, 0]
                project_pixel_to_point(point_a, depth_intrin, img_pixel, depth)

                img_pixel = [int(orb_detector.featureFrame_second[match.trainIdx].pt[0]),
                             int(orb_detector.featureFrame_second[match.trainIdx].pt[1])]
                depth = orb_detector.second_depth_frame[img_pixel[1], img_pixel[0]] / 255.0 * MAX_LENGTH
                depth = depth[0]
                if depth >= orb_detector.max_dis or depth <= orb_detector.min_dis:
                    continue
                point_b_pixel = img_pixel
                point_b = [0, 0, 0]
                project_pixel_to_point(point_b, depth_intrin, img_pixel, depth)

                orb_detector.camera_coordinate_first.append(point_a)
                orb_detector.camera_pixel_second.append(point_b_pixel)
                orb_detector.camera_coordinate_second.append(point_b)

            # An extension which dump all the data of listA and listB for debugging. USELESS during the optimization
            # file_a = open("listA.txt", "w")
            # formatted = [[change_format(v) for v in r] for r in optimizer.listA]
            # file_a.write(str(formatted))
            # file_a.close()
            # file_b = open("listB.txt", "w")
            # formatted = [[change_format(v) for v in r] for r in optimizer.listB]
            # file_b.write(str(formatted))
            # file_b.close()

            # Calculate the position
            orb_detector.optimize()
            orb_detector.get_new_pp()

            # Write the position into the file
            if PLOT_TRAJECTORY:
                content = str(ORBDetector.tm[0, 3]) + ' ' + str(ORBDetector.tm[2, 3])
                result.write(content)
                result.write('\n')
            else:
                content = str(orb_detector.displace_mat[0, 3]) + ' ' + str(orb_detector.displace_mat[2, 3])
                result.write(content)
                result.write('\n')

            print(str(i) + ' ' + pic_name + '\n')

            if len(orb_detector.best_matches) != 0:
                # Visualize the result of best matches
                image = cv2.drawMatches(orb_detector.first_color_frame, orb_detector.featureFrame_first,
                                        orb_detector.second_color_frame, orb_detector.featureFrame_second,
                                        orb_detector.best_matches, orb_detector.first_color_frame)
                if PLOT_TRAJECTORY:
                    for i in range(4):
                        text = str(orb_detector.tm[i, :])
                        cv2.putText(image, text, (40, 50+20*i), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 0, 255), 2)
                else:
                    text = str(orb_detector.res.x)
                    cv2.putText(image, text, (40, 50), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 0, 255), 2)
                    # for i in range(4):
                    #     text = str(orb_detector.displace_mat[i, :])
                    #     cv2.putText(image, text, (40, 50+20*i), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 0, 255), 2)
                cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('RealSense', image)
                cv2.waitKey(0)

    cv2.destroyAllWindows()
    result.close()






