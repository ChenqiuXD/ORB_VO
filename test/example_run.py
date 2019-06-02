import numpy as np
import cv2
import os
from ORB_VO.main import Optimizer, ORBDetector
import matplotlib.pyplot as plt

IS_CAMERA_CONNECTED = False
MAX_LENGTH = 2
MAX_ITER = 500
USE_LM  = True
GAP = 2


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
        if i%GAP  != 0:
            i += 1
            continue
        if i == 0:
            first_pic = cv2.imread('../data/rgb/' + pic_name)
            first_pic = cv2.resize(first_pic, (640, 480))
            i += 1
            continue
        if i > MAX_ITER:
            break
        second_pic = cv2.imread('../data/rgb/'+pic_name)
        second_pic = cv2.resize(second_pic, (640, 480))
        depth_pic = cv2.imread('../data/depth/'+pic_name)
        depth_pic = cv2.resize(depth_pic, (640, 480))
        """some code here"""
        orb_detector = ORBDetector(first_pic)
        orb_detector.detect_features()
        orb_detector.set_frame(second_pic)
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
            depth_intrin = Intrinsic()

            # Create a optimizer and find the displacement
            optimizer = Optimizer(orb_detector.featureFrameA, orb_detector.featureFrameB
                                  , orb_detector.best_matches, depth_intrin,use_lm=USE_LM)

            for match in optimizer.matches:
                img_pixel = [int(optimizer.featureA[match.queryIdx].pt[0]),
                             int(optimizer.featureA[match.queryIdx].pt[1])]
                depth = depth_pic[img_pixel[1], img_pixel[0]] / 255.0 * MAX_LENGTH
                depth = depth[0]
                point_a = [0, 0, 0]
                project_pixel_to_point(point_a, optimizer.intrin, img_pixel, depth)
                point_a = [point_a[0], point_a[2], 1]
                img_pixel = [int(optimizer.featureB[match.trainIdx].pt[0]),
                             int(optimizer.featureB[match.trainIdx].pt[1])]
                depth = depth_pic[img_pixel[1], img_pixel[0]] / 255.0 * MAX_LENGTH
                depth = depth[0]
                point_b = [0, 0, 0]
                project_pixel_to_point(point_b, optimizer.intrin, img_pixel, depth)
                point_b = [point_b[0], point_b[2], 1]
                optimizer.listA.append(point_a)
                optimizer.listB.append(point_b)
            optimizer.optimize()
            # print(optimizer.res.x)

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
            optimizer.get_new_pp()
            pp = optimizer.pp
            # Write the position into the file
            print(pic_name,pp)
            content =  str(pp[0]) +' '+str(pp[1])+'\n'
            result.write(content)

        """"""
        first_pic = second_pic
        i += 1
    arr = np.loadtxt('result.txt')
    plt.plot(arr[:, 0], arr[:, 1])
    plt.show()





