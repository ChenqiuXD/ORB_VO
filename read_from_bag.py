import cv2
import pyrealsense2 as rs
import numpy as np
from orb import ORBDetector, optimize_after
from draw_animation import draw_trace

USE_LM = True
BAG_NAME = 'o.bag'
MAX_DIS = 5
MIN_DIS = 0.1
INLIER_THRE = 0.5
GAP = 3
threshhold_coord = 0.1
threshhold_theta = np.pi/36
PLOT_TRAJECTORY = True
MAX_ITER = 15000
WAIT_KEY = 2
PRINT_DELTA = False
USE_RANSAC = False
USE_RANSAC_2 = False


def main():
    file_path = 'bag/' + BAG_NAME

    p = rs.pipeline()
    cfg = rs.config()
    # cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    # cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    cfg.enable_device_from_file(file_path)
    prof = p.start(cfg)

    prof.get_device().as_playback().set_real_time(False)
    depth_sensor = prof.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    # Align object creation
    align_to = rs.stream.color
    align = rs.align(align_to)

    # Skip the first five frame for stable usage.
    for i in np.arange(300):
        frames = p.wait_for_frames()
    f = open('result_new.txt', 'w')
    iterCount = 0
    while iterCount < MAX_ITER:
        # Wait for a coherent pair of frames: depth and color
        frames = p.wait_for_frames()

        # Align the depth frame and color frame
        aligned_frames = align.process(frames)
        second_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        time = color_frame.timestamp
        if not second_depth_frame or not color_frame:
            continue

        # Intrinsics & Extrinsics
        depth_intrin = second_depth_frame.profile.as_video_stream_profile().intrinsics
        color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
        depth_to_color_extrin = second_depth_frame.profile.get_extrinsics_to(
            color_frame.profile)

        if iterCount == 0:
            orb_detector = ORBDetector(depth_intrin=depth_intrin, use_lm=USE_LM, min_dis=MIN_DIS,
                                       max_dis=MAX_DIS, inlier_thre=INLIER_THRE)
            orb_detector.set_second_frame(color_frame=color_frame,
                                          depth_frame=second_depth_frame)
            orb_detector.detect_second_features()
            iterCount += 1
            continue
        elif iterCount % GAP != 0:
            iterCount += 1
            continue
        else:
            # Update a new frame by set_frame()
            orb_detector.reset_frame(color_frame_next=color_frame, depth_frame_next=second_depth_frame)

            orb_detector.match_features()
            orb_detector.calculate_camera_coordinates(depth_to_color_extrin=depth_to_color_extrin)
            if len(orb_detector.match) != 0:
                # orb_detector.simple_match_filter(threshhold=GAP*0.05)
                orb_detector.find_inlier_3d()
            else:
                print("初始关键点不足")
                orb_detector.match = []
                orb_detector.best_matches = []
                continue

            # Optimize to calculate the transition matrix
            if len(orb_detector.camera_coordinate_first) >= 3:
                if USE_RANSAC:
                    orb_detector.optimize_ransac(three_d=True)
                elif USE_RANSAC_2:
                    orb_detector.optimize_ransac2()
                else:
                    orb_detector.optimize()

                # Check the extimated motion
                judge = orb_detector.check_estimate(threshhold_coord=threshhold_coord,
                                                    threshhold_theta=threshhold_theta)
                if not judge:   # When it is not passed
                    orb_detector.match = []
                    orb_detector.best_matches = []
                    print("变化过快")
                    continue
                orb_detector.get_new_pp()
                draw_trace(orb_detector.pp)

                # Debug information print
                if USE_LM:
                    if not PLOT_TRAJECTORY:
                        result = str(time) + ' ' + str(ORBDetector.pp[0]) + ' ' + str(ORBDetector.pp[1])
                    else:
                        result = str(time) + ' ' + str(ORBDetector.tm[0, 3]) + ' ' + str(ORBDetector.tm[1, 3])

                else:
                    if not PLOT_TRAJECTORY:
                        result = str(time) + ' ' + str(orb_detector.optimized_result[1]) + ' ' + str(
                            orb_detector.optimized_result[2])
                    else:
                        result = str(time) + ' ' + str(ORBDetector.pp[0]) + ' ' + str(ORBDetector.pp[1])
                print(str(iterCount) + ' ' + result)
                f.write(result)
                f.write("\n")

                if PRINT_DELTA:
                    for i in range(len(orb_detector.camera_coordinate_first)):
                        print('X:' + str(
                            orb_detector.camera_coordinate_first[i][0] - orb_detector.camera_coordinate_second[i][0]))
                        print('y:' + str(
                            orb_detector.camera_coordinate_first[i][1] - orb_detector.camera_coordinate_second[i][1]))
                        print('z:' + str(
                            orb_detector.camera_coordinate_first[i][2] - orb_detector.camera_coordinate_second[i][2]))

            else:
                print("关键点不足")
                orb_detector.best_matches = []
                orb_detector.match = []
                continue

            # Draw the features on the image for debugging
            key = draw_image(orb_detector)
            if key == 27:
                break

            # Clear the matches
            orb_detector.best_matches = []
            iterCount += 1

    cv2.destroyAllWindows()
    f.close()
    p.stop()


def draw_image(orb_detector):
    image = cv2.drawMatches(orb_detector.first_color_frame, orb_detector.featureFrame_first,
                            orb_detector.second_color_frame, orb_detector.featureFrame_second,
                            orb_detector.best_matches, orb_detector.first_color_frame)
    if PLOT_TRAJECTORY:
        text2 = str(ORBDetector.pp)
        for i in range(4):
            text1 = str(orb_detector.tm[i, :])
            cv2.putText(image, text1, (40, 50 + 20 * i), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 0, 255), 2)
        cv2.putText(image, text2, (40, 50 + 20 * 4), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 0, 255), 2)
        text3 = 'camera coordinates amount:' + str(len(orb_detector.camera_coordinate_first))
        cv2.putText(image, text3, (40, 50 + 20 * 5), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 0, 255), 2)

    else:
        text2 = str(ORBDetector.pp)
        for i in range(4):
            text = str(orb_detector.displace_mat[i, :])
            cv2.putText(image, text, (40, 50 + 20 * i), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 0, 255), 2)
        cv2.putText(image, text2, (40, 50 + 20 * 4), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 0, 255), 2)
    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('RealSense', image)
    key = cv2.waitKey(WAIT_KEY)
    return key


if __name__ == "__main__":
    main()
