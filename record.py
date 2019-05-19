import pyrealsense2 as rs
import numpy as np
import cv2

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_record_to_file('test.bag')

# Start streaming
pipeline.start(config)

e1 = cv2.getTickCount()

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        # frames = pipeline.wait_for_frames()
        # depth_frame = frames.get_depth_frame()
        # color_frame = frames.get_color_frame()
        # if not depth_frame or not color_frame:
        #     continue

        e2 = cv2.getTickCount()
        t = (e2 - e1) / cv2.getTickFrequency()
        if t>30: # change it to record what length of video you are interested in
            print("Done!")
            break

finally:

    # Stop streaming
    pipeline.stop()