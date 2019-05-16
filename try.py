import numpy as np
import cv2
import pyrealsense2 as rs

THRESHHOLD = 30
FEATUREMAX = 50
INLIER_DIST_THRE = 10


class minimizer:
    def __init__(self, featuresA, featuresB, matches):
        self.featuresA = featuresA
        self.featuresB = featuresB
        self.matches = matches

    def estimate_motion_from_features(self):
        return None


class ORBDetector:
    def __init__(self, frame):
        self.featureFrameA = []
        self.featureFrameB = []
        self.featureDesA = []
        self.featureDesB = []
        self.frameA = []
        self.frameB = frame
        self.orb = cv2.ORB_create(nfeatures=FEATUREMAX, fastThreshold=THRESHHOLD)
        self.score = []
        self.bfMatcher = cv2.BFMatcher_create(normType=cv2.NORM_HAMMING, crossCheck=True)
        self.match = []
        self.W = []
        self.best_matches = []

    def detect_features(self):
        """Detect features and calculate the descriptors"""
        self.featureFrameB, self.featureDesB = self.orb.detectAndCompute(self.frameB, None)

    def match_features(self):
        """Match the features using BrtueForce and sort them by similarity. Store the first 50 matches"""
        matches = self.bfMatcher.match(queryDescriptor=self.featureDesA, trainDescriptor=self.featureDesB)
        self.match = sorted(matches, key=lambda x: x.distance)
        self.match = self.match[:50]

    def find_most_compatible_feature(self, candidate):
        best_featureIdx = -1
        best_featureVal = 0
        len_of_match = len(self.match)
        if not candidate:
            return None
        for i in candidate:
            if self.W[len_of_match+1][i] > best_featureVal:
                best_featureVal = self.W[len_of_match+1][i]
                best_featureIdx = i
        return best_featureIdx

    def find_inlier(self):
        """This function execute the A4 step of the journal"""
        len_of_matches = len(self.match)
        # The last line of W stores the number of consistency of this match
        self.W = np.zeros((len_of_matches+1, len_of_matches))
        for i in np.arange(len_of_matches):
            for j in np.arange(len_of_matches):
                if i <= j:
                    continue

                # ASSUMPTION : the index of descriptor is the same with the index of image
                wa = self.featureFrameA[self.match[i].queryIdx].pt[0]-self.featureFrameA[self.match[j].queryIdx].pt[0]
                wb = self.featureFrameA[self.match[i].queryIdx].pt[1]-self.featureFrameA[self.match[j].queryIdx].pt[1]
                wa_ = self.featureFrameB[self.match[i].trainIdx].pt[0]-self.featureFrameB[self.match[j].trainIdx].pt[0]
                wb_ = self.featureFrameB[self.match[i].trainIdx].pt[1]-self.featureFrameB[self.match[j].trainIdx].pt[1]

                # Compare and complete the matrix W
                if abs(wa-wa_) + abs(wb-wb_) <= INLIER_DIST_THRE:
                    self.W[i, j] = 1
                    self.W[j, i] = 1
                    self.W[len_of_matches+1, j] += 1

        # Choose the best inlier features
        self.best_matches = []
        candidate = np.arange(len_of_matches)
        while True:
            best_matchIdx = self.find_most_compatible_feature(candidate)
            if not best_matchIdx:    # in case no best feature is found
                break
            else:
                self.best_matches.append(self.featureFrameA[self.match[best_matchIdx].queryIdx])
                candidate = np.delete(candidate, np.where(candidate == best_matchIdx), axis=0)

    def set_frame(self, frame_next):
        """This function is applied after each frame is processed intending for reduce the calculation cost
        Refer to the jounal : A2 step last paragraph"""
        self.featureDesA = self.featureDesB
        self.featureFrameA = self.featureFrameB
        self.frameA = self.frameB
        self.frameB = frame_next


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

    iterCount = 0
    pos = [0, 0, 0]
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipe.wait_for_frames()

        # Align the depth frame and color frame
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not aligned_depth_frame or not color_frame:
            continue

        # Intrinsics & Extrinsics
        depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
        color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
        depth_to_color_extrin = aligned_depth_frame.profile.get_extrinsics_to(
            color_frame.profile)

        # Convert images to numpy arrays
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Detect the ORB features by Opencv2 using orb method and match them
        # Corresponding to the A2-A3 steps in the journal
        if iterCount == 0:
            orb_detector = ORBDetector(color_image)
        else:
            orb_detector.set_frame(color_image)
        orb_detector.detect_features()
        orb_detector.match_features()
        orb_detector.find_inlier()

        # Get the 3D coordinate of each feature
        # minimizer(orb_detector.featureFrameA, orb_detector.featureFrameB,
        #           orb_detector.best_matches)
        # motion = minimizer.estimate_motion_from_features()
        # pos = pos + motion


        # Draw the features on the image for debugging
        image = cv2.drawKeypoints(color_image, orb_detector.featureFrameB, color_image, color=(255, 0, 0))

        if iterCount <= 1000:
            iterCount += 1

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', image)
        cv2.imshow('RealSense', color_image)

        cv2.waitKey(1)

