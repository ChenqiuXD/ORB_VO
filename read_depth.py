import cv2
depth_image = cv2.imread('data/depth/1558602009416.png')
print(depth_image.get_distance(0,0))
print(depth_image)
