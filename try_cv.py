import cv2
import numpy as np
# cv2.find
first = np.array([[1,2,1],[1,3,1],[2,7,1]])
second = np.array([[3],[4],[8]])
a = cv2.estimateAffine3D(first,second)
cv2.solvePnPRansac()
pass
