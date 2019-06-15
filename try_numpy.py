import numpy as np
import pyrealsense2 as rs
rs.rs2_project_point_to_pixel()

# a = np.arange(12).reshape((3,4))
sss = np.array([[]])
# print(np.append(sss,np.array([1,2])))
sss = np.concatenate((sss,np.array([1,2])),axis=1)
sss = np.concatenate((sss,np.array([1,2])))
print(sss)
sss = np.delete(sss,np.argwhere(sss==6.32589))
print(sss.round(3))
sss.round(3)
