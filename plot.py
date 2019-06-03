import numpy as np
import matplotlib.pyplot as plt
from ORB_VO.read_from_bag import PLOT_TRAJECTORY
arr = np.loadtxt('result_new.txt')
if PLOT_TRAJECTORY:
    plt.plot(arr[:, 0], arr[:, 1])
else:
    plt.scatter(arr[:, 0], arr[:, 1])
plt.show()
