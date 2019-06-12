import numpy as np
import matplotlib.pyplot as plt
from read_from_bag import PLOT_TRAJECTORY
arr = np.loadtxt('result_new.txt')
# set the lim of x and y axis
fig = plt.figure()
# plt.xlim([-3, 0.5])
# plt.ylim([-0.5, 3])
if PLOT_TRAJECTORY:
    plt.plot(arr[:, 0], arr[:, 1])
else:
    plt.scatter(arr[:, 0], arr[:, 1])
plt.show()
