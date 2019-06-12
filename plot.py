import numpy as np
import matplotlib.pyplot as plt
from read_from_bag import PLOT_TRAJECTORY
arr = np.loadtxt('result_new.txt')
if PLOT_TRAJECTORY:
    plt.plot(arr[:, 1], arr[:, 2])
else:
    plt.scatter(arr[:, 1], arr[:, 2])
plt.show()
