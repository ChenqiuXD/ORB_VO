import numpy as np
import matplotlib.pyplot as plt

arr = np.loadtxt('result.txt')
plt.plot(arr[:, 0], arr[:, 1])
plt.show()
