# encoding: UTF-8

import numpy as np
import matplotlib.pyplot as plt
# plt.switch_backend('agg')
# import time


def draw_trace(new_pp):
    draw_trace.all_pp = np.vstack((draw_trace.all_pp, new_pp))

    plt.cla()
    plt.plot(draw_trace.all_pp[:, 0], draw_trace.all_pp[:, 1])
    # plt.show()
    plt.pause(0.001)


draw_trace.all_pp = np.array([0, 0, 0])


if __name__ == "__main__":
    x = np.arange(1, 10)
    y = np.sin(x)
    fig = plt.figure()
    plt.plot(x, y)
    for i in range(100):
        plt.cla()
        # plt.figure(fig.number)
        x = np.append(x, 10+i)
        y = np.sin(x)
        plt.plot(x, y)
        plt.show()
        plt.pause(0.001)
        # time.sleep(0.01)
