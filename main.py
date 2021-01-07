from numpy import *
import numpy as np

from numpy import log as ln
import math

import matplotlib.pyplot as plt
import sys

pi = np.pi


# accepts a function, the range and then makes a plot using all that
def plotter(fun, array, x1, x2):
    x = np.arange(x1, x2, 0.01)
    # y = zeros(len(x))
    # plt.plot(x, y)
    plt.plot(x, fun(x), color="orange")

    plt.scatter(array, fun(array), color="black")

    plt.show()


if __name__ == '__main__':
    # for i in range(10):
    #     print(round(random.uniform(-pi, pi), 4))

    arrayList = [2.9193,
-1.9475,
0.5656,
-1.379,
2.1096,
-0.2275,
0.0781,
1.1325,
2.7807,
-0.6981,
]

    plotter(np.sin, arrayList, -pi, pi)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
