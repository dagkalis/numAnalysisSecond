from numpy import *
import numpy as np

import time

time.time()

from numpy import log as ln
import math

import matplotlib.pyplot as plt
import sys

pi = np.pi

# the function
# None until it is assigned
f = None


# accepts a function, the range and then makes a plot using all that
def plotter(fun, array, x1, x2):
    x = np.arange(x1, x2, 0.01)
    # y = zeros(len(x))
    # plt.plot(x, y)
    plt.scatter(array, fun(array), color="black")

    plt.plot(x, fun(x), color="orange")

    plt.show()

def funMaker(functionString):
    d = {}
    new_func = functionString
    the_code = compile(new_func, 'test', 'exec')
    exec(the_code, d)
    return d['next_element']


if __name__ == '__main__':
    # for i in range(10):
    #     print(round(random.uniform(-pi, pi), 4))

    arrayList = [2.9193,
                 -1.9475,
                 -1.379,
                 2.1096,
                 -0.2275,
                 0.0781,
                 1.1325,
                 2.7807,
                 -0.6981,
                 1.0045]

    functionString = 'def next_element(x):\n  return x+1'

    f = funMaker(functionString)


    print(f(5))


    #     exec(new_func)
    #
    # timestamp2 = time.time()
    # print ("This took %.2f seconds" % (timestamp2 - timestamp1))
    #
    # timestamp1 = time.time()

    #
    # timestamp2 = time.time()
    # print("This took %.2f seconds" % (timestamp2 - timestamp1))

    # f = next_element
    # print(f(1))

    # print(next_element(1))

    # mathFunctionCompiler()

    # exec(mathFunctionCompiler(5 + 5))
    # print(thunder(5))

    # new_func = 'def ty(x):\n return x + 1'
    # the_code = compile(new_func, 'test', 'exec')
    # exec(the_code)
    # print(ty(5))

    # plotter(np.sin, arrayList, -pi, pi)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
