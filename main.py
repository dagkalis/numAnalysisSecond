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
    plt.scatter(array, sin(array), color="black")

    plt.plot(x, fun(x, points, sin), color="orange")

    plt.show()


# gets a python function in string and dynamically compiles it
# returns the python function (def)
def defMaker(functionString):
    d = {}
    new_func = functionString
    the_code = compile(new_func, 'test', 'exec')
    exec(the_code, d)
    return d['next_element']


def Lagrange(x, points, function):
    toReturn = 0
    for i in range(len(points)):
        Li = 1
        for y in range(len(points)):
            if y != i:
                try:
                    Li *= (x - points[y]) / (points[i] - points[y])
                except:
                    print("Error")
                    print(i, " ", y, " ", points[i], " ", points[y])
        toReturn += function(points[i]) * Li
    return toReturn


if __name__ == '__main__':
    # for i in range(10):
    #     print(round(random.uniform(-pi, pi), 4))

    points = [2.9193,
              -1.9475,
              -1.379,
              2.1096,
              -0.2275,
              0.0781,
              1.1325,
              2.7807,
              -0.6981,
              1.0045]




    # plotter(sin, array, -5*pi, 5*pi )
    plotter(Lagrange, points, -1.5*pi,  1.5*pi )






# functionString = 'def next_element(x):\n  return x+1'
    #
    # f = defMaker(  functionString)

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


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
