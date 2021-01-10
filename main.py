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


def transp(A):
    if A is None:
        print("no matrix")
        return
    # len1 = len(A)
    # len2 = len(A[0])
    # if len1 != len2:
    #     print("matrix is not square")
    #     return

    transpose = np.empty(shape=[len(A[0]), len(A)], dtype=float)

    for i in range(len(A)):
       for y in range(len(A[0])):
        transpose[y][i] = A[i][y]


    return transpose


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


def LeastSquares(x, dic, degree, fun):
    # set A matrix, initialize with zeroes
    # s = (len(dic), len(dic))
    A = np.empty(shape=[len(dic), degree + 1], dtype=float)
    b = list()

    counter = 0
    for i in dic.keys():
        for y in range(degree + 1):
            try:
                A[counter][y] = i ** y
            except:
                print("error")
                print(i, " ", y, i ** y)

        b.append(fun(dic[i]))
        counter += 1

    return A


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

    # pointDic = {}
    pointDic = {-1: 1, 0: 0, 1: 0, 2: -2}
    # for i in range(len(points)):
    #     pointDic[points[i]] = sin(points[i])
    #

    print(LeastSquares(2, pointDic, 2, sin))
    print ("\n\n")
    print(transp(LeastSquares(2, pointDic, 2, sin)))
    #
    # for i in range(len(pointDic)):
    #     print(i, list(pointDic.values())[i], " ")
    #
    #

# plotter(Lagrange, points, -1.5*pi,  1.5*pi )


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
