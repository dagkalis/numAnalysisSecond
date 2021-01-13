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
def plotter(fun, array, dic, x1, x2):
    x = np.arange(x1, x2, 0.01)
    # y = zeros(len(x))
    # plt.plot(x, y)
    # plt.scatter(array, sin(array), color="black")

    plt.plot(x, fun(array, x), color="orange")
    plt.plot(list(dic), list(dic.values()), color="black")

    # plt.plot(x, sin(x), color="black")

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


def LeastSquares(dic, degree):
    # initialize matrix and b list
    A = np.empty(shape=[len(dic), degree + 1], dtype=float)
    b = list()

    # make array
    counter = 0
    for i in dic.keys():
        for y in range(degree + 1):
            try:
                A[counter][y] = i ** y
            except:
                print("error")
                print(i, " ", y, i ** y)
        # add to b the solution of the i key
        b.append((dic[i]))
        counter += 1

    matrixToCalculateWith = transp(A).dot(A)
    vectorToCalculateWith = transp(A).dot(b)
    result = performGaussJordan(matrixToCalculateWith, vectorToCalculateWith)
    return result


def performGaussJordan(array, vector):
    if array is None:
        return
    A = array
    # iterate pivot lines
    for i in range(len(A)):
        # iterate all other lines
        for y in range(len(A)):
            # don't mess with pivot line
            if i != y:
                factor = A[y][i] / A[i][i]
                # iterate remaining columns
                for z in range(i, len(A), 1):
                    # print("was:", "A[y][z]:", A[y][z], "factor:", factor, "A[i][z]:", A[i][z])
                    A[y][z] -= factor * A[i][z]
                    # print("is:","A[y][z]:", A[y][z], "factor:", factor, "A[i][z]:", A[i][z])
                vector[y] -= factor * vector[i]
            # print("i:", i, "y:", y)

    solutionVector = []
    for i in range(len(vector)):
        solutionVector.append(vector[i] / A[i][i])
    # print(solutionVector)
    return solutionVector


def mkfun(vector, x):
    sum = 0
    for i in range(len(vector)):
        sum += vector[i] * (x ** i)
    return sum

def solveEydap(degree):
    # to learn
    # first point is equal to day 0
    # but day 2 is equal to 3 because it is 3 days after 0-day and so on
    pointDic = {13: 74000,  # 07 / 2 / 2020
                12: 75300,  # 06 / 2 / 2020
                11: 75600,  # 05 / 2 / 2020
                10: 75300,  # 04 / 2 / 2020
                9: 74400,  # 03 / 2 / 2020
                6: 74800,  # 30 / 1 / 2020
                5: 76400,  # 29 / 1 / 2020
                4: 76000,  # 28 / 1 / 2020
                3: 76000,  # 27 / 1 / 2020
                0: 76700  # 24 / 1 / 2020
                }

    #calucate LeastSquares with points we have for degree (parameter)
    results = LeastSquares(pointDic, degree)
    print(results)


    # make forecasts
    forecasts = []
    # days to forecast
    # same logic to days-numbering as pointDic
    forecastDays = [20,         #14/2/2020
                    19,         #13/2/2020
                    18,         #12/2/2020
                    17,         #11/2/2020
                    16]         #10/2/2020
    for i in range(len(forecastDays)):
        forecasts.append(round(mkfun(results, forecastDays[i]), 6))

    print(forecasts)

    daysDic = {20: "14 / 2 / 2020",
               19: "13 / 2 / 2020",
               18: "12 / 2 / 2020",
               17: "11 / 2 / 2020",
               16: "10 / 2 / 2020",
               13: "07 / 2 / 2020",
               12: "06 / 2 / 2020",
               11: "05 / 2 / 2020",
               10: "04 / 2 / 2020",
               9:  "03 / 2 / 2020",
               6:  "30 / 1 / 2020",
               5:  "29 / 1 / 2020",
               4:  "28 / 1 / 2020",
               3:  "27 / 1 / 2020",
               0:  "24 / 1 / 2020"}



    plotter(mkfun, results, pointDic, 0, 18)


def solveKarel(degree):
    # to learn
    # first point is equal to day 0
    # but day 2 is equal to 3 because it is 3 days after 0-day and so on
    dayCountTodayDic = {
        0: '24/01/2020',
        3: '27/01/2020',
        4: '28/01/2020',
        5: '29/01/2020',
        6: '30/01/2020',
        10: '03/02/2020',
        11: '04/02/2020',
        12: '05/02/2020',
        13: '06/02/2020',
        14: '07/02/2020',
        17: '10/02/2020',
        18: '11/02/2020',
        19: '12/02/2020',
        20: '13/02/2020',
        21: '14/02/2020'}


    pointDic = {
                0:  2960000,
                3: 3000000,
                4: 3000000,
                5: 3000000,
                6: 2980000,
                10: 2900000,
                11: 2920000,
                12: 2920000,
                13: 2920000,
                14: 2920000,
    }

    # calucate LeastSquares with points we have for degree (parameter)
    results = LeastSquares(pointDic, degree)
    print(results)

    # make forecasts
    forecasts = []
    # days to forecast
    # same logic to days-numbering as pointDic
    forecastDays = [17,  # 14/2/2020
                    18,  # 13/2/2020
                    19,  # 12/2/2020
                    20,  # 11/2/2020
                    21]  # 10/2/2020
    for i in range(len(forecastDays)):
        forecasts.append(round(mkfun(results, forecastDays[i]), 6))

    print(forecasts)

    daysDic = {20: "14 / 2 / 2020",
               19: "13 / 2 / 2020",
               18: "12 / 2 / 2020",
               17: "11 / 2 / 2020",
               16: "10 / 2 / 2020",
               13: "07 / 2 / 2020",
               12: "06 / 2 / 2020",
               11: "05 / 2 / 2020",
               10: "04 / 2 / 2020",
               9: "03 / 2 / 2020",
               6: "30 / 1 / 2020",
               5: "29 / 1 / 2020",
               4: "28 / 1 / 2020",
               3: "27 / 1 / 2020",
               0: "24 / 1 / 2020"}

    plotter(mkfun, results, pointDic, 0, 18)




if __name__ == '__main__':



    solveKarel(4)



    # points = [2.9193,
    #           -1.9475,
    #           -1.379,
    #           2.1096,
    #           -0.2275,
    #           0.0781,
    #           1.1325,
    #           2.7807,
    #           -0.6981,
    #           1.0045]


    # pointDic = {}
    # pointDic = {-1: 1, 0: 0, 1: 0, 2: -2}
    #
    # array = np.array([[9., 3., 4.],
    #                   [4., 3., 4.],
    #                   [1., 1., 1.]])
    # vector = [7, 8, 3]
    #
    # performGaussJordan(array, vector)

    # pointDic = {}
    # for i in range(len(points)):
    #     pointDic[points[i]] = sin(points[i])
    #
    # # print(LeastSquares(2, pointDic, 2))
    #
    # results = LeastSquares(pointDic, 3)
    # print(results)
    #
    # plotter(mkfun, results, -pi, pi)

    # print("\n\n")
    # print(transp(LeastSquares(2, pointDic, 2, sin)))
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
