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
def plotLeastSquares(dic: dict, degree: int, x1, x2):
    x = np.arange(x1, x2, 0.01)

    plt.plot(x, LeastSquares(dic, degree, x), color="orange")

    plt.plot(list(dic), list(dic.values()), color="black")

    # plt.scatter(list(dic), list(dic.values()), color="black")

    # plt.plot(x, sin(x), color="blue")

    plt.show()


def plotLagrange(array, x1, x2):
    x = np.arange(x1, x2, 0.01)
    # plt.scatter(array, sin(array), color="black")

    # plt.plot(list(dic), list(dic.values()), color="black")

    plt.plot(x, Lagrange(x, array, sin), color="orange")
    plt.plot(x, sin(x), color="black")
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


def LeastSquares(dic: dict, degree: int, x: float):
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
    resultList = performGaussJordan(matrixToCalculateWith, vectorToCalculateWith)
    result = mkfun(resultList, x)
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
    # type: (list, float) -> float
    sum = 0
    for i in range(len(vector)):
        sum += vector[i] * (x ** i)
    return sum


class DayForecast:
    def __init__(self, dayNum: int, date: str, original: float):
        self.date = date
        self.dayNum = dayNum
        self.original = original
        self.approximation = None

    def diff(self):
        if self.original is not None and self.approximation is not None:
            return abs(self.original - self.approximation)
        return None


    # this is what is printed
    def __str__(self):
        string = "dayNum:" + str(self.dayNum) + "\ndate:" + str(self.date) + " original:" + str(
            self.original) + " approximation:" + str(self.approximation)
        diff = self.diff()
        if diff is not None:
            string += " diff:" + str(diff)
        return string + "\n\n"



def solveEydap(degree):
    # to learn
    # first point is equal to day 0
    # but day 2 is equal to 3 because it is 3 days after 0-day and so on
    dates = [  # dayNum,  date,   real value
        DayForecast(20, "14/2/2020", 75300),
        DayForecast(19, "13/2/2020", 74200),
        DayForecast(18, "12/2/2020", 75200),
        DayForecast(17, "11/2/2020", 74100),
        DayForecast(16, "10/2/2020", 73200),
        DayForecast(13, "07/2/2020", 74000),
        DayForecast(12, "06/2/2020", 75300),
        DayForecast(11, "05/2/2020", 75600),
        DayForecast(10, "04/2/2020", 75300),
        DayForecast(9 , "03/2/2020", 74400),
        DayForecast(6 , "30/1/2020", 74800),
        DayForecast(5 , "29/1/2020", 76400),
        DayForecast(4 , "28/1/2020", 76000),
        DayForecast(3 , "27/1/2020", 76000),
        DayForecast(0 , "24/1/2020", 76700)
    ]
    # set appropriate order
    dates.reverse()

    # pointDic is the dictionaly LeastSquares is going to use to make the approximation
    # we only give it the values up to 10
    counter = 0
    pointDic = {}
    for i in dates:
        if counter == 10:
            break
        pointDic[i.dayNum] = i.original
        counter += 1

    # get all approximations
    for i in dates:
        i.approximation = round(LeastSquares(pointDic, degree, i.dayNum), 5)
    # print dates
    for i in dates:
        print(i)


    plotLeastSquares(pointDic, degree, 0, 20)


def solveKarel(degree):
    # make dates
    # first point is equal to day 0
    # but day 2 is equal to 3 because it is 3 days after 0-day and so on
    dates = [    # dayNum,  date,   real value
        DayForecast(20, "14/2/2020", None),
        DayForecast(19, "13/2/2020", None),
        DayForecast(18, "12/2/2020", None),
        DayForecast(17, "11/2/2020", None),
        DayForecast(16, "10/2/2020", None),
        DayForecast(13, "07/2/2020", 2960000),
        DayForecast(12, "06/2/2020", 3000000),
        DayForecast(11, "05/2/2020", 3000000),
        DayForecast(10, "04/2/2020", 3000000),
        DayForecast(9 , "03/2/2020", 2980000),
        DayForecast(6 , "30/1/2020", 2900000),
        DayForecast(5 , "29/1/2020", 2920000),
        DayForecast(4 , "28/1/2020", 2920000),
        DayForecast(3 , "27/1/2020", 2920000),
        DayForecast(0 , "24/1/2020", 2920000)
    ]
    # pointDic is the dictionaly LeastSquares is going to use to make the approximation
    # we only give it the values up to 10
    pointDic = {}
    for i in dates:
        if i.original is not None:
            pointDic[i.dayNum] = i.original
    # get all approximations for all days
    for i in dates:
        i.approximation = round(LeastSquares(pointDic, degree, i.dayNum),5)

    # for i in dates:
    #     if i.original is None:
    # print dates
    for i in dates:
        print(i)

    # make a plot with everything
    plotLeastSquares(pointDic, degree, 0, 20)


def plotForDiffLagrange(points: list):
    # Lagrange
    # make a dictionary with 200 keys that each one of then points has as value the difference between
    # the lagrange-function for sin and the value for that key  original value of sin
    pointsToTest = {}
    for i in np.arange(-pi, pi, 2 * pi / 200):
        pointsToTest[i] = abs(sin(i) - Lagrange(i, points, sin))

    # two ways to display data
    # either make multiple plots from y = 0 to y = value for each key
    # or make a scatter for each key to the corresponding value

    # to review any of then uncomment the chosen and comment the other

    # for key in pointsToTest.keys():
    #     plt.plot([key, key], [0, pointsToTest[key]], color="blue")

    plt.scatter(list(pointsToTest), list(pointsToTest.values()), color="black")

    # prints the max and min diff between lagrange and sin
    print("max diff:", max(pointsToTest.values()), " min diff:", min(pointsToTest.values()))

    plt.show()


def plotForDiffLeastSquares(pointDic: dict, degree: int):
    # make a dictionary with 200 keys that each one of then points has as value the difference between
    # the LeastSquares-function for sin and the value for that key  original value of sin
    pointsToTest = {}
    for i in np.arange(-pi, pi, 2 * pi / 200):
        pointsToTest[i] = abs(sin(i) - LeastSquares(pointDic, degree, i))

    # two ways to display data
    # either make multiple plots from y = 0 to y = value for each key
    # or make a scatter for each key to the corresponding value

    # to review any of then uncomment the chosen and comment the other

    for key in pointsToTest.keys():
        plt.plot([key, key], [0, pointsToTest[key]], color="blue")

    # plt.scatter(list(pointsToTest), list(pointsToTest.values()), color="black")

    # prints the max and min diff between LeastSquares and sin
    print("max diff:", max(pointsToTest.values()), " min diff:", min(pointsToTest.values()))

    plt.show()


if __name__ == '__main__':

    solveEydap(3)

    # No 5
    if False:
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

        points.sort()

        pointDic = {}
        for i in range(len(points)):
            pointDic[points[i]] = sin(points[i])

        plotLeastSquares(pointDic, 2, -pi, pi)

        plotForDiffLeastSquares(pointDic, 2)

    # pointDic = {}
    # pointsToTest = {}
    # for i in range(len(points)):
    #     pointDic[points[i]] = sin(points[i])
    #
    # for i in np.arange(-pi, pi, 2 * pi / 200):
    #     pointsToTest[i] = abs(sin(i) - LeastSquares(pointDic, 3))

    # plotForDiff(pointsToTest)

    #
    # plotForDiff(pointsToTest)

    # plotForDiff(Lagrange, points, -pi,  pi)

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
