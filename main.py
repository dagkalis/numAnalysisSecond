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

    # plt.plot(list(dic), list(dic.values()), color="black")

    plt.scatter(list(dic), list(dic.values()), color="black")

    # plt.plot(x, sin(x), color="blue")

    plt.show()


def plotLagrange(array, x1, x2):
    x = np.arange(x1, x2, 0.01)
    plt.scatter(array, sin(array), color="black")

    # plt.plot(list(dic), list(dic.values()), color="black")

    # plt.plot(x, sin(x), color="black")
    plt.plot(x, Lagrange(x, array, sin), color="orange")

    plt.show()


# gets a python function in string and dynamically compiles it
# returns the python function (def)
# def defMaker(functionString):
#     d = {}
#     new_func = functionString
#     the_code = compile(new_func, 'test', 'exec')
#     exec(the_code, d)
#     return d['next_element']


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

    # calculate A^TA
    matrixToCalculateWith = transp(A).dot(A)
    # calculate A^Tb
    vectorToCalculateWith = transp(A).dot(b)
    # calculate with gauss-jordan
    resultList = performGaussJordan(matrixToCalculateWith, vectorToCalculateWith)
    # use mkfun to calculate final-result
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
        # will be set later
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
    # make dates
    dates = [  # dayNum,  date,   real value
        DayForecast(15, "14/2/2020", 75300),
        DayForecast(14, "13/2/2020", 74200),
        DayForecast(13, "12/2/2020", 75200),
        DayForecast(12, "11/2/2020", 74100),
        DayForecast(11, "10/2/2020", 73200),
        DayForecast(10, "07/2/2020", 74000),
        DayForecast(9 , "06/2/2020", 75300),
        DayForecast(8 , "05/2/2020", 75600),
        DayForecast(7 , "04/2/2020", 75300),
        DayForecast(6 , "03/2/2020", 74400),
        DayForecast(5 , "30/1/2020", 74800),
        DayForecast(4 , "29/1/2020", 76400),
        DayForecast(3 , "28/1/2020", 76000),
        DayForecast(2 , "27/1/2020", 76000),
        DayForecast(1 , "24/1/2020", 76700)
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


    plotLeastSquares(pointDic, degree, 0, 15)


def solveKarel(degree):
    # make dates
    dates = [    # dayNum,  date,   real value
        DayForecast(15, "14/2/2020", 2980000),
        DayForecast(14, "13/2/2020", 2980000),
        DayForecast(13, "12/2/2020", 2980000),
        DayForecast(12, "11/2/2020", 2900000),
        DayForecast(11, "10/2/2020", 2920000),
        DayForecast(10, "07/2/2020", 2960000),
        DayForecast(9 , "06/2/2020", 3000000),
        DayForecast(8 , "05/2/2020", 3000000),
        DayForecast(7 , "04/2/2020", 3000000),
        DayForecast(6 , "03/2/2020", 2980000),
        DayForecast(5 , "30/1/2020", 2900000),
        DayForecast(4 , "29/1/2020", 2920000),
        DayForecast(3 , "28/1/2020", 2920000),
        DayForecast(2 , "27/1/2020", 2920000),
        DayForecast(1 , "24/1/2020", 2920000)
    ]

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


    # get all approximations for all days
    for i in dates:
        i.approximation = round(LeastSquares(pointDic, degree, i.dayNum),5)

    # for i in dates:
    #     if i.original is None:
    # print dates
    counter = 0
    for i in dates:
        if counter == 10:
            print ("\n\nfrom now on we have forecasts\n\n\n\n")
        print(i)
        counter += 1

    # make a plot with everything
    plotLeastSquares(pointDic, degree, 0, 15)


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

    for key in pointsToTest.keys():
        plt.plot([key, key], [0, pointsToTest[key]], color="blue")

    # plt.scatter(list(pointsToTest), list(pointsToTest.values()), color="black")

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

def solveSinWithLeastSquares(points: list, degree: int):
    points.sort()

    # leastSquares for sin
    pointDic = {}
    for i in range(len(points)):
        pointDic[points[i]] = sin(points[i])

    # plotLeastSquares(pointDic, degree, -pi, pi)

    plotForDiffLeastSquares(pointDic, degree)

def solveSinWithLagrange(points: list):

    # plotLagrange(points, -pi, pi)

    plotForDiffLagrange(points)



if __name__ == '__main__':

    # Execise 5
    points = [2.9193,
              -1.9475,
              -1.379,
              2.1096,
              -0.2275,
              0.0781,
              1.1325,
              2.7807,
              -2.99,
              1.0045]

    # 1)
    # solveSinWithLagrange(points)
    # 3)
    solveSinWithLeastSquares(points, 3)


    # Exerice 7

    # solveKarel(4)

    # solveEydap(2)


