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


class Matrix:
    def __init__(self, original):
        # fParent = first parent
        # sParent = second >>
        self.A = np.array(original, dtype=float)
        self.L = None
        self.U = None
        self.P = None
        self.choleskyL = None

    def setLu(self, L, U):
        self.L = L
        self.U = U

    # this is what is printed
    def __str__(self):
        return "A\n" + str(self.A) + "\n\nP\n" + str(self.P) + "\n\nU\n" + str(self.U) + "\n\nL\n" + str(
            self.L) + "\n\ncholeskyL\n" + str(self.choleskyL) + "\n\n"

    def print(self):
        print("A\n", self.A)


def PALU(matrix):
    # check for existing matrix and if matrix is square
    if matrix.A is None:
        print("no matrix")
        return
    len1 = len(matrix.A)
    len2 = len(matrix.A[0])
    if len1 != len2:
        print("matrix is not square")
        return

    # initialize U as copy of A
    U = matrix.A.copy()

    # set L as an array of 0

    s = (len(U), len(U))
    L = np.zeros(s, dtype=float)

    # list to collect info about how to form P at the end
    listForP = list()
    # print(U, "\n\n")

    # U iteration
    for i in range(len(U)):
        # first we find the correct row
        rowOfMaxValue = i
        for y in range(i, len(U), 1):
            if abs(U[y][i]) > U[rowOfMaxValue][i]:
                rowOfMaxValue = y
        # swap rows in U
        temp = list(U[i])
        U[i] = U[rowOfMaxValue]
        U[rowOfMaxValue] = temp
        # swap rows is L
        temp = list(L[i])
        L[i] = L[rowOfMaxValue]
        L[rowOfMaxValue] = temp
        # keep info to later form P
        listForP.append(rowOfMaxValue)

        # make PU calculations
        # Gauss calculations
        for y in range(i + 1, len(U), 1):
            factor = U[y][i] / U[i][i]
            L[y][i] = factor
            for z in range(i, len(U), 1):
                U[y][z] -= factor * U[i][z]

    # construct P according to listForP
    P = np.identity(len(U))
    for i in range(len(U)):
        temp = list(P[i])
        P[i] = P[listForP[i]]
        P[listForP[i]] = temp

    # add identity matrix to L
    L += np.identity(len(U))
    # set matrices for object
    matrix.U = U
    matrix.P = P
    matrix.L = L


def calculateVector(matrix, vector):
    # checks if matrix A exists and if it is square
    if matrix.A is None:
        print("no matrix")
        return
    # len1 = len(matrix.A)
    # len2 = len(matrix.A[0])
    # if len1 != len2 or len(vector) != len1:
    #     print("matrix is not square or vector not right")
    #     return

    # check if PALU is calculated
    if matrix.U is None:
        PALU(matrix)

    # calculate vector for U
    vector = matrix.U.dot(vector)
    # print(vector)

    # calculate vector for L
    vector = matrix.L.dot(vector)
    # print(vector)

    # calculate vector for transpose of P
    # vector = matrix.P.transpose().dot(vector)
    vector = transp(matrix.P).dot(vector)
    # print(vector)

    return vector


# accepts a function, the range and then makes a plot using all that
def plotter(fun, array, x1, x2):
    x = np.arange(x1, x2, 0.01)
    # y = zeros(len(x))
    # plt.plot(x, y)
    # plt.scatter(array, sin(array), color="black")

    plt.plot(x, fun(array, x), color="orange")
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


def LeastSquares(x, dic, degree):
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

if __name__ == '__main__':
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
    # pointDic = {-1: 1, 0: 0, 1: 0, 2: -2}
    #
    # array = np.array([[9., 3., 4.],
    #                   [4., 3., 4.],
    #                   [1., 1., 1.]])
    # vector = [7, 8, 3]
    #
    # performGaussJordan(array, vector)

    pointDic = {}
    for i in range(len(points)):
        pointDic[points[i]] = sin(points[i])

    print(pointDic)

    # print(LeastSquares(2, pointDic, 2))

    results = LeastSquares(2, pointDic, 9)
    print(results)

    plotter(mkfun, results, -pi, pi)






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
