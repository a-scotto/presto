#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on April 23, 2019 at 16:38.

@author: a.scotto

Description:
"""

import numpy
import copy

import sys
sys.path.append('/home/disc/a.scotto/Documents/doctorat/python/presto')

from time import time
from numpy.linalg import norm, qr
from linear_map import DiagonalMap
from matrix_operator import MatrixOperator, UpperTriangularMap, SelfAdjointMap


def validate_matvec(A, x, exact_res):
    t = time()
    y = A.dot(x)
    delta = time() - t
    r = norm(y - exact_res)
    print(' Time: {:1.2e}  |  Residual: {:1.2e}'.format(delta, r))


def validate_matmat(A, B, x, exact_res):
    t = time()
    y = A.dot(B.dot(x))
    delta = time() - t
    r = norm(y - exact_res)
    print(' Time: {:1.2e}  |  Residual: {:1.2e}'.format(delta, r))


size = 5000

a = numpy.random.randint(0, size, size=(size, size))
A = MatrixOperator(a)

diag = [i + 1 for i in range(size)]
d = numpy.diag(diag)
D = DiagonalMap(diag)

r = numpy.triu(numpy.ones_like(a)) / size
R = UpperTriangularMap(r)

j = int(size/2)
x = numpy.eye(size)[:, [j]]

# ~~~~~~~~~~~~~~~~~~~ MATRIX ~~~~~~~~~~~~~~~~~~~
print('~ Matrix object validation')
validate_matvec(a, x, a[:, [j]])
validate_matvec(A, x, a[:, [j]])
validate_matvec(a.T, x, a[[j], :].T)
validate_matvec(A.adjoint(), x, a[[j], :].T)

print()
# ~~~~~~~~~~~~~~~~~~ DIAGONAL ~~~~~~~~~~~~~~~~~~
print('~ Diagonal object validation')
validate_matmat(a, d, x, d[j, j] * a[:, [j]])
validate_matmat(A, D, x, d[j, j] * a[:, [j]])
validate_matmat(a, d.dot(d.T), x, d[j, j]**2 * a[:, [j]])
validate_matmat(A, D * D.adjoint(), x, d[j, j]**2 * a[:, [j]])

print()
# ~~~~~~~~~~~~~~ UPPER TRIANGULAR ~~~~~~~~~~~~~~
a = numpy.ones_like(r)
A = MatrixOperator(a)
x = numpy.ones((size, 1))
print('~ UpperTriangular object validation')
e = numpy.asarray([(size - i) / size for i in range(size)]).reshape(size, 1)
validate_matvec(r, x, e)
validate_matvec(R, x, e)
e = size * numpy.asarray([(size - i) / size for i in range(size)]).reshape(size, 1)
validate_matmat(r, a, x, e)
validate_matmat(R, A, x, e)
e = 0.5 * size * numpy.ones(size) + 0.5
validate_matmat(a.T, r.T, x, e)
validate_matmat(A.adjoint(), R.adjoint(), x, e)

print()


size = size // 10

a = numpy.ones((size, size))
A = MatrixOperator(a)

q, _ = qr(a)
Q = MatrixOperator(q)

diag = [i + 1 for i in range(size)]
d = numpy.diag(diag)
D = DiagonalMap(diag)

s = q.dot(d.dot(q.T))
s_copy = copy
S = SelfAdjointMap(s)

# ~~~~~~~~~~~~~~~~ SELF ADJOINT ~~~~~~~~~~~~~~~~
print('~ SelfAdjointMap object validation')
x = q[:, -1:]
e = d[:, -1:]
validate_matmat(q.T, s, x, e)
validate_matmat(Q.adjoint(), S, x, e)

x = q.T[:, -1:]
e = s[-1:, :].T
validate_matmat(s.T, q, x, e)
validate_matmat(S.adjoint(), Q, x, e)
print()
