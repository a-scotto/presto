#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on April 23, 2019 at 16:38.

@author: a.scotto

Description:
"""

import copy
import numpy

from time import time
from numpy.linalg import norm, qr
from linear_map import Matrix, DiagonalMap, UpperTriangularMap, SelfAdjointMap

size = 100

diag = [i + 1 for i in range(size)]

a = numpy.random.randint(0, size, size=(size, size))
A = Matrix(a)

d = numpy.diag(diag)
D = DiagonalMap(diag)

r = numpy.triu(numpy.ones_like(a)) / size
R = UpperTriangularMap(r)

b = numpy.ones((size, 1))

# ################### MATRIX #######################
# ##################################################
print('~ MATRIX-VECTOR product validation')
t = time();
p = a.dot(b)
print('{:1.2e} |'.format(time() - t), end=' ')
t = time()
P = A.dot(b)
print('{:1.2e} |'.format(time() - t), end=' ')
print('{:1.2e}'.format(norm(p - P)))
t = time()
p = a.T.dot(b)
print('{:1.2e} |'.format(time() - t), end=' ')
t = time()
P = A.dot_adj(b)
print('{:1.2e} |'.format(time() - t), end=' ')
print('{:1.2e}'.format(norm(p - P)))

print('~ MATRIX-MATRIX product validation')
t = time()
p = a.dot(a.dot(b))
print('{:1.2e} |'.format(time() - t), end=' ')
t = time()
P = (A * A).dot(b)
print('{:1.2e} |'.format(time() - t), end=' ')
print('{:1.2e}'.format(norm(p - P)))
t = time()
p = a.T.dot(a.T.dot(b))
print('{:1.2e} |'.format(time() - t), end=' ')
t = time()
P = (A * A).dot_adj(b)
print('{:1.2e} |'.format(time() - t), end=' ')
print('{:1.2e}'.format(norm(p - P)))

print()
# ################## DIAGONAL ######################
# ##################################################
print('~ DIAGONAL-VECTOR product validation')
t = time()
p = d.dot(b)
print('{:1.2e} |'.format(time() - t), end=' ')
t = time()
P = D.dot(b)
print('{:1.2e} |'.format(time() - t), end=' ')
print('{:1.2e}'.format(norm(p - P)))
t = time()
p = d.T.dot(b)
print('{:1.2e} |'.format(time() - t), end=' ')
t = time()
P = D.dot_adj(b)
print('{:1.2e} |'.format(time() - t), end=' ')
print('{:1.2e}'.format(norm(p - P)))

print('~ DIAGONAL-MATRIX product validation')
t = time()
p = d.dot(a.dot(b))
print('{:1.2e} |'.format(time() - t), end=' ')
t = time()
P = (D * A).dot(b)
print('{:1.2e} |'.format(time() - t), end=' ')
print('{:1.2e}'.format(norm(p - P)))
t = time()
p = a.T.dot(d.T.dot(b))
print('{:1.2e} |'.format(time() - t), end=' ')
t = time()
P = (D * A).dot_adj(b)
print('{:1.2e} |'.format(time() - t), end=' ')
print('{:1.2e}'.format(norm(p - P)))

print()
# ############### UPPER TRIANGLE ###################
# ##################################################
a = numpy.ones_like(r)
A = Matrix(a)
print('~ UPPER TRIANGLE-VECTOR product validation')
t = time()
p = r.dot(b)
print('{:1.2e} |'.format(time() - t), end=' ')
t = time()
P = R.dot(b)
print('{:1.2e} |'.format(time() - t), end=' ')
print('{:1.2e}'.format(norm(p - P)))
e = numpy.asarray([(size - i) / size for i in range(size)]).reshape(size, 1)
print('{:1.2e} | {:1.2e} | Exact solution distance'.format(norm(p - e), norm(P - e)))

t = time()
p = r.T.dot(b)
print('{:1.2e} |'.format(time() - t), end=' ')
t = time()
P = R.dot_adj(b)
print('{:1.2e} |'.format(time() - t), end=' ')
print('{:1.2e}'.format(norm(p - P)))
e = numpy.asarray([(i + 1) / size for i in range(size)]).reshape(size, 1)
print('{:1.2e} | {:1.2e} | Exact solution distance'.format(norm(p - e), norm(P - e)))

print('~ UPPER TRIANGLE-MATRIX product validation')
t = time()
p = r.dot(a.dot(b))
print('{:1.2e} |'.format(time() - t), end=' ')
t = time()
P = (R * A).dot(b)
print('{:1.2e} |'.format(time() - t), end=' ')
print('{:1.2e}'.format(norm(p - P)))
e = size * numpy.asarray([(size - i) / size for i in range(size)]).reshape(size, 1)
print('{:1.2e} | {:1.2e} | Exact solution distance'.format(norm(p - e), norm(P - e)))

t = time()
p = a.T.dot(r.T.dot(b))
print('{:1.2e} |'.format(time() - t), end=' ')
t = time()
P = (R * A).dot_adj(b)
print('{:1.2e} |'.format(time() - t), end=' ')
print('{:1.2e}'.format(norm(p - P)))
e = 0.5 * size * numpy.ones(size) + 0.5
print('{:1.2e} | {:1.2e} | Exact solution distance'.format(norm(p - e), norm(P - e)))

print()

# q, _ = qr(a)
# s = q.dot(d.dot(q.T))
# s_copy = copy
# S = SelfAdjointMap(s)
#
# ################# SDP MATRIX #####################
# ##################################################
# b = q[:, -1:]
# print('~ SELF ADJOINT-VECTOR product validation')
# t = time()
# p = s.dot(b)
# print('{:1.2e}'.format(time() - t), end=' ')
# t = time()
# P = S.dot(b)
# print('{:1.2e}'.format(time() - t), end=' ')
# print(norm(p - P))
# e = size * b
# print('Residual to exact solution {:1.2e}  |  {:1.2e}'
#       .format(norm(p - e), norm(P - e)))
# t = time()
# p = s.T.dot(b)
# print('{:1.2e}'.format(time() - t), end=' ')
# t = time()
# P = S.dot_adj(b)
# print('{:1.2e}'.format(time() - t), end=' ')
# print(norm(p - P))
# e = size * b
# print('Residual to exact solution {:1.2e}  |  {:1.2e}'
#       .format(norm(p - e), norm(P - e)))
# print()
#
# print('~ SELF ADJOINT-MATRIX product validation')
# Q = Matrix(q)
# b = q[:, -1:]
# t = time()
# p = q.T.dot(s.dot(b))
# print('{:1.2e}'.format(time() - t), end=' ')
# t = time()
# P = (Q.adjoint() * S).dot(b)
# print('{:1.2e}'.format(time() - t), end=' ')
# print(norm(p - P))
# e = d[:, -1:]
# print('Residual to exact solution {:1.2e}  |  {:1.2e}'
#       .format(norm(p - e), norm(P - e)))
#
# b = q.T[:, -1:]
# t = time()
# p = s.T.dot(q.dot(b))
# print('{:1.2e}'.format(time() - t), end=' ')
# t = time()
# P = (Q.adjoint() * S).dot_adj(b)
# print('{:1.2e}'.format(time() - t), end=' ')
# print(norm(p - P))
# e = s[-1:, :].T
# print('Residual to exact solution {:1.2e}  |  {:1.2e}'
#       .format(norm(p - e), norm(P - e)))
