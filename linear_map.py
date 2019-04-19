#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on April 17, 2019 at 10:06.

@author: a.scotto

Description:
"""

import numpy


class LinearMapError(Exception):
    """
    Exception raised when LinearMap object encounters specific errors.
    """


class LinearMap(object):

    def __init__(self, shape, dtype=numpy.float64, dot=None, dot_adj=None):

        if not isinstance(shape, tuple) or len(shape) != 2:
            raise LinearMapError('Shape must be a tuple (n, p).')

        if not isinstance(shape[0], int) or not isinstance(shape[1], int):
            raise LinearMapError('Shape must me a tuple of integers.')

        self.shape = shape

        try:
            self.dtype = numpy.dtype(dtype)
        except TypeError:
            raise LinearMapError('"dtype" provided not understood')

        self._dot = dot
        self._dot_adj = dot_adj

    def dot(self, X):
        X = numpy.asanyarray(X)
        n, p = self.shape

        if X.shape[0] != p:
            raise LinearMapError('Dimensions do not match.')

        if X.shape[1] == 0:
            raise LinearMapError('Array must have at least 2 dimensions.')

        if self._dot is None:
            raise LinearMapError('The "_dot" class method is not implemented. ')

        return self._dot(X)

    def dot_adj(self, X):
        X = numpy.asanyarray(X)
        n, p = self.shape

        if X.shape[0] != n:
            raise LinearMapError('Dimensions do not match.')

        if X.shape[1] == 0:
            raise LinearMapError('Array must have at least 2 dimensions.')

        if self._dot is None:
            raise LinearMapError('The "_dot_adj" class method is not implemented. ')

        return self._dot_adj(X)

    def canonical_repr(self):

        n, p = self.shape

        A = numpy.zeros(self.shape)

        e = numpy.zeros((p, 1))

        for i in range(p):
            if i != 0:
                e[i - 1, 0] = 0
            e[i, 0] = 1
            A[:, i] = self.dot(e)[:, 0]

        return A

    def adjoint(self):
        return _AdjointLinearMap(self)

    def __rmul__(self, scalar):
        return _ScaleLinearMap(self, scalar)

    def __add__(self, B):
        return _SumLinearMap(self, B)

    def __mul__(self, B):
        return _ComposeLinearMap(self, B)

    def __neg__(self):
        return _ScaleLinearMap(self, -1)

    def __sub__(self, B):
        return self + (-B)


class _ScaleLinearMap(LinearMap):

    def __init__(self, A, scalar):

        if not isinstance(A, LinearMap):
            raise LinearMapError('External linear map product must concern LinearMap.')

        if not numpy.isscalar(scalar):
            raise LinearMapError('External linear map product must be with a scalar.')

        self.operands = (scalar, A)

        dtype = numpy.find_common_type([A.dtype], [type(scalar)])

        super().__init__(A.shape, dtype, self._dot, self._dot_adj)

    def _dot(self, X):
        return self.operands[0] * self.operands[1].dot(X)

    def _dot_adj(self, X):
        return numpy.conj(self.operands[0]) * self.operands[1].dot_adj(X)


class _SumLinearMap(LinearMap):

    def __init__(self, A, B):

        if not isinstance(A, LinearMap) or not isinstance(B, LinearMap):
            raise LinearMapError('Both operands must be LinearMap.')

        if A.shape != B.shape:
            raise LinearMapError('Both operands must have the same shape.')

        self.operands = (A, B)

        dtype = numpy.find_common_type([A.dtype, B.dtype], [])

        super().__init__(A.shape, dtype, self._dot, self._dot_adj)

    def _dot(self, X):
        return self.operands[0].dot(X) + self.operands[1].dot(X)

    def _dot_adj(self, X):
        return self.operands[0].dot_adj(X) + self.operands[1].dot_adj(X)


class _ComposeLinearMap(LinearMap):

    def __init__(self, A, B):

        if not isinstance(A, LinearMap) or not isinstance(B, LinearMap):
            raise LinearMapError('Both operands must be LinearMap.')

        if A.shape[1] != B.shape[0]:
            raise LinearMapError('Shape of operands do not match for composition.')

        self.operands = (A, B)

        shape = (A.shape[0], B.shape[1])
        dtype = numpy.find_common_type([A.dtype, B.dtype], [])

        super().__init__(shape, dtype, self._dot, self._dot_adj)

    def _dot(self, X):
        return self.operands[0].dot(self.operands[1].dot(X))

    def _dot_adj(self, X):
        return self.operands[1].dot_adj(self.operands[0].dot_adj(X))


class _AdjointLinearMap(LinearMap):

    def __init__(self, A):

        if not isinstance(A, LinearMap):
            raise LinearMapError('Adjoint is only defined for LinearMap.')

        n, p = A.shape

        super().__init__((p, n), A.dtype, A._dot_adj, A._dot)


class SolvedLinearMap(LinearMap):

    def __init__(self, A):

        if not isinstance(A, LinearMap):
            raise LinearMapError('Requires a Matrix object for solving.')

        if A.shape[0] != A.shape[1]:
            raise LinearMapError('Impossible to solve non square Matrix.')

        super().__init__(A.shape, A.dtype, self._dot, self._dot_adj)

        self._A = A.canonical_repr()

    def _dot(self, X):
        return numpy.linalg.solve(self._A, X)

    def _dot_adj(self, X):
        return numpy.linalg.solve(self._A.T.conj(), X)


class Matrix(LinearMap):

    def __init__(self, A):

        if not type(A) == numpy.ndarray:
            raise LinearMapError('Matrix objects must be defined from a numpy.ndarray')

        super().__init__(A.shape, A.dtype, self._dot, self._dot_adj)

        self._A = A

    def _dot(self, X):
        return self._A @ X

    def _dot_adj(self, X):
        return self._A.T.conj() @ X

    def __repr__(self):
        return self._A.__repr__()


class IdentityMap(LinearMap):

    def __init__(self, shape):

        super().__init__(shape, dot=self._dot, dot_adj=self._dot_adj)

    @staticmethod
    def _dot(X):
        return X

    @staticmethod
    def _dot_adj(X):
        return X

    def canonical_repr(self):
        return numpy.eye(self.shape[0])


class ZeroMap(LinearMap):

    def __init__(self, shape):
        super().__init__(shape, dot=self._dot, dot_adj=self._dot_adj)

    @staticmethod
    def _dot(X):
        return numpy.zeros(X.shape)

    @staticmethod
    def _dot_adj(X):
        return numpy.zeros(X.shape)

    def canonical_repr(self):
        return numpy.zeros(self.shape)


class DiagonalMap(LinearMap):

    def __init__(self, diag):

        if isinstance(diag, list):
            diag = numpy.asarray(diag)

        if not isinstance(diag, numpy.ndarray) or not diag.ndim == 1:
            raise LinearMapError('The diagonal terms must be provided as a numpy.ndarray of 1 '
                                 'dimension.')

        super().__init__((diag.size, diag.size), diag.dtype, self._dot, self._dot_adj)

        self._diag = diag

    def _dot(self, X):
        ret = numpy.zeros_like(X)
        for i in range(self._diag.size):
            ret[i, :] = self._diag[i] * X[i, :]

        return ret

    def _dot_adj(self, X):
        ret = numpy.zeros_like(X)
        for i in range(self._diag.size):
            ret[i, :] = self._diag[i] * X[i, :]

        return ret

    def canonical_repr(self):
        return numpy.diag(self.diag)


class UpperTriangularMap(LinearMap):

    def __init__(self, T):

        if not type(T) == numpy.ndarray:
            raise LinearMapError('UpperTriangular objects must be defined from a numpy.ndarray')

        if T.shape[0] != T.shape[1]:
            raise LinearMapError('Upper triangular operator must be of square shape.')

        super().__init__(T.shape, T.dtype, self._dot, self._dot_adj)

        self._T = numpy.triu(T)

    def _dot(self, X):
        n, _ = self.shape
        ret = numpy.zeros_like(X)
        for i in range(n):
            ret[i, :] = self._T[[i], i:] @ X[i:, :]
        return ret

    def _dot_adj(self, X):
        n, _ = self.shape
        ret = numpy.zeros_like(X)
        for i in range(n):
            ret[i, :] = self._T[:(i + 1), i] @ X[:(i + 1), :]
        return ret

    def __repr__(self):
        return self._A.__repr__()


class SelfAdjointMap(LinearMap):

    def __init__(self, A):

        if not type(A) == numpy.ndarray:
            raise LinearMapError('SelfAdjointMap objects must be defined from a numpy.ndarray')

        if A.shape[0] != A.shape[1]:
            raise LinearMapError('Self-adjoint operator must be of square shape.')

        T = numpy.triu(A)
        T[numpy.diag_indices_from(T)] /= 2

        super().__init__(T.shape, T.dtype, self._dot, self._dot_adj)

        self._half = UpperTriangularMap(T)

    def _dot(self, X):
        return self._half.dot(X) + self._half.dot_adj(X)

    def _dot_adj(self, X):
        return self._half.dot(X) + self._half.dot_adj(X)

    def __repr__(self):
        return self._A.__repr__()


if __name__ == "__main__":

    import copy
    from time import time
    from numpy.linalg import norm, qr

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
    t = time()
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
