#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on April 17, 2019 at 10:06.

@author: a.scotto

Description:
"""

import numpy
import scipy

from utils import qr, convert_to_col


class LinearMapError(Exception):
    """
    Exception raised when LinearMap object encounters specific errors.
    """


class ProjectorError(Exception):
    """
    Exception raised when Projector object encounters specific errors.
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


class LinearMapInverse(LinearMap):

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


class Projector(LinearMap):
    """
    Abstract class for projectors. In the initialization of the Projector, X denotes the range
    and Y denotes the orthogonal complement of the kernel. The matrix formulation is:

    P = X * ( Y^T * X)^{-1} * Y^T

    """

    def __init__(self, X, Y=None, orthogonalize=True):

        if not isinstance(X, numpy.ndarray) or (Y is not None and not isinstance(Y, numpy.ndarray)):
            raise ProjectorError('Projector must be a numpy.ndarray')

        X = convert_to_col(X)
        Y = X if Y is None else convert_to_col(Y)

        if X.shape != Y.shape:
            raise ProjectorError('Projector range and kernel orthogonal must have same dimension.')

        shape = (X.shape[0], X.shape[0])
        dtype = numpy.find_common_type([X.dtype, Y.dtype], [])

        super().__init__(shape, dtype, self._dot, self._dot_adj)

        # If orthogonal projector
        if Y is X:
            if orthogonalize:
                self.V, self.R_v = scipy.linalg.qr(X)
            else:
                self.V, self.R_v = X, None

            self.W, self.R_w = self.V, self.R_v

            self.Q, self.R = None, None

        # If oblique projector
        else:
            if orthogonalize:
                self.V, self.R_v = scipy.linalg.qr(X)
                self.W, self.R_w = scipy.linalg.qr(Y)
            else:
                self.V, self.R_v = X, None
                self.W, self.R_w = Y, None

            self.W, self.R_w = self.V, self.R_v

            self.Q, self.R = scipy.linalg.qr(self.W.T.dot(self.V))

    def _dot(self, X):
        # If orthogonal projector
        if self.Q is None:
            return self.V.dot(self.W.T.dot(X))
        # If oblique projector
        else:
            return self.V.dot(scipy.linalg.solve_triangular(self.R, self.Q.T.dot(self.W.T.dot(X))))

    def _dot_adj(self, X):
        # If orthogonal projector
        if self.Q is None:
            return self.W.dot(self.V.T.dot(X))
        # If oblique projector
        else:
            return self.W.dot(self.Q(scipy.linalg.solve_triangular(self.R,
                                                                   self.V.T.dot(X),
                                                                   lower=True)))


if __name__ == "__main__":

    S = numpy.random.rand(10, 3)
    S, _ = scipy.linalg.qr(S)

    S_orth = numpy.random.rand(10, 3)
    S_orth -= S.dot(S.T.dot(S_orth))

    P = Projector(S)

    print(numpy.linalg.norm(P.dot(S) - S))
    print(numpy.linalg.norm(P.dot(S_orth)))

    P_s = P.adjoint()

    print(numpy.linalg.norm(P_s.dot(S_orth) - S_orth))
    print(numpy.linalg.norm(P_s.dot(S)))
