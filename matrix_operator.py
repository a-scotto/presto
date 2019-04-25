#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on April 25, 2019 at 11:13.

@author: a.scotto

Description:
"""

import numpy
from linear_map import LinearMap


class MatrixOperatorError(Exception):
    """
    Exception raised when LinearMap object encounters specific errors.
    """
    
    
class MatrixOperator(LinearMap):

    def __init__(self, A):

        if not type(A) == numpy.ndarray:
            raise MatrixOperatorError('Matrix objects must be defined from a numpy.ndarray')

        super().__init__(A.shape, A.dtype, self._dot, self._dot_adj)

        self._A = A

    def _dot(self, X):
        return self._A @ X

    def _dot_adj(self, X):
        return self._A.T.conj() @ X

    def __repr__(self):
        return self._A.__repr__()


class UpperTriangularMap(LinearMap):

    def __init__(self, T):

        if not type(T) == numpy.ndarray:
            raise MatrixOperatorError('UpperTriangular objects must be defined from a '
                                      'numpy.ndarray')

        if T.shape[0] != T.shape[1]:
            raise MatrixOperatorError('Upper triangular operator must be of square shape.')

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
            raise MatrixOperatorError('SelfAdjointMap objects must be defined from a numpy.ndarray')

        if A.shape[0] != A.shape[1]:
            raise MatrixOperatorError('Self-adjoint operator must be of square shape.')

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
