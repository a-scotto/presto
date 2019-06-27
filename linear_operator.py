#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on April 17, 2019 at 10:06.

@author: a.scotto

Description:
"""

import numpy
import scipy.sparse

from scipy.sparse.linalg import LinearOperator as scipyLinearOperator


class LinearOperatorError(Exception):
    """
    Exception raised when LinearOperator object encounters specific errors.
    """


class MatrixOperatorError(Exception):
    """
    Exception raised when LinearMap object encounters specific errors.
    """


class LinearOperator(scipyLinearOperator):

    def __init__(self, shape, dtype, apply_cost=None):
        if not isinstance(shape, tuple) or len(shape) != 2:
            raise LinearOperatorError('Shape must be a tuple of the form (n, p).')

        if not isinstance(shape[0], int) or not isinstance(shape[1], int):
            raise LinearOperatorError('Shape must me a tuple of integers.')

        self.shape = shape

        try:
            self.dtype = numpy.dtype(dtype)
        except TypeError:
            raise LinearOperatorError('dtype provided not understood')
        
        self.apply_cost = apply_cost

    def adjoint(self):
        return _AdjointLinearOperator(self)

    def __rmul__(self, scalar):
        return _ScaledLinearOperator(self, scalar)

    def __add__(self, B):
        return _SummedLinearOperator(self, B)

    def __mul__(self, B):
        return _ComposedLinearOperator(self, B)

    def __neg__(self):
        return _ScaledLinearOperator(self, -1)

    def __sub__(self, B):
        return self + (-B)


class _ScaledLinearOperator(LinearOperator):

    def __init__(self, A, scalar):
        if not isinstance(A, LinearOperator):
            raise LinearOperatorError('External product should involve a LinearOperator.')

        if not numpy.isscalar(scalar):
            raise LinearOperatorError('External product should involve a scalar.')

        self.operands = (scalar, A)

        dtype = numpy.find_common_type([A.dtype], [type(scalar)])

        super().__init__(A.shape, dtype, A.apply_cost)

    def _matvec(self, X):
        return self.operands[0] * self.operands[1].dot(X)

    def _rmatvec(self, X):
        return numpy.conj(self.operands[0]) * self.operands[1].H.dot(X)


class _SummedLinearOperator(LinearOperator):

    def __init__(self, A, B):
        if not isinstance(A, LinearOperator) or not isinstance(B, LinearOperator):
            raise LinearOperatorError('Both operands in summation must be LinearOperator.')

        if A.shape != B.shape:
            raise LinearOperatorError('Both operands in summation must have the same shape.')

        self.operands = (A, B)

        dtype = numpy.find_common_type([A.dtype, B.dtype], [])
        apply_cost = A.apply_cost + B.apply_cost

        super().__init__(A.shape, dtype, apply_cost)

    def _matvec(self, X):
        return self.operands[0].dot(X) + self.operands[1].dot(X)

    def _rmatvec(self, X):
        return self.operands[0].H.dot(X) + self.operands[1].H.dot(X)


class _ComposedLinearOperator(LinearOperator):

    def __init__(self, A, B):
        if not isinstance(A, LinearOperator) or not isinstance(B, LinearOperator):
            raise LinearOperatorError('Both operands must be LinearOperator.')

        if A.shape[1] != B.shape[0]:
            raise LinearOperatorError('Shape of operands do not match for composition.')

        self.operands = (A, B)

        shape = (A.shape[0], B.shape[1])
        dtype = numpy.find_common_type([A.dtype, B.dtype], [])
        apply_cost = A.apply_cost + B.apply_cost

        super().__init__(shape, dtype, apply_cost)

    def _matvec(self, X):
        return self.operands[0].dot(self.operands[1].dot(X))

    def _rmatvec(self, X):
        return self.operands[1].H.dot(self.operands[0].H.dot(X))


class _AdjointLinearOperator(LinearOperator):

    def __init__(self, A):
        if not isinstance(A, LinearOperator):
            raise LinearOperatorError('Adjoint is only defined for LinearOperator.')

        n, p = A.shape

        super().__init__((p, n), A.dtype, A.apply_cost)

        self._matvec = A._rmatvec
        self._rmatvec = A._matvec


class IdentityOperator(LinearOperator):

    def __init__(self, size):
        if not isinstance(size, int) or size < 1:
            raise LinearOperatorError('IdentityOperator should have a positive integer size.')

        super().__init__((size, size), numpy.float64, 0)

    def _matvec(self, x):
        return x

    def _rmatvec(self, x):
        return x


class MatrixOperator(LinearOperator):

    def __init__(self, A):
        if isinstance(A, (numpy.ndarray, numpy.matrix)):
            self.sparse = False

        elif isinstance(A, scipy.sparse.spmatrix):
            self.sparse = True

        else:
            raise MatrixOperatorError('MatrixOperator should be defined from either numpy.ndarray, '
                                      'numpy.matrix, or scipy.sparse.spmatrix')

        self.A = A

        super().__init__(A.shape, A.dtype, A.size)

    def _matvec(self, X):
        X = numpy.asanyarray(X)
        n, p = self.shape

        if X.shape[0] != p:
            raise MatrixOperatorError('Dimensions do not match.')

        if X.shape[1] != 1:
            raise MatrixOperatorError('Array must have shape of the form (n, 1).')

        return self.A.dot(X)

    def _rmatvec(self, X):
        X = numpy.asanyarray(X)
        n, p = self.shape

        if X.shape[0] != p:
            raise MatrixOperatorError('Dimensions do not match.')

        if X.shape[1] != 1:
            raise MatrixOperatorError('Array must have shape of the form (n, 1).')

        return self.A.H.dot(X)

    def dot(self, x):
        if scipy.sparse.isspmatrix(x):
            return self.A.dot(x)
        else:
            return super().dot(x)

    def get_format(self):
        if self.sparse:
            return self.A.getformat()
        else:
            return 'dense'


class DiagonalMatrix(MatrixOperator):

    def __init__(self, diagonals):
        if isinstance(diagonals, list):
            diagonals = numpy.asarray(diagonals)

        A = scipy.sparse.diags(diagonals)

        super().__init__(A)


class TriangularMatrix(MatrixOperator):

    def __init__(self, A, lower=False):
        if not hasattr(A, 'shape') or len(A.shape) != 2 or A.shape[0] != A.shape[1]:
            raise MatrixOperatorError('TriangularMatrix matrix must be of square shape.')

        if lower:
            super().__init__(A)
        else:
            super().__init__(A)

        self.lower = lower


class SelfAdjointMatrix(MatrixOperator):

    def __init__(self, A, def_pos=False):
        if not hasattr(A, 'shape') or len(A.shape) != 2 or A.shape[0] != A.shape[1]:
            raise MatrixOperatorError('SelfAdjointMatrix matrix must be of square shape.')

        super().__init__(A)

        self.def_pos = def_pos
