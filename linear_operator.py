#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on April 17, 2019 at 10:06.

@author: a.scotto

Description:
"""

import numpy
import scipy.sparse

from utils import qr, convert_to_col

from scipy.sparse.linalg import LinearOperator as scipyLinearOperator


class LinearOperatorError(Exception):
    """
    Exception raised when LinearOperator object encounters specific errors.
    """


class MatrixOperatorError(Exception):
    """
    Exception raised when LinearMap object encounters specific errors.
    """


class ProjectorError(Exception):
    """
    Exception raised when Projector object encounters specific errors.
    """


class LinearOperator(scipyLinearOperator):

    def __init__(self, shape, dtype):

        if not isinstance(shape, tuple) or len(shape) != 2:
            raise LinearOperatorError('Shape must be a tuple of the form(n, p).')

        if not isinstance(shape[0], int) or not isinstance(shape[1], int):
            raise LinearOperatorError('Shape must me a tuple of integers.')

        self.shape = shape
        self.size = None

        try:
            self.dtype = numpy.dtype(dtype)
        except TypeError:
            raise LinearOperatorError('dtype provided not understood')

    # def _matmat(self, X):
    #     raise LinearOperatorError('Matrix-matrix product not defined for LinearOperator objects.')

    def _rmatmat(self, X):
        raise LinearOperatorError('rMatrix-matrix product not defined for LinearOperator objects.')

    def _matvec(self, x):
        return None

    def _rmatvec(self, x):
        return None

    def adjoint(self):
        return _AdjointLinearOperator(self)

    def __rmul__(self, scalar):
        return _ScaleLinearOperator(self, scalar)

    def __add__(self, B):
        return _SumLinearOperator(self, B)

    def __mul__(self, B):
        return _ComposedLinearOperator(self, B)

    def __neg__(self):
        return _ScaleLinearOperator(self, -1)

    def __sub__(self, B):
        return self + (-B)


class _ScaleLinearOperator(LinearOperator):

    def __init__(self, A, scalar):

        if not isinstance(A, LinearOperator):
            raise LinearOperatorError('External product should involve a LinearOperator.')

        if not numpy.isscalar(scalar):
            raise LinearOperatorError('External product should involve a scalar.')

        self.operands = (scalar, A)

        dtype = numpy.find_common_type([A.dtype], [type(scalar)])

        super().__init__(A.shape, dtype)

    def _matvec(self, X):
        return self.operands[0] * self.operands[1].dot(X)

    def _rmatvec(self, X):
        return numpy.conj(self.operands[0]) * self.operands[1].H.dot(X)


class _SumLinearOperator(LinearOperator):

    def __init__(self, A, B):

        if not isinstance(A, LinearOperator) or not isinstance(B, LinearOperator):
            raise LinearOperatorError('Both operands in summation must be LinearOperator.')

        if A.shape != B.shape:
            raise LinearOperatorError('Both operands in summation must have the same shape.')

        self.operands = (A, B)

        dtype = numpy.find_common_type([A.dtype, B.dtype], [])

        super().__init__(A.shape, dtype)

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

        super().__init__(shape, dtype)

    def _matvec(self, X):
        return self.operands[0].dot(self.operands[1].dot(X))

    def _rmatvec(self, X):
        return self.operands[1].H.dot(self.operands[0].H.dot(X))


class _AdjointLinearOperator(LinearOperator):

    def __init__(self, A):

        if not isinstance(A, LinearOperator):
            raise LinearOperatorError('Adjoint is only defined for LinearOperator.')

        n, p = A.shape

        super().__init__((p, n), A.dtype)

        self._matvec = A._rmatvec
        self._rmatvec = A._matvec


class IdentityOperator(LinearOperator):

    def __init__(self, size):

        if not isinstance(size, int) or size < 1:
            raise LinearOperatorError('IdentityOperator should have a positive integer size.')

        super().__init__((size, size), dtype=numpy.float64)

    def _matvec(self, x):
        return x

    def _rmatvec(self, x):
        return x


class MatrixOperator(LinearOperator):

    _sparse_formats = ['csr', 'csc', 'dia', 'bsr', 'coo', 'dok', 'lil']

    def __init__(self, A, sparse_format='dense'):

        if isinstance(A, numpy.ndarray) or isinstance(A, numpy.matrix):
            if sparse_format is 'dense':
                self.A = scipy.sparse.linalg.aslinearoperator(A)
            elif sparse_format in self._sparse_formats:
                self.A = scipy.sparse.csc_matrix(A).asformat(sparse_format)
            else:
                raise MatrixOperatorError('Sparse format {} not understood.'.format(sparse_format))

        elif isinstance(A, scipy.sparse.spmatrix):
            self.A = A

        else:
            raise MatrixOperatorError('MatrixOperator of dense format should be defined from either'
                                      'a numpy.ndarray, numpy.matrix, or scipy.sparse.spmatrix')

        self.sparse_format = sparse_format
        super().__init__(A.shape, A.dtype)

        self.size = self.A.size

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


class DiagonalMatrix(MatrixOperator):

    def __init__(self, diagonals, sparse_format='dia'):

        if isinstance(diagonals, list):
            diagonals = numpy.asarray(diagonals)

        super().__init__(numpy.diag(diagonals), sparse_format)


class TriangularMatrix(MatrixOperator):

    def __init__(self, A, lower=False, sparse_format='csc'):

        if hasattr(A, 'shape') and A.shape[0] != A.shape[1]:
            raise MatrixOperatorError('TriangularMatrix matrix must be of square shape.')

        if lower:
            super().__init__(numpy.tril(A), sparse_format)
        else:
            super().__init__(numpy.triu(A), sparse_format)


class SelfAdjointMatrix(MatrixOperator):

    def __init__(self, A, def_pos=False, sparse_format='csc'):

        if hasattr(A, 'shape') and A.shape[0] != A.shape[1]:
            raise MatrixOperatorError('SelfAdjointMatrix matrix must be of square shape.')

        super().__init__(A, sparse_format)

        self.def_pos = def_pos


class Projector(LinearOperator):
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

        super().__init__(shape, dtype)

        # If orthogonal projector
        if Y is X:
            if orthogonalize:
                self.V, self.R_v = qr(X)
                self.Q, self.R = None, None
            else:
                self.V, self.R_v = X, None
                self.Q, self.R = scipy.linalg.qr(self.V.T.dot(self.V))

            self.W, self.R_w = self.V, self.R_v

        # If oblique projector
        else:
            if orthogonalize:
                self.V, self.R_v = qr(X)
                self.W, self.R_w = qr(Y)
            else:
                self.V, self.R_v = X, None
                self.W, self.R_w = Y, None

            self.Q, self.R = scipy.linalg.qr(self.W.T.dot(self.V))

    def _matvec(self, X):
        # If orthonormalized
        if self.Q is None:
            return self.V.dot(self.W.T.dot(X))
        else:
            c = self.Q.T.dot(self.W.T.dot(X))
            return self.V.dot(scipy.linalg.solve_triangular(self.R, c))

    def _rmatvec(self, X):
        # If orthonormalized
        if self.Q is None:
            return self.W.dot(self.V.T.dot(X))
        else:
            c = self.V.T.dot(X)
            return self.W.dot(self.Q.dot(scipy.linalg.solve(self.R.T, c)))

    def get_complement(self):
        return IdentityOperator(self.shape[0]) - self
