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
    Exception raised when MatrixOperator object encounters specific errors.
    """


class LinearOperator(scipyLinearOperator):
    """
    Abstract class for linear operator.
    """

    def __init__(self, shape: tuple, dtype: object) -> None:
        # Sanitize the shape attribute
        if not isinstance(shape, tuple) or len(shape) != 2:
            raise LinearOperatorError('Shape must be a tuple of the form (n, p).')

        if not isinstance(shape[0], int) or not isinstance(shape[1], int):
            raise LinearOperatorError('Shape must me a tuple of integers.')

        self.shape = shape

        # Sanitize the dtype attribute
        try:
            self.dtype = numpy.dtype(dtype)
        except TypeError:
            raise LinearOperatorError('dtype provided not understood.')

        self.apply_cost = self._matvec_cost()

    def adjoint(self):
        return _AdjointLinearOperator(self)

    def _matvec_cost(self):
        raise LinearOperatorError('Method _matvec_cost not implemented.')

    def __rmul__(self, scalar):
        return _ScaledLinearOperator(self, scalar)

    def __add__(self, lin_op):
        return _SummedLinearOperator(self, lin_op)

    def __mul__(self, lin_op):
        return _ComposedLinearOperator(self, lin_op)

    def __neg__(self):
        return _ScaledLinearOperator(self, -1)

    def __sub__(self, lin_op):
        return self + (-lin_op)


class _ScaledLinearOperator(LinearOperator):
    """
    Abstract class for scaled linear operator.
    """

    def __init__(self, lin_op: LinearOperator, scalar: object) -> None:
        # Sanitize the lin_op attribute
        if not isinstance(lin_op, LinearOperator):
            raise LinearOperatorError('External product should involve a LinearOperator.')

        # Sanitize the scalar attribute
        if not numpy.isscalar(scalar):
            raise LinearOperatorError('External product should involve a scalar.')

        # Initialize operands attribute
        self.operands = (scalar, lin_op)

        dtype = numpy.find_common_type([lin_op.dtype], [type(scalar)])

        super().__init__(lin_op.shape, dtype)

    def _matvec(self, X):
        return self.operands[0] * self.operands[1].dot(X)

    def _rmatvec(self, X):
        return numpy.conj(self.operands[0]) * self.operands[1].H.dot(X)

    def _matvec_cost(self):
        m, _ = self.operands[1].shape
        return self.operands[1].apply_cost + 2 * m


class _SummedLinearOperator(LinearOperator):
    """
    Abstract class for summed linear operator.
    """

    def __init__(self, lin_op1: LinearOperator, lin_op2: LinearOperator) -> None:
        # Sanitize the lin_op1 and lin_op2 attributes
        if not isinstance(lin_op1, LinearOperator) or not isinstance(lin_op2, LinearOperator):
            raise LinearOperatorError('Both operands in summation must be LinearOperator.')

        # Check the operators compatibility
        if lin_op1.shape != lin_op2.shape:
            raise LinearOperatorError('Both operands in summation must have the same shape.')

        # Initialize operands attribute
        self.operands = (lin_op1, lin_op2)

        dtype = numpy.find_common_type([lin_op1.dtype, lin_op2.dtype], [])

        super().__init__(lin_op1.shape, dtype)

    def _matvec(self, X):
        return self.operands[0].dot(X) + self.operands[1].dot(X)

    def _rmatvec(self, X):
        return self.operands[0].H.dot(X) + self.operands[1].H.dot(X)

    def _matvec_cost(self):
        m, _ = self.operands[0].shape
        return self.operands[0].apply_cost + self.operands[1].apply_cost + m


class _ComposedLinearOperator(LinearOperator):
    """
    Abstract class for composed linear operator.
    """

    def __init__(self, lin_op1: LinearOperator, lin_op2: LinearOperator) -> None:
        # Sanitize the lin_op1 and lin_op2 attributes
        if not isinstance(lin_op1, LinearOperator) or not isinstance(lin_op2, LinearOperator):
            raise LinearOperatorError('Both operands must be LinearOperator.')

        # Check the operators compatibility
        if lin_op1.shape[1] != lin_op2.shape[0]:
            raise LinearOperatorError('Shape of operands do not match for composition.')

        # Initialize operands attribute
        self.operands = (lin_op1, lin_op2)

        shape = (lin_op1.shape[0], lin_op2.shape[1])
        dtype = numpy.find_common_type([lin_op1.dtype, lin_op2.dtype], [])

        super().__init__(shape, dtype)

    def _matvec(self, X):
        return self.operands[0].dot(self.operands[1].dot(X))

    def _rmatvec(self, X):
        return self.operands[1].H.dot(self.operands[0].H.dot(X))

    def _matvec_cost(self):
        return self.operands[0].apply_cost + self.operands[1].apply_cost


class _AdjointLinearOperator(LinearOperator):
    """
    Abstract class for adjoint linear operator.
    """

    def __init__(self, lin_op: LinearOperator) -> None:
        # Sanitize the lin_op attribute
        if not isinstance(lin_op, LinearOperator):
            raise LinearOperatorError('Adjoint is only defined for LinearOperator.')

        n, p = lin_op.shape

        # Invert the matvec and rmatvec methods
        self._matvec = lin_op._rmatvec
        self._rmatvec = lin_op._matvec
        self._matvec_cost = lin_op._matvec_cost

        super().__init__((p, n), lin_op.dtype)


class IdentityOperator(LinearOperator):
    """
    Generic class for identity linear operator.
    """

    def __init__(self, order: int) -> None:
        # Sanitize the order attribute
        if not isinstance(order, int) or order < 1:
            raise LinearOperatorError('IdentityOperator should have a positive integer order.')

        super().__init__((order, order), numpy.float64)

    def _matvec(self, x):
        return x

    def _rmatvec(self, x):
        return x

    def _matvec_cost(self):
        return 0.


class MatrixOperator(LinearOperator):
    """
    Abstract class for linear operator owing a matrix representation or behaviour.
    """

    def __init__(self, matrix) -> None:
        # Sanitize the matrix attribute
        if isinstance(matrix, (numpy.ndarray, numpy.matrix)):
            self.sparse = False

        elif isinstance(matrix, scipy.sparse.spmatrix):
            self.sparse = True

        else:
            raise MatrixOperatorError('MatrixOperator should be defined from either numpy.ndarray, '
                                      'numpy.matrix, or scipy.sparse.spmatrix')

        self.matrix = matrix

        super().__init__(matrix.shape, matrix.dtype)

    def _matvec(self, X):
        X = numpy.asanyarray(X)
        n, p = self.shape

        if X.shape[0] != p:
            raise MatrixOperatorError('Dimensions do not match.')

        if X.shape[1] != 1:
            raise MatrixOperatorError('Array must have shape of the form (n, 1).')

        return self.matrix.dot(X)

    def _rmatvec(self, X):
        X = numpy.asanyarray(X)
        n, p = self.shape

        if X.shape[0] != p:
            raise MatrixOperatorError('Dimensions do not match.')

        if X.shape[1] != 1:
            raise MatrixOperatorError('Array must have shape of the form (n, 1).')

        return self.matrix.H.dot(X)

    def _matvec_cost(self):
        return 2 * self.matrix.size

    def dot(self, x):
        # Precaution required to handle sparse matrix product.
        if scipy.sparse.isspmatrix(x):
            return self.matrix.dot(x)
        else:
            return super().dot(x)

    def get_format(self):
        if self.sparse:
            return self.matrix.getformat()
        else:
            return 'dense'


class DiagonalMatrix(MatrixOperator):
    """
    Abstract class for diagonal matrix operator.
    """

    def __init__(self, diagonals: list) -> None:
        # Sanitize the diagonals attribute
        if isinstance(diagonals, list):
            diagonals = numpy.asarray(diagonals)

        matrix = scipy.sparse.diags(diagonals)

        super().__init__(matrix)


class TriangularMatrix(MatrixOperator):
    """
    Abstract class for triangular matrix operator.
    """

    def __init__(self, matrix, lower: bool = False) -> None:
        # Sanitize the matrix attribute
        if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
            raise MatrixOperatorError('TriangularMatrix matrix must be of square shape.')

        super().__init__(matrix)

        self.lower = lower


class SelfAdjointMatrix(MatrixOperator):
    """
    Abstract class for definite positive self-adjoint matrix operator.
    """

    def __init__(self, matrix, def_pos: bool = False) -> None:
        # Sanitize the matrix attribute
        if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
            raise MatrixOperatorError('SelfAdjointMatrix matrix must be of square shape.')

        super().__init__(matrix)

        self.def_pos = def_pos


class NormalEquations(SelfAdjointMatrix):
    """
    Abstract class for normal equations operator. Namely take A in R^{m \times n} with m > n and
    compute the matrix-vector product A*Ax.
    """

    def __init__(self, matrix):
        # Build normal equations related to matrix
        m, n = matrix.shape

        if m >= n:
            matrix = matrix.H @ matrix
        else:
            matrix = matrix @ matrix.H

        super().__init__(matrix, def_pos=True)
