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

__all__ = ['LinearOperator', 'IdentityOperator', 'MatrixOperator']


class LinearOperatorError(Exception):
    """
    Exception raised when LinearOperator object encounters specific errors.
    """


class MatrixOperatorError(Exception):
    """
    Exception raised when MatrixOperator object encounters specific errors.
    """


class LinearOperator(scipyLinearOperator):
    def __init__(self, shape: tuple, dtype: object):
        """
        Abstract representation of linear operators. Extends the existing LinearOperator class in Python library
        scipy.linalg.LinearOperator so as to handle sparse matrix representations.

        :param shape: Shape of the linear operator corresponding to the dimensions of domain and co-domain of the linear
            operator
        :param dtype: Type of the elements in the linear operator.
        """
        # Sanitize the shape attribute
        if not isinstance(shape, tuple) or len(shape) != 2:
            if not isinstance(shape[0], int) or not isinstance(shape[1], int):
                raise LinearOperatorError('Shape must be a tuple of integers of the form (n, p).')

        self.shape = shape

        # Sanitize the dtype attribute
        try:
            self.dtype = numpy.dtype(dtype)
        except TypeError:
            raise LinearOperatorError('dtype provided not understood.')

        self.self_adjoint = None
        self.positive_definite = None

        self.matvec_cost = self._matvec_cost()

    def _matvec(self, x: numpy.ndarray) -> numpy.ndarray:
        """
        Method to compute the action of the linear operator on vector x of shape (p, 1).

        :param x: Input vector to apply the linear operator on.
        """
        raise NotImplemented('Method _matvec_cost not implemented.')

    def _rmatvec(self, x: numpy.ndarray) -> numpy.ndarray:
        raise NotImplemented('Method _rmatvec_cost not implemented.')

    def _matvec_cost(self) -> float:
        """
        Method to compute the computational cost of an application of the linear operator.
        """
        raise NotImplemented('Method _matvec_cost not implemented.')

    def __rmul__(self, scalar):
        return _ScaledLinearOperator(self, scalar)

    def __add__(self, linear_op):
        return _SummedLinearOperator(self, linear_op)

    def __mul__(self, linear_op):
        return _ComposedLinearOperator(self, linear_op)

    def __neg__(self):
        return _ScaledLinearOperator(self, -1)

    def __sub__(self, linear_op):
        return self + (-linear_op)

    def _adjoint(self):
        return _AdjointLinearOperator(self)

    T = property(_adjoint)


class _ScaledLinearOperator(LinearOperator):
    def __init__(self, linear_op: LinearOperator, scalar: object):
        """
        Abstract representation of scaled linear operator, that is the scalar (or external) multiplication of linear
        operators.

        :param linear_op: Linear operator involved in the external multiplication.
        :param scalar: Scalar involved in the external multiplication.
        """
        # Sanitize the linear operator attribute
        if not isinstance(linear_op, LinearOperator):
            raise LinearOperatorError('External product should involve an instance of LinearOperator.')

        # Sanitize the scalar attribute
        if not numpy.isscalar(scalar):
            raise LinearOperatorError('External product should involve a scalar.')

        # Initialize operands attribute
        self.operands = (scalar, linear_op)

        dtype = numpy.find_common_type([linear_op.dtype], [type(scalar)])

        super().__init__(linear_op.shape, dtype)

    def _matvec(self, x):
        return self.operands[0] * self.operands[1].dot(x)

    def _rmatvec(self, x):
        return numpy.conj(self.operands[0]) * self.operands[1].H.dot(x)

    def _matvec_cost(self):
        n, _ = self.operands[1].shape
        return self.operands[1].matvec_cost + n


class _SummedLinearOperator(LinearOperator):
    def __init__(self, linear_op1: LinearOperator, linear_op2: LinearOperator):
        """
        Abstract representation of the sum of two linear operators.

        :param linear_op1: First linear operator involved in the summation.
        :param linear_op2: Second linear operator involved in the summation.
        """
        # Sanitize the linear operators attributes
        if not isinstance(linear_op1, LinearOperator) or not isinstance(linear_op2, LinearOperator):
            raise LinearOperatorError('Both operands in summation must be instances of LinearOperator.')

        # Check the operators shapes consistency
        if linear_op1.shape != linear_op2.shape:
            raise LinearOperatorError('Operands in summation have inconsistent shapes: {} and {}.'
                                      .format(linear_op1.shape, linear_op2.shape))

        # Initialize operands attribute
        self.operands = (linear_op1, linear_op2)

        dtype = numpy.find_common_type([linear_op1.dtype, linear_op2.dtype], [])

        super().__init__(linear_op1.shape, dtype)

    def _matvec(self, x):
        return self.operands[0].dot(x) + self.operands[1].dot(x)

    def _rmatvec(self, x):
        return self.operands[0].H.dot(x) + self.operands[1].H.dot(x)

    def _matvec_cost(self):
        n, _ = self.operands[0].shape
        return self.operands[0].matvec_cost + self.operands[1].matvec_cost + n


class _ComposedLinearOperator(LinearOperator):
    def __init__(self, linear_op1: LinearOperator, linear_op2: LinearOperator):
        """
        Abstract representation of the composition of two linear operators.

        :param linear_op1: First linear operator involved in the composition.
        :param linear_op2: Second linear operator involved in the composition.
        """
        # Sanitize the linear operators attributes
        if not isinstance(linear_op1, LinearOperator) or not isinstance(linear_op2, LinearOperator):
            raise LinearOperatorError('Both operands must be instances of LinearOperator.')

        # Check the operators compatibility
        if linear_op1.shape[1] != linear_op2.shape[0]:
            raise LinearOperatorError('Operands in composition have inconsistent shapes: {} and {}.'
                                      .format(linear_op1.shape, linear_op2.shape))

        # Initialize operands attribute
        self.operands = (linear_op1, linear_op2)

        # Resulting shape (n, p) @ (p, m) -> (n, m)
        shape = (linear_op1.shape[0], linear_op2.shape[1])
        dtype = numpy.find_common_type([linear_op1.dtype, linear_op2.dtype], [])

        super().__init__(shape, dtype)

    def _matvec(self, x):
        return self.operands[0].dot(self.operands[1].dot(x))

    def _rmatvec(self, x):
        return self.operands[1].H.dot(self.operands[0].H.dot(x))

    def _matvec_cost(self):
        return self.operands[0].matvec_cost + self.operands[1].matvec_cost


class _AdjointLinearOperator(LinearOperator):
    def __init__(self, linear_op: LinearOperator):
        """
        Abstract representation of the adjoint of a linear operator.

        :param linear_op: Linear operator to define the adjoint operator of.
        """
        # Sanitize the linear operator attribute
        if not isinstance(linear_op, LinearOperator):
            raise LinearOperatorError('Adjoint can only be defined for a LinearOperator instance.')

        n, p = linear_op.shape

        # Invert the _matvec and _rmatvec methods
        self._matvec = linear_op._rmatvec
        self._rmatvec = linear_op._matvec
        self._matvec_cost = linear_op._matvec_cost

        super().__init__((p, n), linear_op.dtype)


class IdentityOperator(LinearOperator):
    def __init__(self, n: int):
        """
        Abstract representation for identity linear operator.

        :param n: Dimension of the vector space identity operator.
        """
        # Sanitize the order attribute
        if not isinstance(n, int) or n < 1:
            raise LinearOperatorError('Impossible to define the identity operator of a vector space of dimension {}.'
                                      .format(n))

        super().__init__((n, n), numpy.float64)

    def _matvec(self, x):
        return x

    def _rmatvec(self, x):
        return x

    def _matvec_cost(self):
        return 0.


class MatrixOperator(LinearOperator):
    def __init__(self, matrix: object,
                 self_adjoint: bool = False,
                 positive_definite: bool = False):
        """
        Abstract class for matrix representations of linear operators.

        :param matrix: Matrix representation of a linear operator. Must be one of:
            * numpy.ndarray
            * scipy.sparse.spmatrix
        :param self_adjoint: Whether the linear operator is self-adjoint.
        :param positive_definite: Whether the self-adjoint linear operator is positive-definite.
        """
        # Sanitize the matrix attribute and check for potential sparse representation
        if not isinstance(matrix, (numpy.ndarray, scipy.sparse.spmatrix)):
            raise MatrixOperatorError('Matrix representation requires a matrix-like format but received {}'
                                      .format(type(matrix)))

        self.is_sparse = scipy.sparse.issparse(matrix)
        self.matrix = matrix

        super().__init__(matrix.shape, matrix.dtype)

        self.self_adjoint = self_adjoint
        self.positive_definite = positive_definite

    def _matvec(self, x):
        x = numpy.asanyarray(x)
        return self.matrix.dot(x)

    def _rmatvec(self, x):
        x = numpy.asanyarray(x)
        return self.matrix.H.dot(x)

    def dot(self, x) -> [numpy.ndarray, scipy.sparse.spmatrix]:
        """
        Override scipy.linalg.linearOperator dot method to handle cases of a sparse input.

        :param x: Input vector, maybe represented in sparse format, to compute the dot product with.
        """
        if scipy.sparse.isspmatrix(x):
            return self.matrix.dot(x)
        else:
            return super().dot(x)

    def _matvec_cost(self):
        return 2 * self.matrix.size
