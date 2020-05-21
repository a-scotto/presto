#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on April 17, 2019 at 10:06.

@author: a.scotto

Description:
"""

import numpy
import warnings
import scipy.sparse

from typing import Union
from utils.linalg import qr
from scipy.sparse.linalg import LinearOperator as scipyLinearOperator

__all__ = ['LinearOperator', 'IdentityOperator', 'LinearSubspace', 'Subspace', 'MatrixOperator', 'OrthogonalProjector']

MatrixType = Union[numpy.ndarray, scipy.sparse.spmatrix]


class LinearOperatorError(Exception):
    """
    Exception raised when LinearOperator object encounters specific errors.
    """


class SubspaceError(Exception):
    """
    Exception raised when KrylovSubspace object encounters specific errors.
    """


class LinearOperator(scipyLinearOperator):
    def __init__(self, dtype: object, shape: tuple):
        """
        Abstract representation of linear operators. Extends the existing LinearOperator class in Python library
        scipy.linalg.LinearOperator so as to handle sparse matrix representations, and some additional attributes.

        :param dtype: Type of the elements in the linear operator.
        :param shape: Shape of the linear operator, i.e. the tuple (n, p) such that the linear operator maps R^n to R^p.
        """
        # Sanitize the data type attribute
        try:
            self.dtype = numpy.dtype(dtype)
        except TypeError:
            raise LinearOperatorError('dtype provided not understood.')

        # Sanitize the shape attribute
        if not isinstance(shape, tuple) or len(shape) != 2:
            if not isinstance(shape[0], int) or not isinstance(shape[1], int):
                raise LinearOperatorError('Shape must be a tuple of integers of the form (n, p).')

        self.shape = shape

        self.matvec_cost = self._matvec_cost()

    def _matvec(self, x: numpy.ndarray) -> numpy.ndarray:
        """
        Method to compute the action of the linear operator on vector x of shape (n, 1).

        :param x: Input vector to apply the linear operator on.
        """
        raise NotImplementedError('Method _matvec_cost not implemented for class objects {}.'.format(self.__class__))

    def _rmatvec(self, x: numpy.ndarray) -> numpy.ndarray:
        """
        Method to compute the action of the adjoint linear operator on vector x of shape (p, 1).

        :param x: Input vector to apply the adjoint linear operator on.
        """
        raise NotImplementedError('Method _rmatvec_cost not implemented for class objects {}.'.format(self.__class__))

    def _matvec_cost(self) -> float:
        """
        Method to compute the computational cost of an application of the linear operator.
        """
        raise NotImplementedError('Method _matvec_cost not implemented for class objects {}.'.format(self.__class__))

    def _mat(self):
        """
        Method to get a matrix representation of the linear operator, if existing.
        """
        raise NotImplementedError('Method _mat not implemented for class objects {}.'.format(self.__class__))

    def __rmul__(self, scalar):
        return _ScaledLinearOperator(self, scalar)

    def __add__(self, linear_op):
        return _SummedLinearOperator(self, linear_op)

    def __mul__(self, other):
        if isinstance(other, LinearSubspace):
            return _ImageSubspace(self, other)
        else:
            return _ComposedLinearOperator(self, other)

    def __neg__(self):
        return _ScaledLinearOperator(self, -1)

    def __sub__(self, linear_op):
        return self + (-linear_op)

    def _adjoint(self):
        return _AdjointLinearOperator(self)

    @property
    def T(self):
        return self._adjoint()

    @property
    def mat(self):
        return self._mat()


class _ScaledLinearOperator(LinearOperator):
    def __init__(self, linear_op: LinearOperator, scalar: numpy.ScalarType):
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

        super().__init__(dtype, linear_op.shape)

    def _matvec(self, x):
        return self.operands[0] * self.operands[1].dot(x)

    def _rmatvec(self, x):
        return numpy.conj(self.operands[0]) * self.operands[1].T.dot(x)

    def _matvec_cost(self):
        n, _ = self.operands[1].shape
        return self.operands[1].matvec_cost + n

    def _mat(self):
        return self.operands[0] * self.operands[1].mat


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

        # Check the consistency of linear operators shapes
        if linear_op1.shape != linear_op2.shape:
            raise LinearOperatorError('Operands in summation have inconsistent shapes: {} and {}.'
                                      .format(linear_op1.shape, linear_op2.shape))

        # Initialize operands attribute
        self.operands = (linear_op1, linear_op2)

        dtype = numpy.find_common_type([linear_op1.dtype, linear_op2.dtype], [])

        super().__init__(dtype, linear_op1.shape)

    def _matvec(self, x):
        return self.operands[0].dot(x) + self.operands[1].dot(x)

    def _rmatvec(self, x):
        return self.operands[0].T.dot(x) + self.operands[1].T.dot(x)

    def _matvec_cost(self):
        n, _ = self.operands[0].shape
        return self.operands[0].matvec_cost + self.operands[1].matvec_cost + n

    def _mat(self):
        return self.operands[0].mat + self.operands[1].mat


class _ComposedLinearOperator(LinearOperator):
    def __init__(self, linear_op1: LinearOperator, linear_op2: LinearOperator):
        """
        Abstract representation of the composition of two linear operators.

        :param linear_op1: First linear operator involved in the composition.
        :param linear_op2: Second linear operator involved in the composition.
        """
        # Sanitize the linear operators attributes
        if not isinstance(linear_op1, LinearOperator) or not isinstance(linear_op2, LinearOperator):
            raise LinearOperatorError('Both operands in composition must be instances of LinearOperator.')

        # Check the consistency of linear operators shapes
        if linear_op1.shape[1] != linear_op2.shape[0]:
            raise LinearOperatorError('Operands in composition have inconsistent shapes: {} and {}.'
                                      .format(linear_op1.shape, linear_op2.shape))

        # Initialize operands attribute
        self.operands = (linear_op1, linear_op2)

        # Resulting shape (n, p) @ (p, m) -> (n, m)
        shape = (linear_op1.shape[0], linear_op2.shape[1])
        dtype = numpy.find_common_type([linear_op1.dtype, linear_op2.dtype], [])

        super().__init__(dtype, shape)

    def _matvec(self, x):
        return self.operands[0].dot(self.operands[1].dot(x))

    def _rmatvec(self, x):
        return self.operands[1].T.dot(self.operands[0].T.dot(x))

    def _matvec_cost(self):
        return self.operands[0].matvec_cost + self.operands[1].matvec_cost

    def _mat(self):
        return self.operands[0].mat @ self.operands[1].mat


class _AdjointLinearOperator(LinearOperator):
    def __init__(self, linear_op: LinearOperator):
        """
        Abstract representation of the adjoint of a linear operator.

        :param linear_op: Linear operator to define the adjoint operator of.
        """
        # Sanitize the linear operator attribute
        if not isinstance(linear_op, LinearOperator):
            raise LinearOperatorError('Adjoint can only be defined for a LinearOperator instance.')

        self._linear_op = linear_op

        n, p = linear_op.shape
        super().__init__(linear_op.dtype, (p, n))

    def _matvec(self, x):
        return self._linear_op._rmatvec(x)

    def _rmatvec(self, x):
        return self._linear_op._matvec(x)

    def _matvec_cost(self):
        return self._linear_op._matvec_cost()

    def _mat(self):
        return self._linear_op.mat.T.conj()


class LinearSubspace(LinearOperator):
    def __init__(self, dtype: object, shape: tuple):
        """
        Abstract representation of linear subspaces. Extends the existing LinearOperator class defined above while
        defining specific operations for linear subspaces.

        :param dtype: Type of the elements in the linear subspace.
        :param shape: Shape of the linear subspace of the form (n, p), n being the dimension of the vector space, and d
        the number of vectors in the generating set. Dimension of the subspace is therefore <= d.
        """
        # Instantiation of base class
        super().__init__(dtype, shape)

        self.matvec_cost = self._matvec_cost()

    def _matvec_cost(self) -> float:
        """
        Method to compute the computational cost of a product by subspace, i.e. the making up of a linear combination.
        """
        raise NotImplementedError('Method _matvec_cost not implemented for class objects {}.'.format(self.__class__))

    def _mat(self):
        """
        Method to get a matrix representation of the linear subspace, if existing.
        """
        raise NotImplementedError('Method _mat not implemented for class objects {}.'.format(self.__class__))

    def __add__(self, subspace):
        return _SummedSubspace(self, subspace)

    def __mul__(self, other):
        if (isinstance(other, LinearOperator) and not isinstance(other, LinearSubspace))\
                and isinstance(self, _AdjointSubspace):
            return _ImageSubspace(other, self.T).T
        else:
            return _ComposedLinearOperator(self, other)

    def _adjoint(self):
        return _AdjointSubspace(self)


class _SummedSubspace(LinearSubspace):
    def __init__(self, subspace1: LinearSubspace, subspace2: LinearSubspace):
        """
        Abstract representation of the addition of two linear subspaces, that is, of the sum F + G of linear subspace.

        :param subspace1: First subspace involved in the summation.
        :param subspace2: Second subspace involved in the summation.
        """
        # Sanitize the subspaces attributes
        if not isinstance(subspace1, LinearSubspace) or not isinstance(subspace2, LinearSubspace):
            raise SubspaceError('Both operands in summation must be instances of LinearSubspace.')

        # Check the consistency of subspaces shapes
        if subspace1.shape[0] != subspace2.shape[0]:
            raise SubspaceError('Operands in summation have inconsistent shapes: {} and {}.'
                                .format(subspace1.shape, subspace2.shape))

        if subspace1.shape[1] + subspace2.shape[1] > subspace1.shape[0]:
            warnings.warn('Dimensions of the 2 subspaces exceed vector space dimension, hence no longer full rank.')

        # Initialize operands attribute
        self.operands = (subspace1, subspace2)

        # Resulting shape and data type
        shape = (subspace1.shape[0], subspace1.shape[1] + subspace2.shape[1])
        dtype = numpy.find_common_type([subspace1.dtype, subspace2.dtype], [])

        super().__init__(dtype, shape)

    def _matvec(self, x: numpy.ndarray) -> numpy.ndarray:
        _, k = self.operands[0].shape
        y = self.operands[0].dot(x[:k])
        z = self.operands[1].dot(x[k:])
        return y + z

    def _rmatvec(self, x: numpy.ndarray) -> numpy.ndarray:
        return numpy.vstack([self.operands[0].T.dot(x), self.operands[1].T.dot(x)])

    def _matvec_cost(self) -> float:
        n, _ = self.operands[0].shape
        return self.operands[0].matvec_cost + self.operands[1].matvec_cost + n

    def _mat(self):
        mat1, mat2 = self.operands[0].mat, self.operands[1].mat
        if scipy.sparse.isspmatrix(mat1) and scipy.sparse.isspmatrix(mat1):
            return scipy.sparse.hstack([self.operands[0].mat, self.operands[1].mat])
        elif not scipy.sparse.isspmatrix(mat1) and scipy.sparse.isspmatrix(mat1):
            return numpy.hstack([self.operands[0].mat.todense(), self.operands[1].mat])
        elif scipy.sparse.isspmatrix(mat1) and not scipy.sparse.isspmatrix(mat1):
            return numpy.hstack([self.operands[0].mat, self.operands[1].mat.todense()])
        else:
            return numpy.hstack([self.operands[0].mat, self.operands[1].mat])


class _ImageSubspace(LinearSubspace):
    def __init__(self, linear_op: LinearOperator, subspace: LinearSubspace):
        """
        Abstract representation of the image of a subspace under the action of a linear operator.

        :param linear_op: Linear operator to be acting on the given subspace.
        :param subspace: Subspace which image is considered.
        """
        # Sanitize the linear operator and subspace attributes
        if not isinstance(linear_op, LinearOperator) or not isinstance(subspace, LinearSubspace):
            raise SubspaceError('Image of a subspace requires instances of LinearOperator and LinearSubspace.')

        # Check the consistency of operands shapes
        if linear_op.shape[1] != subspace.shape[0]:
            raise SubspaceError('Operands for the image have inconsistent shapes: {} and {}.'
                                .format(linear_op.shape, subspace.shape))

        # Initialize operands attribute
        self.operands = (linear_op, subspace)

        # Resulting shape and data type
        shape = subspace.shape
        dtype = numpy.find_common_type([linear_op.dtype, subspace.dtype], [])

        super().__init__(dtype, shape)

    def _matvec(self, x):
        return self.operands[0].dot(self.operands[1].dot(x))

    def _rmatvec(self, x):
        return self.operands[1].T.dot(self.operands[0].T.dot(x))

    def _matvec_cost(self):
        return self.operands[0].matvec_cost + self.operands[1].matvec_cost

    def _mat(self):
        return self.operands[0].mat @ self.operands[1].mat


class _AdjointSubspace(LinearSubspace):
    def __init__(self, subspace: LinearSubspace):
        """
        Abstract representation of the adjoint of a linear subspace.

        :param subspace: Linear subspace to define the adjoint operator of.
        """
        # Sanitize the linear subspace attribute
        if not isinstance(subspace, LinearSubspace):
            raise LinearOperatorError('Adjoint must be defined from a LinearSubspace instance.')

        self._subspace = subspace

        n, p = subspace.shape
        super().__init__(subspace.dtype, (p, n))

    def _matvec(self, x):
        return self._subspace._rmatvec(x)

    def _rmatvec(self, x):
        return self._subspace._matvec(x)

    def _matvec_cost(self):
        return self._subspace._matvec_cost()

    def _mat(self):
        return self._subspace.mat.T.conj()


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

        super().__init__(numpy.float64, (n, n))

    def _matvec(self, x):
        return x

    def _rmatvec(self, x):
        return x

    def _matvec_cost(self):
        return 0.

    def _mat(self):
        return scipy.sparse.eye(self.shape[0])


class MatrixOperator(LinearOperator):
    def __init__(self, matrix: MatrixType):
        """
        Abstract class for matrix representation of a linear operator.

        :param matrix: Matrix representation of a linear operator. Must be one of:
            * numpy.ndarray
            * scipy.sparse.spmatrix
        """
        # Sanitize the matrix attribute and check for potential sparse representation
        if not isinstance(matrix, (numpy.ndarray, scipy.sparse.spmatrix)):
            raise LinearOperatorError('Matrix representation requires a matrix-like format but received {}'
                                      .format(type(matrix)))

        self.is_sparse = scipy.sparse.issparse(matrix)
        self.matrix = matrix

        super().__init__(matrix.dtype, matrix.shape)

    def _matvec(self, x):
        x = numpy.asanyarray(x)
        return self.matrix.dot(x)

    def _rmatvec(self, x):
        x = numpy.asanyarray(x)
        return self.matrix.H.dot(x)

    def dot(self, x) -> MatrixType:
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

    def _mat(self):
        return self.matrix


class Subspace(LinearSubspace):
    def __init__(self, subspace: MatrixType):
        """
        Abstract class for the matrix representation of a linear subspace, that is a tuple of p vectors is size n,
        represented as a rectangular matrix of shape (n, p).

        :param subspace: Matrix representation of a linear subspace. Must be one of:
            * numpy.ndarray
            * scipy.sparse.spmatrix
        """
        # Sanitize the matrix attribute and check for potential sparse representation
        if not isinstance(subspace, (numpy.ndarray, scipy.sparse.spmatrix)):
            raise SubspaceError('Matrix representation requires a matrix-like format but received {}'
                                .format(type(subspace)))

        self.is_sparse = scipy.sparse.issparse(subspace)
        self.subspace = subspace

        super().__init__(subspace.dtype, subspace.shape)

    def _matvec(self, x):
        x = numpy.asanyarray(x)
        return self.subspace.dot(x)

    def _rmatvec(self, x):
        x = numpy.asanyarray(x)
        return self.subspace.T.dot(x)

    def dot(self, x) -> MatrixType:
        """
        Override scipy.linalg.linearOperator dot method to handle cases of a sparse input.

        :param x: Input vector, maybe represented in sparse format, to compute the dot product with.
        """
        if scipy.sparse.isspmatrix(x):
            return self.subspace.dot(x)
        else:
            return super().dot(x)

    def _matvec_cost(self):
        return 2 * self.subspace.size

    def _mat(self):
        return self.subspace


SubspaceType = Union[LinearSubspace, numpy.ndarray, scipy.sparse.spmatrix]


class OrthogonalProjector(LinearOperator):
    def __init__(self,
                 V: SubspaceType,
                 factorized: bool = False,
                 ip_B: LinearOperator = None):
        """
        Abstract class for orthogonal projectors in the sens of the self-adjoint positive definite operator ip_B.

        :param V: subspace-type like for linear subspace onto which the projection is made.
        :param factorized: whether to compute QR factorizations of the subspace.
        :param ip_B: self-adjoint positive definite linear operator for which induced inner-product the projector is
            orthogonal
        """
        # Sanitize the subspace argument
        if not isinstance(V, (LinearSubspace, numpy.ndarray, scipy.sparse.spmatrix)):
            raise LinearOperatorError('Subspaces must be of subspace type format but received {}'
                                      .format(type(V)))

        self.V = Subspace(V) if not isinstance(V, LinearSubspace) else V
        self.factorized = factorized
        n, _ = self.V.shape
        self.ip_B = IdentityOperator(n) if ip_B is None else ip_B

        # Process QR factorization in case of orthogonal projection if asked to
        if self.factorized:
            self.VQ, _ = qr(V, ip_B=ip_B)
            self.AVQ = self.ip_B.dot(self.VQ)
            self.L_factor = None
            self.AV = None
        else:
            self.AV = self.ip_B.dot(self.V.mat)
            M = self.V.mat.T.dot(self.AV)
            try:
                M = M.todense()
            except AttributeError:
                pass

            self.L_factor = scipy.linalg.cho_factor(M, lower=True)
            self.VQ = None
            self.AVQ = None

        dtype = self.V.dtype
        super().__init__(dtype, (n, n))

    def _matvec(self, x):
        if self.factorized:
            y = self.AVQ.T.dot(x)
            y = self.VQ.dot(y)
        else:
            y = self.AV.T.dot(x)
            scipy.linalg.cho_solve(self.L_factor, y, overwrite_b=True)
            y = self.V.dot(y)
        return y

    def _rmatvec(self, x):
        if self.factorized:
            y = self.VQ.T.dot(x)
            y = self.AVQ.dot(y)
        else:
            y = self.V.T.dot(x)
            scipy.linalg.cho_solve(self.L_factor, y, overwrite_b=True)
            y = self.AV.dot(y)
        return y

    def _matvec_cost(self):
        if self.factorized:
            return 2 * (self.VQ.size + self.AVQ.size)
        else:
            return self.V.T.matvec_cost + 2 * (self.L_factor[0].size + self.AV.size)
