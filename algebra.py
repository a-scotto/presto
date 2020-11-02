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
from scipy.sparse.linalg import LinearOperator as scipyLinearOperator

__all__ = ['LinearOperator', 'IdentityOperator', 'LinearSubspace', 'Subspace', 'MatrixOperator', 'OrthogonalProjector',
           'Preconditioner', 'PreconditionerError']

MatrixType = Union[numpy.ndarray, scipy.sparse.spmatrix]


class LinearOperatorError(Exception):
    """
    Exception raised when LinearOperator object encounters specific errors.
    """


class LinearSubspaceError(Exception):
    """
    Exception raised when KrylovSubspace object encounters specific errors.
    """


class PreconditionerError(Exception):
    """
    Exception raised when Preconditioner object encounters specific errors.
    """


class LinearOperator(scipyLinearOperator):
    def __init__(self, dtype: object, shape: tuple):
        """
        Abstract representation of linear operators. Extends the existing LinearOperator class in Python library
        scipy.linalg.LinearOperator so as to handle sparse matrix representations, and some additional attributes.

        :param dtype: Type of the elements in the linear operator.
        :param shape: Shape of the linear operator, i.e. the tuple (n, p) such that the linear operator maps R^n to R^p.
        """
        # Sanitize the data type argument
        try:
            self.dtype = numpy.dtype(dtype)
        except TypeError:
            raise LinearOperatorError('dtype provided not understood.')

        # Sanitize the shape argument
        if not isinstance(shape, tuple) or len(shape) != 2:
            if not isinstance(shape[0], int) or not isinstance(shape[1], int):
                raise LinearOperatorError('Shape must be a tuple of integers of the form (n, p).')

        self.shape = shape

        try:
            self.matvec_cost = self._matvec_cost()
        except NotImplementedError:
            self.matvec_cost = numpy.Inf

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
            return _ImageLinearSubspace(self, other)
        elif isinstance(other, (numpy.ndarray, scipy.sparse.spmatrix)):
            return self.dot(other)
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
        # Sanitize the linear operator argument
        if not isinstance(linear_op, LinearOperator):
            raise LinearOperatorError('External product should involve an instance of LinearOperator.')

        # Sanitize the scalar argument
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
        # Sanitize the linear operators arguments
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
        # Sanitize the linear operators arguments
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
        try:
            return self.operands[0] @ self.operands[1].mat
        except ValueError:
            return self.operands[0] @ self.operands[1].mat.todense()


class _AdjointLinearOperator(LinearOperator):
    def __init__(self, linear_op: LinearOperator):
        """
        Abstract representation of the adjoint of a linear operator.

        :param linear_op: Linear operator to define the adjoint operator of.
        """
        # Sanitize the linear operator argument
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
        return _SummedLinearSubspace(self, subspace)

    def __mul__(self, other):
        if (isinstance(other, LinearOperator) and not isinstance(other, LinearSubspace))\
                and isinstance(self, _AdjointLinearSubspace):
            return _ImageLinearSubspace(other, self._adjoint()).T
        elif isinstance(other, (numpy.ndarray, scipy.sparse.spmatrix)):
            return self.dot(other)
        else:
            return _ComposedLinearSubspace(self, other)

    def _adjoint(self):
        return _AdjointLinearSubspace(self)


class _SummedLinearSubspace(LinearSubspace):
    def __init__(self, subspace1: LinearSubspace, subspace2: LinearSubspace):
        """
        Abstract representation of the addition of two linear subspaces, that is, of the sum F + G of linear subspace.

        :param subspace1: First subspace involved in the summation.
        :param subspace2: Second subspace involved in the summation.
        """
        # Sanitize the subspaces arguments
        if not isinstance(subspace1, LinearSubspace) or not isinstance(subspace2, LinearSubspace):
            raise LinearSubspaceError('Both operands in summation must be instances of LinearSubspace.')

        # Check the consistency of subspaces shapes
        if subspace1.shape[0] != subspace2.shape[0]:
            raise LinearSubspaceError('Operands in summation have inconsistent shapes: {} and {}.'
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
        if scipy.sparse.isspmatrix(mat1) and scipy.sparse.isspmatrix(mat2):
            return scipy.sparse.hstack([self.operands[0].mat, self.operands[1].mat])
        elif not scipy.sparse.isspmatrix(mat1) and scipy.sparse.isspmatrix(mat2):
            return numpy.hstack([self.operands[0].mat, self.operands[1].mat.todense()])
        elif scipy.sparse.isspmatrix(mat1) and not scipy.sparse.isspmatrix(mat2):
            return numpy.hstack([self.operands[0].mat.todense(), self.operands[1].mat])
        else:
            return numpy.hstack([self.operands[0].mat, self.operands[1].mat])


class _ComposedLinearSubspace(LinearSubspace):
    def __init__(self, subspace1: LinearSubspace, subspace2: LinearSubspace):
        """
        Abstract representation of the composition of two linear subspaces.

        :param subspace1: First linear subspace involved in the composition.
        :param subspace2: Second linear subspace involved in the composition.
        """
        # Sanitize the linear operators arguments
        if not isinstance(subspace1, LinearOperator) or not isinstance(subspace2, LinearOperator):
            raise LinearSubspaceError('Both operands in composition must be instances of LinearSubspace.')

        # Check the consistency of linear operators shapes
        if subspace1.shape[1] != subspace2.shape[0]:
            raise LinearSubspaceError('Operands in composition have inconsistent shapes: {} and {}.'
                                      .format(subspace1.shape, subspace2.shape))

        # Initialize operands attribute
        self.operands = (subspace1, subspace2)

        # Resulting shape (n, p) @ (p, m) -> (n, m)
        shape = (subspace1.shape[0], subspace2.shape[1])
        dtype = numpy.find_common_type([subspace1.dtype, subspace2.dtype], [])

        super().__init__(dtype, shape)

    def _matvec(self, x):
        return self.operands[0].dot(self.operands[1].dot(x))

    def _rmatvec(self, x):
        return self.operands[1].T.dot(self.operands[0].T.dot(x))

    def _matvec_cost(self):
        return self.operands[0].matvec_cost + self.operands[1].matvec_cost

    def _mat(self):
        try:
            return self.operands[0] @ self.operands[1].mat
        except ValueError:
            return self.operands[0] @ self.operands[1].mat.todense()


class _ImageLinearSubspace(LinearSubspace):
    def __init__(self, linear_op: LinearOperator, subspace: LinearSubspace):
        """
        Abstract representation of the image of a subspace under the action of a linear operator.

        :param linear_op: Linear operator to be acting on the given subspace.
        :param subspace: Subspace which image is considered.
        """
        # Sanitize the linear operator and subspace arguments
        if not isinstance(linear_op, LinearOperator) or not isinstance(subspace, LinearSubspace):
            raise LinearSubspaceError('Image of a subspace requires instances of LinearOperator and LinearSubspace.')

        # Check the consistency of operands shapes
        if linear_op.shape[1] != subspace.shape[0]:
            raise LinearSubspaceError('Operands for the image have inconsistent shapes: {} and {}.'
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
        try:
            return self.operands[0] @ self.operands[1].mat
        except ValueError:
            return self.operands[0] @ self.operands[1].mat.todense()


class _AdjointLinearSubspace(LinearSubspace):
    def __init__(self, subspace: LinearSubspace):
        """
        Abstract representation of the adjoint of a linear subspace.

        :param subspace: Linear subspace to define the adjoint operator of.
        """
        # Sanitize the linear subspace argument
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


class Preconditioner(LinearOperator):
    def __init__(self, linear_op: LinearOperator):
        """
        Abstract representation of preconditioners. Preconditioners are linear operators dedicated to improve the
        convergence of iterative methods for the solution of linear systems. For a given linear system involving a
        linear operator A and a right-hand side b, the preconditioned linear system is written as MAx = Mb, where M is
        a preconditioner. The preconditioner is therefore closely related to the linear operator A, hence the following
        constructor for this abstract class.

        :param linear_op: Linear operator to which the preconditioner is related.
        """
        # Sanitize the linear operator argument
        if not isinstance(linear_op, LinearOperator):
            raise PreconditionerError('Linear operator must be an instance of LinearOperator.')

        self.linear_op = linear_op

        super().__init__(linear_op.dtype, linear_op.shape)

        try:
            self.construction_cost = self._construction_cost()
        except NotImplementedError:
            self.construction_cost = numpy.Inf

    def _construction_cost(self):
        """
        Method to get the computational cost of the construction of the preconditioner.
        """
        raise NotImplementedError('Method _construction_cost not implemented for class objects {}.'
                                  .format(self.__class__))

    def __rmul__(self, scalar):
        return _ScaledPreconditioner(self, scalar)

    def __add__(self, preconditioner):
        return _SummedPreconditioner(self, preconditioner)

    def __mul__(self, other):
        if isinstance(other, Preconditioner):
            return _ComposedPreconditioner(self, other)
        elif isinstance(other, (numpy.ndarray, scipy.sparse.spmatrix)):
            return self.dot(other)
        else:
            return _ComposedLinearOperator(self, other)

    def __neg__(self):
        return _ScaledPreconditioner(self, -1)

    def __sub__(self, preconditioner):
        return self + (-preconditioner)


class _ScaledPreconditioner(Preconditioner):
    def __init__(self, preconditioner: Preconditioner, scalar: numpy.ScalarType):
        """
        Abstract representation of scaled preconditioners, that is, the scalar (or external) multiplication of linear
        operators.

        :param preconditioner: Preconditioner involved in the external multiplication.
        :param scalar: Scalar involved in the external multiplication.
        """
        # Sanitize the preconditioner argument
        if not isinstance(preconditioner, Preconditioner):
            raise PreconditionerError('External product should involve instances of Preconditioner.')

        # Sanitize the scalar argument
        if not numpy.isscalar(scalar):
            raise PreconditionerError('External product should involve a scalar.')

        # Initialize operands attribute
        self.operands = (scalar, preconditioner)

        super().__init__(preconditioner.linear_op)

    def _matvec(self, x):
        return self.operands[0] * self.operands[1].dot(x)

    def _rmatvec(self, x):
        return numpy.conj(self.operands[0]) * self.operands[1].T.dot(x)

    def _matvec_cost(self):
        n, _ = self.operands[1].shape
        return self.operands[1].matvec_cost + n

    def _construction_cost(self):
        return self.operands[1].construction_cost


class _SummedPreconditioner(Preconditioner):
    def __init__(self, precond1: Preconditioner, precond2: Preconditioner):
        """
        Abstract representation of the sum of two preconditioners, that is, the summation of linear operators.

        :param precond1: First preconditioner involved in the summation.
        :param precond2: Second preconditioner involved in the summation.
        """
        # Sanitize the preconditioners arguments
        if not isinstance(precond1, Preconditioner) or not isinstance(precond2, Preconditioner):
            raise PreconditionerError('Both operands in summation must be instances of Preconditioner.')

        # Check linear operator consistency
        if not (precond1.linear_op is precond2.linear_op):
            raise PreconditionerError('Both operands must be preconditioners of the same linear operator.')

        self.operands = (precond1, precond2)

        super().__init__(precond1.linear_op)

    def _matvec(self, x):
        return self.operands[0].dot(x) + self.operands[1].dot(x)

    def _rmatvec(self, x):
        return self.operands[0].T.dot(x) + self.operands[1].T.dot(x)

    def _matvec_cost(self):
        n, _ = self.operands[0].shape
        return self.operands[0].matvec_cost + self.operands[1].matvec_cost + n

    def _construction_cost(self):
        return self.operands[0].construction_cost + self.operands[1].construction_cost


class _ComposedPreconditioner(Preconditioner):
    def __init__(self, precond1: Preconditioner, precond2: Preconditioner):
        """
        Abstract representation of the composition of two preconditioner, that is, the composition of linear operators.

        :param precond1: First preconditioner involved in the composition.
        :param precond2: Second preconditioner involved in the composition.
        """
        # Sanitize the precond1 and precond2 attribute
        if not isinstance(precond1, Preconditioner) or not isinstance(precond2, Preconditioner):
            raise PreconditionerError('Both operands in composition must be instances of Preconditioner.')

        # Check linear operator consistency
        if not (precond1.linear_op is precond2.linear_op):
            raise PreconditionerError('Both operands must be preconditioners of the same linear operator.')

        self.operands = (precond1, precond2)

        super().__init__(precond1.linear_op)

    def _matvec(self, x):
        y = self.operands[0].dot(x)
        z = x - self.linear_op.dot(y)
        return y + self.operands[1].dot(z)

    def _matvec_cost(self):
        n, _ = self.shape
        cost = self.operands[0].matvec_cost + self.operands[1].matvec_cost + self.linear_op.matvec_cost + 2 * n
        return cost

    def _construction_cost(self):
        return self.operands[0].construction_cost + self.operands[1].construction_cost


class IdentityOperator(LinearOperator):
    def __init__(self, n: int):
        """
        Abstract representation for identity linear operator.

        :param n: Dimension of the vector space identity operator.
        """
        # Sanitize the order argument
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
        # Sanitize the matrix argument and check for potential sparse representation
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
        return self.matrix.dot(x)

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
        # Sanitize the matrix argument and check for potential sparse representation
        if not isinstance(subspace, (numpy.ndarray, scipy.sparse.spmatrix)):
            raise LinearSubspaceError('Matrix representation requires a matrix-like format but received {}'
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

    def _matvec_cost(self):
        return 2 * self.subspace.size

    def _mat(self):
        return self.subspace


SubspaceType = Union[LinearSubspace, numpy.ndarray, scipy.sparse.spmatrix]


class OrthogonalProjector(LinearOperator):
    def __init__(self,
                 V: SubspaceType,
                 factorize: bool = False,
                 ip_A: LinearOperator = None):
        """
        Abstract class for orthogonal projectors in the sens of the self-adjoint positive definite operator ip_B.

        :param V: subspace-type like for linear subspace onto which the projection is made.
        :param factorize: whether to compute QR factorizations of the subspace.
        :param ip_A: self-adjoint positive definite linear operator for which induced inner-product the projector is
            orthogonal
        """
        # Sanitize the subspace argument
        if not isinstance(V, (LinearSubspace, numpy.ndarray, scipy.sparse.spmatrix)):
            raise LinearOperatorError('Subspaces must be of subspace type format but received {}'
                                      .format(type(V)))

        n, _ = V.shape
        self.V = Subspace(V) if not isinstance(V, LinearSubspace) else V
        self.ip_A = IdentityOperator(n) if ip_A is None else ip_A
        self.factorize = factorize

        self.AV = self.ip_A @ self.V
        self.L_factor = (self.V.T @ self.AV).mat
        self.L_factor = self.L_factor.todense() if scipy.sparse.isspmatrix(self.L_factor) else self.L_factor
        self.L_factor = scipy.linalg.cho_factor(self.L_factor, lower=True, overwrite_a=False)

        # Process QR factorization in case of orthogonal projection if asked to
        if self.factorize:
            try:
                self.V = Subspace(scipy.linalg.solve_triangular(self.L_factor[0], self.V.mat.T, lower=True).T)
            except ValueError:
                self.V = Subspace(scipy.linalg.solve_triangular(self.L_factor[0], self.V.mat.todense().T, lower=True).T)

            self.AV = Subspace(scipy.linalg.solve_triangular(self.L_factor[0], self.AV.mat.T, lower=True).T)

        dtype = self.V.dtype
        super().__init__(dtype, (n, n))

        # Set complementary projector
        self.complementary = IdentityOperator(n) - self

    def _matvec(self, x):
        y = self.AV.T.dot(x)
        if not self.factorize:
            y = scipy.linalg.cho_solve(self.L_factor, y, overwrite_b=False)
        y = self.V.dot(y)
        return y

    def _rmatvec(self, x):
        y = self.V.T.dot(x)
        if not self.factorize:
            y = scipy.linalg.cho_solve(self.L_factor, y, overwrite_b=False)
        y = self.AV.dot(y)
        return y

    def _matvec_cost(self):
        cost = self.V.matvec_cost + self.AV.T.matvec_cost
        if not self.factorize:
            cost += self.L_factor[0].size
        return cost

    def _mat(self):
        if self.factorize:
            return self.V @ self.AV.T.mat
        else:
            try:
                Y = scipy.linalg.cho_solve(self.L_factor, self.AV.T.mat, overwrite_b=False)
            except ValueError:
                Y = scipy.linalg.cho_solve(self.L_factor, self.AV.T.mat.todense(), overwrite_b=False)
            return self.V @ Y
