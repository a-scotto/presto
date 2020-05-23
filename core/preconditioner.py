#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on April 17, 2019 at 13:15.

@author: a.scotto

Description:
"""

import pyamg
import numpy
import scipy.linalg
import scipy.sparse.linalg
import pyamg.relaxation.relaxation

from typing import Union
from core.algebra import *

__all__ = ['IdentityPreconditioner', 'Jacobi', 'BlockJacobi', 'SymmetricGaussSeidel', 'BlockSymmetricGaussSeidel',
           'AlgebraicMultiGrid', 'CoarseGridCorrection', 'LimitedMemoryPreconditioner', 'PreconditionerGenerator',
           'Preconditioner']

SubspaceType = Union[LinearSubspace, numpy.ndarray, scipy.sparse.spmatrix]


class IdentityPreconditioner(Preconditioner):
    def __init__(self, linear_op: LinearOperator):
        """
        Abstract representation of the identity preconditioner.

        :param linear_op: Linear operator to precondition.
        """
        super().__init__(linear_op)

    def _matvec(self, x):
        return x

    def _rmatvec(self, x):
        return x

    def _matvec_cost(self):
        return 0.

    def _construction_cost(self):
        return 0.


class Jacobi(Preconditioner):

    def __init__(self, linear_op: LinearOperator):
        """
        Abstract representation of Jacobi preconditioner, that is the inverse of the matrix diagonal.

        :param linear_op: Linear operator to build the Jacobi preconditioner on.
        """
        # Sanitize matrix operator argument
        if not isinstance(linear_op, MatrixOperator):
            raise PreconditionerError('Jacobi preconditioner requires an instance of LinearOperator.')

        # Retrieve diagonal elements of the matrix operator
        matrix_repr = linear_op.mat
        if scipy.sparse.isspmatrix(matrix_repr):
            diagonal = matrix_repr.diagonal()
        else:
            diagonal = numpy.diag(matrix_repr)

        # Handle potential zeros in the diagonal elements
        diagonal[diagonal == 0] += 1.

        self._diagonal = scipy.sparse.diags(1. / diagonal)

        super().__init__(linear_op)

    def _matvec(self, x):
        return self._diagonal.dot(x)

    def _rmatvec(self, x):
        return self._diagonal.dot(x)

    def _matvec_cost(self):
        return self._diagonal.size

    def _construction_cost(self):
        return self._diagonal.size


class BlockJacobi(Preconditioner):
    def __init__(self, linear_op: LinearOperator, block_size: int):
        """
        Abstract representation for the block Jacobi preconditioner.

        :param linear_op: Linear operator to build the block Jacobi preconditioner on.
        :param block_size: Desired size for the blocks.
        """
        # Sanitize matrix operator argument
        if not isinstance(linear_op, LinearOperator):
            raise PreconditionerError('Block Jacobi preconditioner requires an instance of LinearOperator.')

        self.matrix = linear_op.mat

        # Sanitize block size argument
        if not isinstance(block_size, int) or block_size < 1:
            raise PreconditionerError('Block size must be a positive integer, received {}.'.format(block_size))

        if linear_op.shape[0] % block_size != 0:
            raise PreconditionerError('Block size inconsistent with operator shape. {} not divisible by {}.'
                                      .format(linear_op.shape[0], block_size))

        self.block_size = block_size

        super().__init__(linear_op)

    def _matvec(self, x):
        y = numpy.zeros_like(x)
        pyamg.relaxation.relaxation.block_jacobi(self.matrix, y, x, blocksize=self.block_size)
        return y

    def _matvec_cost(self):
        raise NotImplementedError('Matrix-vector product computational cost for class {} not yet implemented.'
                                  .format(self.__class__.__name__))

    def _construction_cost(self):
        raise NotImplementedError('Construction cost for class {} not yet implemented.'
                                  .format(self.__class__.__name__))


class SymmetricGaussSeidel(Preconditioner):
    def __init__(self, linear_op: LinearOperator):
        """
        Abstract representation for the symmetric Gauss-Seidel preconditioner. Formally, if the matrix operator A is
        split such that A = L + D + L.T with D the diagonal and L the strict lower triangular part of A, then the
        preconditioner has the following closed form:

            M = (D + L) * D^(-1) * (D + L^T)

        :param linear_op: Linear operator to build the symmetric Gauss-Seidel preconditioner on.
        """
        # Sanitize matrix operator argument
        if not isinstance(linear_op, LinearOperator):
            raise PreconditionerError('Symmetric Gauss-Seidel preconditioner requires an instance of LinearOperator.')

        self.matrix = linear_op.mat

        super().__init__(linear_op)

    def _matvec(self, x):
        y = numpy.zeros_like(x)
        pyamg.relaxation.relaxation.gauss_seidel(self.matrix, y, x, sweep='symmetric')
        return y

    def _matvec_cost(self):
        raise NotImplementedError('Matrix-vector product computational cost for class {} not yet implemented.'
                                  .format(self.__class__.__name__))

    def _construction_cost(self):
        raise NotImplementedError('Construction cost for class {} not yet implemented.'
                                  .format(self.__class__.__name__))


class BlockSymmetricGaussSeidel(Preconditioner):
    def __init__(self, linear_op: LinearOperator, block_size: int):
        """
        Abstract representation for the block symmetric Gauss-Seidel preconditioner.

        :param linear_op: Linear operator to build the block symmetric Gauss-Seidel preconditioner on.
        :param block_size: Desired size for the blocks.
        """
        # Sanitize matrix operator argument
        if not isinstance(linear_op, LinearOperator):
            raise PreconditionerError('Block symmetric Gauss-Seidel preconditioner requires an instance of '
                                      'LinearOperator.')

        self.matrix = linear_op.mat

        # Sanitize block size argument
        if not isinstance(block_size, int) or block_size < 1:
            raise PreconditionerError('Block size must be a positive integer, received {}.'.format(block_size))

        if linear_op.shape[0] % block_size != 0:
            raise PreconditionerError('Block size inconsistent with operator shape. {} not divisible by {}.'
                                      .format(linear_op.shape[0], block_size))

        self.block_size = block_size

        super().__init__(linear_op)

    def _matvec(self, x):
        y = numpy.zeros_like(x)
        pyamg.relaxation.relaxation.block_gauss_seidel(self.matrix, y, x, blocksize=self.block_size, sweep='symmetric')
        return y

    def _matvec_cost(self):
        raise NotImplementedError('Matrix-vector product computational cost for class {} not yet implemented.'
                                  .format(self.__class__.__name__))

    def _construction_cost(self):
        raise NotImplementedError('Construction cost for class {} not yet implemented.'
                                  .format(self.__class__.__name__))


class AlgebraicMultiGrid(Preconditioner):
    def __init__(self, linear_op: LinearOperator,
                 heuristic: str = 'ruge_stuben',
                 n_cycles: int = 1,
                 **kwargs):
        """
        Abstract representation for Algebraic Multi-grid (AMG) class of preconditioners. Available heuristics are the
        one from the PyAMG Python library.

        :param linear_op: Linear operator to build the algebraic multi-grid preconditioner on.
        :param heuristic: Name of the algebraic multi-grid heuristic used for multi-level hierarchy construction.
        :param n_cycles: Number of cycles done per application of the multi-grid method as a preconditioner.
        """
        # Sanitize the heuristic argument
        if heuristic not in ['ruge_stuben', 'smoothed_aggregated', 'rootnode']:
            raise PreconditionerError('AMG heuristic {} unknown.'.format(heuristic))

        matrix_repr = linear_op.mat

        # Setup multi-grid hierarchical structure with corresponding heuristic
        if heuristic == 'ruge_stuben':
            self.amg = pyamg.ruge_stuben_solver(matrix_repr.tocsr(), **kwargs)

        elif heuristic == 'smoothed_aggregated':
            self.amg = pyamg.smoothed_aggregation_solver(matrix_repr.tocsr(), **kwargs)

        elif heuristic == 'rootnode':
            self.amg = pyamg.rootnode_solver(matrix_repr.tocsr(), **kwargs)

        # Sanitize the number of cycles argument
        if not isinstance(n_cycles, int) or n_cycles < 1:
            raise PreconditionerError('Number of cycles must be a positive integer, received {}.'.format(n_cycles))

        self.n_cycles = n_cycles

        super().__init__(linear_op)

    def _matvec(self, x):
        y = self.amg.solve(x, tol=1e-15, maxiter=self.n_cycles, cycle='W', accel=None)
        return numpy.atleast_2d(y).T

    def _matvec_cost(self):
        raise NotImplementedError('Matrix-vector product computational cost for class {} not yet implemented.'
                                  .format(self.__class__.__name__))

    def _construction_cost(self):
        raise NotImplementedError('Construction cost for class {} not yet implemented.'
                                  .format(self.__class__.__name__))


class CoarseGridCorrection(Preconditioner):
    def __init__(self, linear_op: LinearOperator, subspace: SubspaceType):
        """
        Abstract representation for Coarse-Grid Correction preconditioner. Formally, for a given subspace S, and linear
        operator A, the coarse-grid correction has the following closed form:

            Q = S * (S.T * A * S)^(-1) * S.T

        :param linear_op: Linear operator to coarsen.
        :param subspace: Subspace on which the linear operator is restricted and interpolated.
        """
        # Sanitize the subspace attribute
        if not isinstance(subspace, (LinearSubspace, numpy.ndarray, scipy.sparse.spmatrix)):
            raise PreconditionerError('Subspace should either be numpy.ndarray or scipy.sparse.spmatrix.')

        self._S = Subspace(subspace) if not isinstance(subspace, LinearSubspace) else subspace

        # Compute the Galerkin operator S^T @ A @ S
        self._A = (self._S.T @ linear_op @ self._S).mat
        try:
            self._A = self._A.todense()
        except AttributeError:
            pass

        # Cholesky decomposition of the reduced operator
        self._cho_factor = scipy.linalg.cho_factor(self._A,
                                                   lower=True,
                                                   overwrite_a=False)

        super().__init__(linear_op)

        self.construction_cost = self._construction_cost()

    def _matvec(self, x):
        y = self._S.T.dot(x)
        scipy.linalg.cho_solve(self._cho_factor, y, overwrite_b=True)
        y = self._S.dot(y)
        return y

    def _matvec_cost(self):
        _, k = self._S.shape
        return 2 * self._S.matvec_cost + 2 * k**2

    def _construction_cost(self):
        n, k = self._S.shape
        cost = 2 * n * k**2 + 2 * k**3
        cost += 2 * k * self.linear_op.matvec_cost + 2 * n * k**2
        cost += 2 / 3 * k**3
        return cost


class LimitedMemoryPreconditioner(Preconditioner):
    def __init__(self, linear_op: LinearOperator, subspace: SubspaceType, M: Preconditioner = None):
        """
        Abstract representation for Limited Memory Preconditioner (LMP). Formally, for Q being the Coarse-Grid
        correction operator defined with subspace S, linear operator A, and first-level preconditioner M, it has the
        following closed form:

            H = (I - Q*A) * M * (I - A*Q) + Q
        Constructor of the Limited-Memory Preconditioner.

        :param linear_op: Linear operator to built the LMP on.
        :param subspace: Subspace on which the projection is made.
        :param M: First-level preconditioner.
        """
        # Sanitize the first-level preconditioner attribute
        M = IdentityPreconditioner(linear_op) if M is None else M

        if not isinstance(M, Preconditioner) or M.linear_op is not linear_op:
            raise PreconditionerError('Preconditioner must be an instance of Preconditioner related to the same '
                                      'linear operator as provided for the LMP construction.')

        Q = CoarseGridCorrection(linear_op, subspace)

        self.lmp = Q * M * Q

        super().__init__(linear_op)

    def _matvec(self, x):
        return self.lmp.matvec(x)

    def _matvec_cost(self):
        return self.lmp.matvec_cost

    def _construction_cost(self):
        return self.lmp.construction_cost


class PreconditionerGenerator(object):
    def __init__(self, linear_op: LinearOperator):
        """
        Abstract class for Preconditioner Generator. Returns all algebraic type preconditioners, that is preconditioners
        solely making use of the linear operator.
        """
        self.linear_op = linear_op

    def get(self, preconditioner: str, *args, **kwargs) -> Preconditioner:
        """
        Method to get preconditioners.

        :param preconditioner: Name of the preconditioner to build.
        :param args: Optional arguments for preconditioner.
        :param kwargs: Optional named arguments for preconditioner.
        """

        if preconditioner == 'identity':
            return IdentityPreconditioner(self.linear_op)

        elif preconditioner == 'jacobi':
            return Jacobi(self.linear_op)

        elif preconditioner == 'block_jacobi':
            return BlockJacobi(self.linear_op, *args, **kwargs)

        elif preconditioner == 'sym_gs':
            return SymmetricGaussSeidel(self.linear_op)

        elif preconditioner == 'block_sym_gs':
            return BlockSymmetricGaussSeidel(self.linear_op, *args, **kwargs)

        elif preconditioner == 'amg':
            return AlgebraicMultiGrid(self.linear_op, *args, **kwargs)

        else:
            raise PreconditionerError('Preconditioner {} not implemented. Please check syntax'.format(preconditioner))
