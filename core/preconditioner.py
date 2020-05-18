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
from core.linop import LinearOperator, MatrixOperator, LinearSubspace, Subspace

__all__ = ['IdentityPreconditioner', 'Jacobi', 'BlockJacobi', 'SymmetricGaussSeidel', 'BlockSymmetricGaussSeidel',
           'AlgebraicMultiGrid', 'CoarseGridCorrection', 'LimitedMemoryPreconditioner', 'PreconditionerGenerator',
           'Preconditioner']

SubspaceType = Union[LinearSubspace, numpy.ndarray, scipy.sparse.spmatrix]


class PreconditionerError(Exception):
    """
    Exception raised when Preconditioner object encounters specific errors.
    """


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
        # Sanitize the linear_op attribute
        if not isinstance(linear_op, LinearOperator):
            raise PreconditionerError('Linear operator must be an instance of LinearOperator.')

        self.linear_op = linear_op

        super().__init__(linear_op.shape, linear_op.dtype)

        self.construction_cost = self._construction_cost()

    def _construction_cost(self):
        return None

    def __rmul__(self, scalar):
        return _ScaledPreconditioner(self, scalar)

    def __add__(self, preconditioner):
        return _SummedPreconditioner(self, preconditioner)

    def __mul__(self, preconditioner):
        return _ComposedPreconditioner(self, preconditioner)

    def __neg__(self):
        return _ScaledPreconditioner(self, -1)

    def __sub__(self, preconditioner):
        return self + (-preconditioner)


class _ScaledPreconditioner(Preconditioner):
    def __init__(self, preconditioner: Preconditioner, scalar: object):
        """
        Abstract representation of scaled preconditioners, that is, the scalar (or external) multiplication of linear
        operators.

        :param preconditioner: Preconditioner as a linear operator involved in the external multiplication.
        :param scalar: Scalar involved in the external multiplication.
        """
        # Sanitize the preconditioner attribute
        if not isinstance(preconditioner, Preconditioner):
            raise PreconditionerError('External product should involve instances of Preconditioner.')

        # Sanitize the scalar attribute
        if not numpy.isscalar(scalar):
            raise PreconditionerError('External product should involve a scalar.')

        self.operands = (scalar, preconditioner)

        super().__init__(preconditioner.linear_op)

    def _matvec(self, x):
        return self.operands[0] * self.operands[1].dot(x)

    def _rmatvec(self, x):
        return numpy.conj(self.operands[0]) * self.operands[1].H.dot(x)

    def _matvec_cost(self):
        n, _ = self.operands[1].shape
        return self.operands[1].matvec_cost + n


class _SummedPreconditioner(Preconditioner):
    def __init__(self, precond1: Preconditioner, precond2: Preconditioner):
        """
        Abstract representation of the sum of two preconditioners, that is, the summation of linear operators.

        :param precond1: First preconditioner involved in the summation.
        :param precond2: Second preconditioner involved in the summation.
        """
        # Sanitize the precond1 and precond2 attributes
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
        return self.operands[0].H.dot(x) + self.operands[1].H.dot(x)

    def _matvec_cost(self):
        n, _ = self.operands[0].shape
        return self.operands[0].matvec_cost + self.operands[1].matvec_cost + n


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

    def _rmatvec(self, x):
        raise NotImplemented('Method _rmatvec not implemented.')

    def _matvec_cost(self):
        cost = self.operands[0].matvec_cost + self.operands[1].matvec_cost + self.linear_op.matvec_cost
        cost = cost + 2 * self.shape[0]
        return cost


class IdentityPreconditioner(Preconditioner):
    def __init__(self, linear_op: LinearOperator):
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

    def __init__(self, matrix_op: MatrixOperator):
        """
        Abstract representation of Jacobi preconditioner. This preconditioner is the inverse of the matrix diagonal.

        :param matrix_op: Matrix to which the Jacobi preconditioner is built on.
        """
        # Sanitize matrix operator argument
        if not isinstance(matrix_op, MatrixOperator):
            raise PreconditionerError('Matrix operator must be an instance of MatrixOperator.')

        # Retrieve diagonal elements of the matrix operator
        if matrix_op.is_sparse:
            diagonal = matrix_op.matrix.diagonal()
        else:
            diagonal = numpy.diag(matrix_op.matrix)

        # Handle potential zeros in the diagonal elements
        diagonal[diagonal == 0] += 1

        self._diagonal = scipy.sparse.diags(1 / diagonal)

        super().__init__(matrix_op)

    def _matvec(self, x):
        return self._diagonal.dot(x)

    def _rmatvec(self, x):
        return self._matvec(x)

    def _matvec_cost(self):
        return self._diagonal.size

    def _construction_cost(self):
        return self._diagonal.size


class BlockJacobi(Preconditioner):
    def __init__(self, matrix_op: MatrixOperator, block_size: int):
        """
        Abstract representation for the block Jacobi preconditioner.

        :param matrix_op: Matrix to which the symmetric Gauss-Seidel preconditioner is built on.
        :param block_size: Desired size for the blocks.
        """
        # Sanitize matrix operator argument
        if not isinstance(matrix_op, MatrixOperator):
            raise PreconditionerError('Matrix operator must be an instance of MatrixOperator.')

        self.matrix = matrix_op.matrix

        # Sanitize block size argument
        if not isinstance(block_size, int) or block_size < 1:
            raise PreconditionerError('Block size must be a positive integer, received {}.'.format(block_size))

        if matrix_op.shape[0] % block_size != 0:
            raise PreconditionerError('Block size inconsistent with operator shape. {} not divisible by {}.'
                                      .format(matrix_op.shape[0], block_size))

        self.block_size = block_size

        super().__init__(matrix_op)

    def _matvec(self, x):
        y = numpy.zeros_like(x)
        pyamg.relaxation.relaxation.block_jacobi(self.matrix, y, x, blocksize=self.block_size)
        return y

    def _matvec_cost(self):
        return self.linear_op.matvec_cost


class SymmetricGaussSeidel(Preconditioner):
    def __init__(self, matrix_op: MatrixOperator):
        """
        Abstract representation for the symmetric Gauss-Seidel preconditioner. Formally, if the matrix operator A is
        split such that A = L + D + L.T with D the diagonal and L the strict lower triangular part of A, then the
        preconditioner has the following closed form:

            M = (D + L) * D^(-1) * (D + L^T)

        :param matrix_op: Matrix to which the symmetric Gauss-Seidel preconditioner is built on.
        """
        # Sanitize matrix operator argument
        if not isinstance(matrix_op, MatrixOperator):
            raise PreconditionerError('Matrix operator must be an instance of MatrixOperator.')

        self.matrix = matrix_op.matrix

        super().__init__(matrix_op)

    def _matvec(self, x):
        y = numpy.zeros_like(x)
        pyamg.relaxation.relaxation.gauss_seidel(self.matrix, y, x, sweep='symmetric')
        return y

    def _matvec_cost(self):
        return self.linear_op.matvec_cost


class BlockSymmetricGaussSeidel(Preconditioner):
    def __init__(self, matrix_op: MatrixOperator, block_size: int):
        """
        Abstract representation for the block symmetric Gauss-Seidel preconditioner.

        :param matrix_op: Matrix to which the symmetric Gauss-Seidel preconditioner is built on.
        :param block_size: Desired size for the blocks.
        """
        # Sanitize matrix operator argument
        if not isinstance(matrix_op, MatrixOperator):
            raise PreconditionerError('Matrix operator must be an instance of MatrixOperator.')

        self.matrix = matrix_op.matrix

        # Sanitize block size argument
        if not isinstance(block_size, int) or block_size < 1:
            raise PreconditionerError('Block size must be a positive integer, received {}.'.format(block_size))

        if matrix_op.shape[0] % block_size != 0:
            raise PreconditionerError('Block size inconsistent with operator shape. {} not divisible by {}.'
                                      .format(matrix_op.shape[0], block_size))

        self.block_size = block_size

        super().__init__(matrix_op)

    def _matvec(self, x):
        y = numpy.zeros_like(x)
        pyamg.relaxation.relaxation.block_gauss_seidel(self.matrix, y, x, blocksize=self.block_size, sweep='symmetric')
        return y

    def _matvec_cost(self):
        return self.linear_op.matvec_cost


class AlgebraicMultiGrid(Preconditioner):
    def __init__(self, matrix_op: MatrixOperator,
                 heuristic: str = 'ruge_stuben',
                 n_cycles: int = 1,
                 **kwargs):
        """
        Abstract representation for Algebraic Multi-grid (AMG) preconditioners. Several heuristics are available from
        the PyAMG python library.

        :param matrix_op: Matrix to which the algebraic multi-grid preconditioner is built on.
        :param heuristic: Name of the algebraic multi-grid heuristic used to construct the hierarchy.
        :param n_cycles: Number of cycles done per application of the multi-grid method as a preconditioner.
        """
        # Sanitize heuristic argument
        if heuristic not in ['ruge_stuben', 'smoothed_aggregated', 'rootnode']:
            raise PreconditionerError('AMG heuristic {} unknown.'.format(heuristic))

        # Setup multi-grid hierarchical structure with corresponding heuristic
        if heuristic == 'ruge_stuben':
            self.amg = pyamg.ruge_stuben_solver(matrix_op.matrix.tocsr(), **kwargs)

        elif heuristic == 'smoothed_aggregated':
            self.amg = pyamg.smoothed_aggregation_solver(matrix_op.matrix.tocsr(), **kwargs)

        elif heuristic == 'rootnode':
            self.amg = pyamg.rootnode_solver(matrix_op.matrix.tocsr(), **kwargs)

        # Sanitize the number of cycles argument
        if not isinstance(n_cycles, int) or n_cycles < 1:
            raise PreconditionerError('Number of cycles must be a positive integer, received {}.'.format(n_cycles))

        self.n_cycles = n_cycles

        super().__init__(matrix_op)

    def _matvec(self, x):
        y = self.amg.solve(x, tol=1e-15, maxiter=self.n_cycles, cycle='F', accel=None)

        return numpy.atleast_2d(y).T

    def _matvec_cost(self):
        return self.shape[0]


class CoarseGridCorrection(Preconditioner):
    def __init__(self, linear_op: LinearOperator, subspace: SubspaceType, rank_tol: float = 1e-10):
        """
        Abstract representation for Coarse-Grid Correction preconditioner. Formally, for a given subspace S, and linear
        operator A, the coarse-grid correction has the following closed form:

            Q = S * (S.T * A * S)^(-1) * S.T

        :param linear_op: Linear operator to coarsen.
        :param subspace: Subspace on which the linear operator is restricted and interpolated.
        :param rank_tol: Numerical rank tolerance to state a rank deficiency of the subspace.
        """
        # Sanitize the subspace attribute
        if not isinstance(subspace, (LinearSubspace, numpy.ndarray, scipy.sparse.spmatrix)):
            raise PreconditionerError('Subspace should either be numpy.ndarray or scipy.sparse.spmatrix.')

        if not isinstance(subspace, LinearSubspace):
            subspace = Subspace(subspace)

        self._subspace = subspace

        # Compute the Galerkin operator S^T @ A @ S
        self._reduced_linear_op = self._subspace.T @ linear_op @ self._subspace

        # Cholesky decomposition of the reduced operator
        self._cho_factor = scipy.linalg.cho_factor(self._reduced_linear_op.mat,
                                                   lower=True,
                                                   overwrite_a=False)

        super().__init__(linear_op)

        self.construction_cost = self._construction_cost()

    def _matvec(self, x):
        y = self._subspace.T.dot(x)
        scipy.linalg.cho_solve(self._cho_factor, y, overwrite_b=True)
        y = self._subspace.dot(y)
        return y

    def _matvec_cost(self):
        _, k = self._subspace.shape
        return 2 * self._subspace.matvec_cost + 2 * k**2

    def _construction_cost(self):
        n, k = self._subspace.shape
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
