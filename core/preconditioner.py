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
import pyamg.relaxation.relaxation
import scipy.sparse.linalg

from core.linear_operator import LinearOperator, MatrixOperator


class PreconditionerError(Exception):
    """
    Exception raised when LinearOperator object encounters specific errors.
    """


class Preconditioner(LinearOperator):
    """
    Abstract class for iterative methods preconditioner.
    """

    def __init__(self, shape: tuple, dtype: object, lin_op: LinearOperator) -> None:
        # Sanitize the lin_op attribute
        if not isinstance(lin_op, LinearOperator):
            raise PreconditionerError('Preconditioner should be built from LinearOperator.')

        # Check operators compatibility
        if lin_op.shape != shape:
            raise PreconditionerError('Preconditioner must have the same shape as LinearOperator.')

        self.lin_op = lin_op
        self.name = 'unnamed'

        super().__init__(shape, dtype)

    def _apply(self, x):
        raise PreconditionerError('Preconditioner object must implement an _apply method.')

    def _matvec(self, x):
        return self._apply(x)

    def apply(self, x):
        return self._apply(x)

    def __rmul__(self, scalar):
        return _ScaledPreconditioner(self, scalar)

    def __add__(self, preconditioner):
        return _SummedPreconditioner(self, preconditioner)

    def __mul__(self, preconditioner):
        return _ComposedPreconditioner(self, preconditioner)


class _ScaledPreconditioner(Preconditioner):
    """
    Abstract class for scaled preconditioner.
    """

    def __init__(self, preconditioner: Preconditioner, scalar: object) -> None:
        # Sanitize the preconditioner attribute
        if not isinstance(preconditioner, Preconditioner):
            raise PreconditionerError('External product should involve a Preconditioner.')

        # Sanitize the scalar attribute
        if not numpy.isscalar(scalar):
            raise PreconditionerError('External product should involve a scalar.')

        self.operands = (scalar, preconditioner)

        shape = preconditioner.shape
        dtype = numpy.find_common_type([preconditioner.dtype], [type(scalar)])
        lin_op = preconditioner.lin_op

        super().__init__(shape, dtype, lin_op)

    def _apply(self, X):
        return self.operands[0] * self.operands[1].dot(X)

    def _matvec_cost(self):
        m, _ = self.operands[1].shape
        return self.operands[1].apply_cost + 2 * m


class _SummedPreconditioner(Preconditioner):
    """
    Abstract class for summation of preconditioner.
    """

    def __init__(self, precond1: Preconditioner, precond2: Preconditioner) -> None:
        # Sanitize the precond1 and precond2 attributes
        if not isinstance(precond1, Preconditioner) or not isinstance(precond2, Preconditioner):
            raise PreconditionerError('Both operands in summation must be Preconditioner.')

        # Check linear operator compatibility
        if not (precond1.lin_op is precond2.lin_op):
            raise PreconditionerError('Both operands must precondition the same linear operator.')

        # Check shape compatibility
        if precond1.shape != precond2.shape:
            raise PreconditionerError('Both operands in summation must have the same shape.')

        self.operands = (precond1, precond2)

        shape = precond1.shape
        dtype = numpy.find_common_type([precond1.dtype, precond2.dtype], [])
        lin_op = precond1.lin_op

        super().__init__(shape, dtype, lin_op)

    def _apply(self, X):
        return self.operands[0].dot(X) + self.operands[1].dot(X)

    def _matvec_cost(self):
        m, _ = self.operands[0].shape
        return self.operands[0].apply_cost + self.operands[1].apply_cost + m


class _ComposedPreconditioner(Preconditioner):
    """
    Abstract class for composition of preconditioner.
    """

    def __init__(self, precond1: Preconditioner, precond2: Preconditioner) -> None:
        # Sanitize the precond1 and precond2 attribute
        if not isinstance(precond1, Preconditioner) or not isinstance(precond2, Preconditioner):
            raise PreconditionerError('Both operands in composition must be Preconditioner.')

        # Check linear operator compatibility
        if not (precond1.lin_op is precond2.lin_op):
            raise PreconditionerError('Both operands must precondition the same linear operator.')

        # Check shape compatibility
        if precond1.shape != precond2.shape:
            raise PreconditionerError('Both operands in summation must have the same shape.')

        self.operands = (precond1, precond2)

        shape = precond1.shape
        dtype = numpy.find_common_type([precond1.dtype, precond2.dtype], [])
        lin_op = precond1.lin_op

        super().__init__(shape, dtype, lin_op)

    def _apply(self, X):
        Y = self.operands[0].dot(X)
        Z = X - self.lin_op.dot(Y)
        return Y + self.operands[1].dot(Z)

    def _matvec_cost(self):
        cost = self.operands[0].apply_cost + self.operands[1].apply_cost + self.lin_op.apply_cost
        cost = cost + 2 * self.shape[0]
        return cost


class IdentityPreconditioner(Preconditioner):
    """
    Abstract class for identity preconditioner.
    """

    def __init__(self, lin_op: LinearOperator) -> None:
        super().__init__(lin_op.shape, lin_op.dtype, lin_op)

        self.name = 'Identity'

    def _apply(self, x):
        return x

    def _matvec_cost(self):
        return 0.


class DiagonalPreconditioner(Preconditioner):
    """
    Abstract class for Diagonal (Jacobi) preconditioner.
    """

    def __init__(self, matrix_op: MatrixOperator) -> None:
        """
        Constructor of the Jacobi preconditioner.

        :param matrix_op: MatrixOperator to precondition.
        """
        n, _ = matrix_op.shape

        # Handle the different types for lin_op
        if scipy.sparse.issparse(matrix_op.matrix):
            diag = matrix_op.matrix.diagonal()
        else:
            diag = numpy.diag(matrix_op.matrix)

        diag[diag == 0] += 1

        self._diag = scipy.sparse.diags(1 / diag)

        super().__init__(matrix_op.shape, matrix_op.dtype, matrix_op)

        self.name = 'Jacobi'

    def _apply(self, x):
        return self._diag.dot(x)

    def _matvec_cost(self):
        return 2 * self.shape[0]


class SymmetricGaussSeidel(Preconditioner):
    """
    Abstract class for Symmetric Successive Over-Relaxation (SSOR) preconditioners. Formally, it has
    the following expression, L being the lower triangular part of A:

        M = (D + L) * D^(-1) * (D + L^T)

    """

    def __init__(self, matrix_op: MatrixOperator) -> None:
        """
        Constructor of the SSOR preconditioner.

        :param matrix_op: MatrixOperator to precondition.
        """

        # Sanitize matrix operator argument
        if not isinstance(matrix_op, MatrixOperator):
            raise PreconditionerError('Gauss-Seidel method needs a MatrixOperator object.')

        super().__init__(matrix_op.shape, matrix_op.dtype, matrix_op)

        # Prevent unresolved references
        self.lin_op = matrix_op

        self.name = 'Symmetric Gauss-Seidel'

    def _apply(self, x):
        x_0 = numpy.zeros_like(x)
        pyamg.relaxation.relaxation.gauss_seidel(self.lin_op.matrix, x_0, x, sweep='symmetric')
        return x_0

    def _matvec_cost(self):
        return self.lin_op.apply_cost


class AlgebraicMultiGrid(Preconditioner):
    """
    Abstract class for Algebraic Multi-grid (AMG) operators.
    """

    def __init__(self, matrix: MatrixOperator, heuristic: str = 'ruge_stuben'):

        # Sanitize heuristic argument
        if heuristic not in ['ruge_stuben', 'rs', 'smoothed_aggregated', 'sa', 'rootnode', 'rn']:
            raise PreconditionerError('AMG heuristic {} unknown.'.format(heuristic))

        # Setup multi-grid hierarchical structure with corresponding heuristic
        if heuristic in ['ruge_stuben', 'rs']:
            self.amg = pyamg.ruge_stuben_solver(matrix.matrix.tocsr(), max_coarse=1000)

        elif heuristic in ['smoothed_aggregated', 'sa']:
            self.amg = pyamg.smoothed_aggregation_solver(matrix.matrix.tocsr(), max_coarse=1000)

        elif heuristic in ['rootnode', 'rn']:
            self.amg = pyamg.rootnode_solver(matrix.matrix.tocsr(), max_coarse=1000)

        super().__init__(matrix.shape, matrix.dtype, matrix)

        self.name = 'AMG'

    def _matvec_cost(self):
        return 0

    def _apply(self, x):
        y = self.amg.solve(x, tol=1e-15, maxiter=1, cycle='F', accel=None)

        return numpy.atleast_2d(y).T


class CoarseGridCorrection(Preconditioner):
    """
    Abstract class for Coarse-Grid Correction preconditioner. Formally, for a given subspace S, the
    preconditioner is written as:

        Q = S * (S^T * A * S)^(-1) * S^T

    """

    def __init__(self, lin_op: LinearOperator, subspace: object, rank_tol: float = 1e-10) -> None:
        """
        Constructor of the Coarse-Grid Correction preconditioner.

        :param lin_op: MatrixOperator to precondition.
        :param subspace: Subspace on which the projection is made.
        :param rank_tol: Maximum ratio of extrema singular values above which the effective rank is
        decreased.
        """
        # Sanitize the subspace attribute
        self.sparse_subspace = scipy.sparse.isspmatrix(subspace)

        if not isinstance(subspace, (numpy.ndarray, numpy.matrix)) and not self.sparse_subspace:
            raise PreconditionerError('CoarseGridCorrection projection subspace should be either a '
                                      'numpy.ndarray or numpy.matrix or a sparse matrix.')

        if subspace.ndim != 2:
            raise PreconditionerError('CoarseGridCorrection projection subspace should be 2-D.')

        if subspace.shape[0] < subspace.shape[1]:
            subspace = subspace.T

        # Process QR factorization of the subspace when not of sparse format
        if not self.sparse_subspace:
            q, r = scipy.linalg.qr(subspace, mode='economic')
            s = scipy.linalg.svd(r, compute_uv=False)

            # Potentially decrease the effective rank of S
            rank = numpy.sum(s * (1 / s[0]) > rank_tol)
            self._subspace = q[:, :rank]
        else:
            self._subspace = subspace

        # Compute the reduced operator S^T * A *S
        self._reduced_lin_op = self._subspace.T.dot(lin_op.dot(self._subspace))

        if self.sparse_subspace:
            self._reduced_lin_op = self._reduced_lin_op.todense()

        # Cholesky decomposition of the reduced operator
        self._cho_factor = scipy.linalg.cho_factor(self._reduced_lin_op,
                                                   lower=True,
                                                   overwrite_a=True)

        dtype = numpy.find_common_type([lin_op.dtype, subspace.dtype], [])

        super().__init__(lin_op.shape, dtype, lin_op)

        self.name = 'Coarse-Grid'

        self.building_cost = self._get_building_cost()

    def _get_building_cost(self):
        n, k = self._subspace.shape
        cost = 2 * n * k**2 + 2 * k**3                          # R-SVD
        cost += 2 * k * self.lin_op.apply_cost + 2 * n * k**2   # A_m = S^T*A*S
        cost += 2 / 3 * k**3                                    # PLU
        return cost

    def _matvec_cost(self):
        _, k = self._subspace.shape
        return 4 * self._subspace.size + 2 * k**2

    def _apply(self, x):
        y = self._subspace.T.dot(x)
        scipy.linalg.cho_solve(self._cho_factor, y, overwrite_b=True)
        y = self._subspace.dot(y)
        return y


class LimitedMemoryPreconditioner(Preconditioner):
    """
    Abstract class for Limited Memory Preconditioner (LMP). Formally, for Q being the Coarse-Grid
    correction preconditioner defined with S, and first-level preconditioner M, it is written:

        H = (I - Q*A) * M * (I - A*Q) + Q

    """

    def __init__(self, lin_op: LinearOperator, subspace: object, M: Preconditioner = None) -> None:
        """
        Constructor of the Limited-Memory Preconditioner.

        :param lin_op: MatrixOperator to precondition.
        :param subspace: Subspace on which the projection is made.
        :param M: First-level preconditioner.
        """
        # Sanitize the first-level preconditioner attribute
        if M is not None and not isinstance(M, Preconditioner):
            raise PreconditionerError('LMP first-level preconditioner must be a Preconditioner.')

        M = IdentityPreconditioner(lin_op) if M is None else M
        Q = CoarseGridCorrection(lin_op, subspace)
        H = Q * M * Q

        self.cost = H.apply_cost
        self._apply = H.apply

        super().__init__(H.shape, H.dtype, H.lin_op)

        self.name = 'LMP'

    def _matvec_cost(self):
        return self.cost


class AlgebraicPreconditionerFactory(object):
    """
    Abstract class for Algebraic preconditioner factory. Allow to get different Preconditioner
    object depending on the name provided.
    """

    def __init__(self, lin_op: LinearOperator):
        self.lin_op = lin_op

    def get(self, preconditioner: str, *args, **kwargs) -> Preconditioner:
        """
        Generic method to generate subspaces from various distribution.

        :param preconditioner: Name of the preconditioner to build.
        :param args: Optional arguments for preconditioner.
        :param kwargs: Optional named arguments for preconditioner.
        """

        if preconditioner == 'identity':
            return IdentityPreconditioner(self.lin_op)

        elif preconditioner == 'jacobi':
            return DiagonalPreconditioner(self.lin_op)

        elif preconditioner == 'symmetric_gs':
            return SymmetricGaussSeidel(self.lin_op)

        elif preconditioner == 'amg':
            return AlgebraicMultiGrid(self.lin_op, *args, **kwargs)

        else:
            raise PreconditionerError('This preconditioner name in unrecognized.')
