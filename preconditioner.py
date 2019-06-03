#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on April 17, 2019 at 13:15.

@author: a.scotto

Description:
"""

import numpy
import scipy.linalg

from linear_operator import LinearOperator


class PreconditionerError(Exception):
    """
    Exception raised when LinearOperator object encounters specific errors.
    """


class Preconditioner(LinearOperator):

    def __init__(self, shape, dtype, apply_cost, lin_op, approx_inverse=True):

        if not isinstance(lin_op, LinearOperator):
            raise PreconditionerError('Preconditioner should be built from LinearOperator.')

        super().__init__(shape, dtype, apply_cost)

        self.lin_op = lin_op
        self.approx_inverse = approx_inverse

    def _apply(self, x):
        return None

    def _matvec(self, x):
        return self._apply(x)

    def apply(self, x):
        return self._apply(x)

    def __add__(self, B):
        return _SummedPreconditioner(self, B)

    def __mul__(self, B):
        return _ComposedPreconditioner(self, B)


class _SummedPreconditioner(Preconditioner):

    def __init__(self, A, B):

        if not isinstance(A, Preconditioner) or not isinstance(B, Preconditioner):
            raise PreconditionerError('Both operands in summation must be Preconditioner.')

        if not (A.lin_op is B.lin_op):
            raise PreconditionerError('Both operands must precondition the same linear operator.')

        if A.shape != B.shape:
            raise PreconditionerError('Both operands in summation must have the same shape.')

        if A.approx_inverse != B.approx_inverse:
            raise PreconditionerError('PBoth operands must be of same type.')

        self.operands = (A, B)

        shape = A.shape
        dtype = numpy.find_common_type([A.dtype, B.dtype], [])
        lin_op = A.lin_op
        apply_cost = A.apply_cost + B.apply_cost

        super().__init__(shape, dtype, apply_cost, lin_op)

    def _apply(self, X):
        return self.operands[0].dot(X) + self.operands[1].dot(X)


class _ComposedPreconditioner(Preconditioner):

    def __init__(self, A, B):

        if not isinstance(A, Preconditioner) or not isinstance(B, Preconditioner):
            raise PreconditionerError('Both operands must be Preconditioner.')

        if not (A.lin_op is B.lin_op):
            raise PreconditionerError('Both operands must precondition the same linear operator.')

        if A.shape != B.shape:
            raise PreconditionerError('Shape of operands do not match for composition.')

        if A.approx_inverse != B.approx_inverse:
            raise PreconditionerError('PBoth operands must be of same type.')

        self.operands = (A, B)

        shape = A.shape
        dtype = numpy.find_common_type([A.dtype, B.dtype], [])
        lin_op = A.lin_op
        apply_cost = A.apply_cost + 2 * B.apply_cost + lin_op.apply_cost + 2*shape[0]

        super().__init__(shape, dtype, apply_cost, lin_op)

    def _apply(self, X):
        Y = self.operands[0].dot(X)
        Z = self.operands[1].dot(X)

        return Y + Z - self.operands[1].dot(self.lin_op.dot(Y))
    

class IdentityPreconditioner(Preconditioner):

    def __init__(self, lin_op):

        super().__init__(lin_op.shape, lin_op.dtype, 0, lin_op)

    def _apply(self, x):
        return x


class CoarseGridCorrection(Preconditioner):

    def __init__(self, lin_op, subspace):

        if not isinstance(subspace, (numpy.ndarray, numpy.matrix)):
            raise PreconditionerError('CoarseGridCorrection projection subspace should be either a'
                                      'numpy.ndarray or numpy.matrix.')

        if subspace.ndim != 2:
            raise PreconditionerError('CoarseGridCorrection projection subspace should be 2-D.')

        if subspace.shape[0] < subspace.shape[1]:
            subspace = subspace.T

        n, k = subspace.shape

        q, r = scipy.linalg.qr(subspace, mode='economic')
        s = scipy.linalg.svd(r, compute_uv=False)
        rank = numpy.sum(s * (1 / s[0]) > 1e-6)

        self._subspace = q[:, :rank]
        self._reduced_lin_op = self._subspace.T.dot(lin_op.dot(self._subspace))

        try:
            L = scipy.linalg.cholesky(self._reduced_lin_op, lower=True)
            self._l = L
            self._u = L.T
            self._p = None

        except numpy.linalg.LinAlgError:
            P, L, U = scipy.linalg.lu(self._reduced_lin_op)
            self._l = L
            self._u = U
            self._p = P

        self.building_cost = self._get_building_cost(lin_op.apply_cost, n, k, rank)
        self.size = self._matvec_cost()

        dtype = numpy.find_common_type([lin_op.dtype, subspace.dtype], [])

        super().__init__(lin_op.shape, dtype, self._matvec_cost(), lin_op, approx_inverse=True)

    @staticmethod
    def _get_building_cost(lin_op_size, n, k, rank):
        cost = 2*n*k**2 + 2*k**3            # R-SVD
        cost += 2*k*lin_op_size + 2*n*k**2  # A_m = S^T*A*S
        cost += 2/3*rank**3                 # PLU

        return cost

    def _matvec_cost(self):
        cost = 2*self._subspace.size            # S^T * x
        cost += 2*self._subspace.shape[1]**2    # P.T * x
        cost += 2*self._subspace.shape[1]**2    # L^-1 * x
        cost += 2*self._subspace.shape[1]**2    # U^-1 * x
        cost = 2*self._subspace.size            # S * x
        return cost

    def _apply(self, x):

        y = self._subspace.T.dot(x)

        if self._p is not None:
            y = self._p.T.dot(y)

        y = scipy.linalg.solve_triangular(self._l, y, lower=True)
        y = scipy.linalg.solve_triangular(self._u, y)

        y = self._subspace.dot(y)

        return y


class LimitedMemoryPreconditioner(Preconditioner):

    def __init__(self, lin_op, subspace, M=None):

        if M is not None and not isinstance(M, Preconditioner):
            raise PreconditionerError('LMP first level preconditioner must be a Preconditioner.')

        M = IdentityPreconditioner(lin_op) if M is None else M
        Q = CoarseGridCorrection(lin_op, subspace)

        dtype = numpy.find_common_type([Q.dtype, M.dtype], [])
        apply_cost = (Q * M * Q).apply_cost

        super().__init__(lin_op.shape, dtype, apply_cost, lin_op)

        self._apply = (Q * M * Q).apply


if __name__ == "__main__":

    import scipy.sparse

    from utils import norm
    from linear_operator import SelfAdjointMatrix

    size = 200
    sub_size = 50

    A_ = scipy.sparse.rand(size, size, density=1e-3)
    A_ = A_.T + A_ + scipy.sparse.diags([i**1.125 for i in range(size)])
    A_ = SelfAdjointMatrix(A_)

    S = numpy.random.rand(size, sub_size)

    H = LimitedMemoryPreconditioner(A_, S)
    print(H.apply_cost)
    print(norm(H.dot(A_.dot(S)) - S))
