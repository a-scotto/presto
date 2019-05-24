#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on April 17, 2019 at 13:15.

@author: a.scotto

Description:
"""

import numpy
import scipy.linalg

from linear_operator import LinearOperator, IdentityOperator, SelfAdjointMatrix


class PreconditionerError(Exception):
    """
    Exception raised when LinearOperator object encounters specific errors.
    """


class Preconditioner(LinearOperator):

    def __init__(self, shape, dtype, direct=True):

        super().__init__(shape, dtype)

        self.direct = direct

    def _matvec(self, x):
        return self._apply(x)

    def _apply(self, x):
        return None

    def apply(self, x):
        return self._apply(x)


class IdentityPreconditioner(Preconditioner):

    def __init__(self, n):

        if not isinstance(n, int) or n < 1:
            raise PreconditionerError('IdentityPreconditioner should have a positive integer size.')

        super().__init__((n, n), dtype=numpy.float64)

    def _apply(self, x):
        return x


class LimitedMemoryPreconditioner(Preconditioner):

    def __init__(self, A, S, M=None):

        if not isinstance(A, LinearOperator):
            raise PreconditionerError('LMP should be built from LinearOperator.')

        if not isinstance(S, numpy.ndarray) and not isinstance(S, numpy.matrix):
            raise PreconditionerError('LMP subspace should be numpy.ndarray or numpy.matrix.')

        if M is None:
            M = IdentityOperator(A.shape[0])

        dtype = numpy.find_common_type([A.dtype, S.dtype, M.dtype], [])

        self._S = S
        self._S_tilde = A.dot(self._S)

        A_tilde = S.T.dot(self._S_tilde)

        try:
            self._L = scipy.linalg.cholesky(A_tilde)
            self._U = self._L.T
            self._P = None

        except numpy.linalg.LinAlgError:
            self._P, self._L, self._U = scipy.linalg.lu(A_tilde)

        self._M = M

        super().__init__(A.shape, dtype, direct=True)

    def _matvec(self, x):
        return self._apply(x)

    def _apply(self, x):

        y = self._S.T.dot(x)

        if self._P is not None:
            y = scipy.linalg.solve(self._P, y)

        z = scipy.linalg.solve_triangular(self._L, y, lower=True)
        z = scipy.linalg.solve_triangular(self._U, z)

        f = self._M.dot(x) - self._M.dot(self._S_tilde.dot(z))

        g = self._S_tilde.T.dot(f)

        if self._P is not None:
            g = scipy.linalg.solve(self._P, g)

        h = scipy.linalg.solve_triangular(self._L, g, lower=True)
        h = scipy.linalg.solve_triangular(self._U, h)

        return f + self._S.dot(z - h)


if __name__ == "__main__":

    import numpy

    from utils import qr, norm

    size = 100

    A = numpy.random.rand(size, size)
    A = A * (A > 0.95)
    A = A.T + A + numpy.diag([i**1.5 for i in range(size)])

    S = numpy.random.rand(size, 4)
    X = S.shape[1] * numpy.random.rand(S.shape[1], S.shape[1])
    Z = S.dot(X)

    A = SelfAdjointMatrix(A)
    S = S
    Z = Z

    H1 = LimitedMemoryPreconditioner(A, S)
    H2 = LimitedMemoryPreconditioner(A, Z)

    b = numpy.random.rand(size, 1)

    print("Invariance of LMP under change of basis Z = SX | {:1.2e}"
          .format(norm(H1.dot(b) - H2.dot(b))))
    print("LMP identity HAS = S | {:1.2e}"
          .format(norm((H1.dot(A.dot(S)) - S))))

