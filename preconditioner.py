#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on April 17, 2019 at 13:15.

@author: a.scotto

Description:
"""

import numpy

from linear_map import LinearMap, LinearMapError, IdentityMap, SolvedLinearMap


class LimitedMemoryPreconditioner(LinearMap):

    def __init__(self, A, S, M=None):

        if not isinstance(A, LinearMap) or not isinstance(S, LinearMap):
            raise LinearMapError('LMP is built from LinearMap.')

        if M is None:
            M = IdentityMap(A.shape)

        dtype = numpy.find_common_type([A.dtype, S.dtype, M.dtype], [])

        self._S_tilde = A * S
        self._S = S
        self._M = M

        super().__init__(A.shape, dtype, self._dot)

    def _dot(self, X):

        Y = self._S.dot_adj(X)

        A_tilde = SolvedLinearMap(self._S.adjoint() * self._S_tilde)

        Z = A_tilde.dot(Y)

        F = self._M.dot(X) - (self._M * self._S_tilde).dot(Z)

        G = self._S_tilde.dot_adj(F)

        H = A_tilde.dot(G)

        return F + self._S.dot(Z - H)


if __name__ == "__main__":

    from numpy.linalg import norm
    from linear_map import Matrix

    size = 100

    A = size * numpy.random.rand(size, size)

    S = numpy.random.rand(size, size // 20 + 1)
    X = S.shape[1] * numpy.random.rand(S.shape[1], S.shape[1])
    Z = S.dot(X)

    A = Matrix(A)
    S = Matrix(S)
    Z = Matrix(Z)

    H1 = LimitedMemoryPreconditioner(A, S)
    H2 = LimitedMemoryPreconditioner(A, Z)

    b = numpy.random.rand(size, 1)

    print("Invariance of LMP under change of basis Z = SX | {:1.2e}"
          .format(norm(H1.dot(b) - H2.dot(b))))
    print("LMP identity HAS = S | {:1.2e}"
          .format(norm((H1 * A * S - S).canonical_repr())))

