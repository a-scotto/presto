#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on April 25, 2019 at 11:02.

@author: a.scotto

Description:
"""

import numpy
import utils

from linear_map import LinearMap
from matrix_operator import MatrixOperator, SelfAdjointMap


class LinearSystemError(Exception):
    """
    Exception raised when LinearSystem object encounters specific errors.
    """


class LinearSolverError(Exception):
    """
    Exception raised when LinearSolver object encounters specific errors.
    """


class LinearSystem(object):

    def __init__(self, A, b,
                 symmetric=False,
                 definite_positive=False):

        self.symmetric = symmetric
        self.definite_positive = definite_positive

        if self.symmetric:
            self.A = SelfAdjointMap(A)
        else:
            self.A = MatrixOperator(A)

        if not isinstance(b, numpy.ndarray):
            raise LinearSystemError('Linear system left-hand side must be numpy.ndarray')

        self.b = b

        self.shape = A.shape
        self.dtype = A.dtype

    def get_residual(self, x):
        return self.A.dot(x) - self.b

    def left_precond(self, M):
        self
        return None

    def right_precond(self, M):
        self
        return None


class LinearSolver(object):

    def __init__(self, lin_sys, x_0=None, M=[], ip_B=None, tol=1e-5, maxiter=None):

        if not isinstance(lin_sys, LinearSystem):
            raise LinearSolverError('LinearSolver requires a LinearSystem.')

        if not isinstance(x_0, numpy.ndarray):
            raise LinearSolverError('Initial guess x_0 must be a numpy.ndarray.')

        if not isinstance(M, list):
            raise LinearSolverError('The preconditioners must be provided as list.')

        for M_i in M:
            if not isinstance(M_i, LinearMap):
                raise LinearSolverError('Preconditioners must be LinearMap.')
            if M_i.shape != lin_sys.shape:
                raise LinearSolverError('Preconditioners shape do not match.')

        self.lin_sys = lin_sys
        self.x = x_0
        self.M_i = M
        self.ip_B = ip_B

        self.tol = tol
        self.maxiter = maxiter

        self.auxiliaries = self._initialize()
        self.residue = utils.norm(self.lin_sys.get_residual(self.x), ip_B=ip_B)
        self.result = LinearSolverResult(self)

    def _initialize(self):
        return []

    def run(self):
        return None


class LinearSolverResult(object):

    def __init__(self, lin_solv, store=False):

        if not isinstance(lin_solv, LinearSolver):
            raise LinearSolverError('LinearSolverResult object must be defined from a '
                                    'LinearSolver.')

        self.x_opt = lin_solv.x

        if store:
            self.auxiliaries = []

            for a_i in lin_solv.auxiliaries:
                if not isinstance(a_i, list):
                    self.auxiliaries.append([a_i])
                else:
                    self.auxiliaries.append(a_i)
        else:
            self.auxiliaries = lin_solv.auxiliaries

        self.store = store
        self.residue = lin_solv.residue
        self.n_it = 0

    def update(self, x, auxiliaries=[]):

        self.x_opt = x

        if len(auxiliaries) != 0:
            if len(auxiliaries) != len(self.auxiliaries):
                raise LinearSolverError('Updating a different number of auxiliaries than expected.')

            for i in range(len(auxiliaries)):

                if not isinstance(auxiliaries[i], numpy.ndarray):
                    raise LinearSolverError('Auxiliaries must be numpy.ndarray.')

                if auxiliaries[i].shape != self.auxiliaries[i].shape:
                    raise LinearSolverError('Auxiliary {} shape do not match previous entry.'
                                            .format(i))

                if self.store:
                    self.auxiliaries[i].append(auxiliaries[i])
                else:
                    self.auxiliaries[i] = auxiliaries[i]

        self.n_it += 1

    def draw(self):

        if not self.store:
            raise LinearSolverError('No data stored hence impossibility to draw.')


class ConjugateGradient(LinearSolver):

    def __init__(self, lin_sys, x_0=None, tol=1e-5):

        if not lin_sys.symmetric and not lin_sys.definite_positive:
            raise LinearSolverError('Impossible to apply Conjugate Gradient to linear operator '
                                    'not s.d.p.')

        self.x = numpy.zeros_like(lin_sys.b) if x_0 is None else x_0

        self.maxiter = self.x.shape[0]

        super().__init__(lin_sys, self.x, [], lin_sys.A, tol, self.maxiter)

    def _initialize(self):

        r = self.lin_sys.get_residual(self.x)
        p = r.copy()

        return [p, r]

    def run(self, verbose=False):

        for k in range(self.maxiter * 5):

            p, r = self.result.auxiliaries

            q_k = self.lin_sys.A.dot(p)

            alpha = r.T.dot(r) / p.T.dot(q_k)
            x_k = self.x + alpha * p
            r_k = r - alpha * q_k

            self.residue = utils.norm(r_k, ip_B=self.ip_B)

            if self.residue < self.tol:
                break

            beta = r_k.T.dot(r_k) / r.T.dot(r)
            p_k = r_k + beta * p

            self.result.update(x_k, [p_k, r_k])


if __name__ == '__main__':

    A = 5 * numpy.random.rand(5, 5)
    b = numpy.random.rand(5, 1)

    l = LinearSystem(A.dot(A.T), b, symmetric=True, definite_positive=True)

    cg = ConjugateGradient(l)

    cg.run()

    print(cg.residue)
