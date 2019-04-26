#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on April 25, 2019 at 11:02.

@author: a.scotto

Description:
"""

import numpy
import utils

from matplotlib import pyplot
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
    """
    Abstract class to model linear systems.
    """

    def __init__(self,
                 lin_op,
                 lhs,
                 symmetric=False,
                 definite_positive=False):

        self.symmetric = symmetric
        self.definite_positive = definite_positive

        if self.symmetric:
            self.A = SelfAdjointMap(lin_op)
        else:
            self.A = MatrixOperator(lin_op)

        # Verify consistency between the linear operator and the left-hand side shape
        if not isinstance(lhs, numpy.ndarray):
            raise LinearSystemError('LinearSystem left-hand side must be numpy.ndarray')

        if lhs.shape[0] != A.shape[1]:
            raise LinearSystemError('Linear map and left-hand side shapes do not match.')

        self.b = lhs

        self.block = False if self.b.shape[1] == 1 else True

        self.shape = self.A.shape
        self.dtype = numpy.find_common_type([self.A.dtype, self.b.dtype], [])

    def get_residual(self, x):
        return self.A.dot(x) - self.b

    def __repr__(self):
        _repr = 'Linear system of shape {} with left-hand side of shape {}.'\
                .format(self.A.shape, self.b.shape)
        return _repr


class LinearSolver(object):
    """
    Abstract class to model linear systems.
    """

    def __init__(self,
                 lin_sys,
                 x_0=None,
                 M=None,
                 ip_B=None,
                 tol=1e-5,
                 maxiter=None):
        """
        Constructor of LinearSolver class. Instantiate a LinearSolver object.

        :param lin_sys:
        :param x_0:
        :param M:
        :param ip_B:
        :param tol:
        :param maxiter:
        """

        # Sanitize the initialization of class attributes
        if not isinstance(lin_sys, LinearSystem):
            raise LinearSolverError('LinearSolver requires a LinearSystem.')

        self.lin_sys = lin_sys

        x_0 = numpy.zeros_like(lin_sys.b) if x_0 is None else x_0

        if x_0 is not None and not isinstance(x_0, numpy.ndarray):
            raise LinearSolverError('Initial guess x_0 must be a numpy.ndarray.')

        if x_0.shape != lin_sys.b.shape:
            raise LinearSolverError('Shapes of initial guess x_0 and left-hand side b mismatch.')

        self.x = x_0

        M = [] if M is None else M

        if not isinstance(M, list):
            raise LinearSolverError('Non preconditioners must be provided as list.')

        for M_i in M:
            if not isinstance(M_i, LinearMap):
                raise LinearSolverError('Preconditioners must be LinearMap.')
            if M_i.shape != lin_sys.shape:
                raise LinearSolverError('Preconditioners shape do not match.')

        self.M_i = M

        if ip_B is not None and not (isinstance(ip_B, numpy.ndarray) or
                                     isinstance(ip_B, SelfAdjointMap)):
            raise LinearSystemError('Internal inner_product must be either SelfAdjointMap or '
                                    'numpy.ndarray.')

        if ip_B is not None and ip_B.shape != self.lin_sys.A.shape:
            raise LinearSystemError('Internal inner product shape do not match LinearSystem one.')

        self.ip_B = ip_B

        self.tol = tol
        self.maxiter = maxiter
        self.output = self._initialize()

    def _initialize(self):

        residue = utils.norm(self.lin_sys.get_residual(self.x), ip_B=self.ip_B)

        return SolverOutput(residue, auxiliaries=[])

    def run(self):
        return None


class SolverOutput(object):
    """
    Abstract class for LinearSolver output class. This class is meant to be used by LinearSolver
    object to store the different quantities involved in the LinearSolver run as much as the
    historic of the residues.
    """

    def __init__(self, residue, auxiliaries=None, store=False):
        """

        :param residue:
        :param auxiliaries:
        :param store:
        """

        # Sanitize the initialization of class attributes
        auxiliaries = [] if auxiliaries is None else auxiliaries

        if not isinstance(auxiliaries, list):
            raise LinearSolverError('"auxiliaries" must be a list.')

        for a_i in auxiliaries:
            if not isinstance(a_i, numpy.ndarray):
                    raise LinearSolverError('Auxiliary quantity must be a numpy.ndarray.')

        self._auxiliaries = auxiliaries
        self.n_aux = len(auxiliaries)

        if not isinstance(residue, float) and residue <= 0:
            raise LinearSolverError('Residue must be a non-negative float.')

        self.residue = residue

        if store and not isinstance(store, int) and not store >= 0:
            raise LinearSolverError('store must be either "False" or positive integer.')

        if store:
            for i in range(len(self._auxiliaries)):
                self._auxiliaries[i] = [self._auxiliaries[i]]

        self._store = 0 if not store else store
        self.history = [self.residue]
        self.n_it = 0

    def update(self, residue, auxiliaries):
        self.residue = residue

        previous_aux = self.get_previous()

        if len(auxiliaries) != 0:
            if len(auxiliaries) != self.n_aux:
                raise LinearSolverError('Updating a different number of auxiliaries than expected.')

            for i in range(len(auxiliaries)):

                if not isinstance(auxiliaries[i], numpy.ndarray):
                    raise LinearSolverError('Auxiliaries must be numpy.ndarray.')

                if auxiliaries[i].shape != previous_aux[i].shape:
                    raise LinearSolverError('Auxiliary {} shape do not match previous entry.'
                                            .format(i))

                if not self._store:
                    self._auxiliaries[i] = auxiliaries[i]
                else:
                    if len(self._auxiliaries) < self._store:
                        self._auxiliaries[i].append(auxiliaries[i])
                    else:
                        del self._auxiliaries[i][0]
                        self._auxiliaries[i].append(auxiliaries[i])

        self.history.append(self.residue)
        self.n_it += 1

    def get_previous(self, index=-1):
        if self._store:
            previous_auxiliary = []
            for aux_i in self._auxiliaries:
                previous_auxiliary.append(aux_i[index])
        else:
            previous_auxiliary = self._auxiliaries

        return previous_auxiliary

    def get_auxiliaries(self):
        ret = []
        if not self._store:
            return self._auxiliaries

        for i in range(self.n_aux):
            aux_history = numpy.stack(self._auxiliaries[i])
            ret.append(aux_history.T)

        return ret

    def convergence_history(self):
        pyplot.semilogy(self.history)
        pyplot.xlim(0, self.n_it)
        pyplot.show()

    def __repr__(self):
        string = 'Run of {} iteration(s) | Final residual = {:1.2e}'\
                 .format(self.n_it, self.residue)
        return string


class ConjugateGradient(LinearSolver):

    def __init__(self, lin_sys, x_0=None, store=None, tol=1e-5):

        if not lin_sys.symmetric and not lin_sys.definite_positive:
            raise LinearSolverError('Conjugate Gradient only apply to s.d.p linear map.')

        if lin_sys.block:
            raise LinearSolverError('Conjugate Gradient only apply to simple left-hand side.')

        if lin_sys.A.shape[0] != lin_sys.A.shape[1]:
            raise LinearSolverError('Conjugate Gradient only apply to square problems.')

        self.store = store

        super().__init__(lin_sys, x_0, [], lin_sys.A, tol, lin_sys.shape[0])

    def _initialize(self):
        r = - self.lin_sys.get_residual(self.x)
        p = r.copy()

        auxiliaries = [p, r]
        residue = utils.norm(r, ip_B=self.ip_B)

        return SolverOutput(residue, auxiliaries=auxiliaries, store=self.store)

    def run(self, verbose=False):
        for k in range(self.maxiter):

            p, r = self.output.get_previous()

            q_k = self.lin_sys.A.dot(p)

            alpha = r.T.dot(r) / p.T.dot(q_k)
            self.x += alpha * p
            r_k = r - alpha * q_k

            residue = utils.norm(r_k, ip_B=self.ip_B)

            if residue < self.tol:
                self.output.update(residue, [])
                break

            beta = r_k.T.dot(r_k) / r.T.dot(r)
            p_k = r_k + beta * p

            self.output.update(residue, [p_k, r_k])

        return self.output


if __name__ == '__main__':

    from scipy.sparse.linalg import cg

    size = 5000

    A = numpy.random.rand(size, size)
    A = A + A.T + numpy.diag([(i + 1) * size for i in range(size)])
    b = numpy.ones((size, 1))

    linSys = LinearSystem(A, b, symmetric=True, definite_positive=True)

    cg_own = ConjugateGradient(linSys, tol=1e-15)

    from time import time

    t = time()
    x_opt, _ = cg(A, b, tol=1e-15, maxiter=size)
    print(time() - t)

    t = time()
    output = cg_own.run()
    print(time() - t)

    print(output)
    print(numpy.linalg.norm(A.dot(cg_own.x) - b))
    print(numpy.linalg.norm(A.dot(x_opt) - b))

