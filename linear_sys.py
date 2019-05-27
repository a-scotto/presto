#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on April 25, 2019 at 11:02.

@author: a.scotto

Description:
"""

import numpy
import utils
import scipy.linalg

from linear_operator import LinearOperator, SelfAdjointMatrix
from preconditioner import Preconditioner, IdentityPreconditioner


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
                 x_sol=None):

        if not isinstance(lin_op, LinearOperator):
            raise LinearSystemError('A LinearSystem must be defined with a LinearOperator.')

        self.lin_op = lin_op

        if not isinstance(lhs, numpy.ndarray):
            raise LinearSystemError('LinearSystem left-hand side must be numpy.ndarray')

        if lhs.shape[0] != A.shape[1]:
            raise LinearSystemError('Linear map and left-hand side shapes do not match.')

        self.lhs, self.scale = self._normalize_lhs(lhs)
        self.x_sol = x_sol

        self.block = False if self.lhs.shape[1] == 1 else True
        self.shape = self.lin_op.shape
        self.dtype = numpy.find_common_type([self.lin_op.dtype, self.lhs.dtype], [])

    @staticmethod
    def _normalize_lhs(lhs):
        scale = numpy.linalg.norm(lhs, axis=0)
        lhs = lhs / scale
        return lhs, scale

    def get_residual(self, x):
        return self.lin_op.dot(x / self.scale) - self.lhs

    def __repr__(self):
        _repr = 'Linear system of shape {} with left-hand side of shape {}.'\
                .format(self.lin_op.shape, self.lhs.shape)
        return _repr


class _LinearSolver(object):
    """
    Abstract class to model linear systems.
    """

    def __init__(self,
                 lin_sys,
                 x_0=None,
                 M=None,
                 tol=1e-5,
                 maxiter=None):
        """
        Constructor of LinearSolver class. Instantiate a LinearSolver object.

        :param lin_sys:
        :param x_0:
        :param M:
        :param tol:
        :param maxiter:
        """

        # Sanitize the initialization of class attributes
        if not isinstance(lin_sys, LinearSystem):
            raise LinearSolverError('LinearSolver requires a LinearSystem.')

        self.lin_sys = lin_sys

        x_0 = numpy.zeros_like(lin_sys.lhs) if x_0 is None else x_0

        if x_0 is not None and not isinstance(x_0, numpy.ndarray):
            raise LinearSolverError('Initial guess x_0 must be a numpy.ndarray.')

        if x_0.shape != lin_sys.lhs.shape:
            raise LinearSolverError('Shapes of initial guess x_0 and left-hand side b mismatch.')

        self.x = x_0

        if M is None:
            M = IdentityPreconditioner(self.x.size)

        elif isinstance(M, list):
            for M_i in M:
                if not isinstance(M_i, Preconditioner):
                    raise LinearSolverError('Preconditioner must be of type Preconditioner.')
                if M_i.shape != lin_sys.shape:
                    raise LinearSolverError('Preconditioner shape do not match with LinearSystem.')

        elif isinstance(M, Preconditioner):
            pass

        else:
            raise LinearSystemError('Preconditioner must be None, Preconditioner, or list of '
                                    'Preconditioner.')

        self.M_i = M

        self.tol = tol
        self.maxiter = maxiter
        self.monitor = self._initialize()

    def _initialize(self):
        return None

    def run(self):
        raise LinearSolverError('Linear solvers must implement a run method.')


class _SolverMonitor(object):
    """
    Abstract class for LinearSolver output class. This class is meant to be used by LinearSolver
    object to store the different quantities involved in the LinearSolver run as much as the
    historic of the residues.
    """

    def __init__(self, history, auxiliaries=None, store=False):
        """

        :param history:
        :param auxiliaries:
        :param store:
        """

        # Sanitize the initialization of class attributes
        if isinstance(history, list) and not all(numpy.isscalar(item) for item in history):
                raise LinearSolverError('History quantity must be a scalar or list of scalars.')
        if not isinstance(history, list) and not numpy.isscalar(history):
            raise LinearSolverError('History quantity must be a scalar or list of scalars.')

        try:
            self._history = [[h_i] for h_i in history]
        except TypeError:
            self._history = [[history]]

        self.n_hist = len(self._history)

        auxiliaries = [] if auxiliaries is None else auxiliaries
        if not isinstance(auxiliaries, list):
            raise LinearSolverError('"auxiliaries" must be a list.')
        if not all(isinstance(a_i, numpy.ndarray) for a_i in auxiliaries):
                raise LinearSolverError('Auxiliary quantity must be a numpy.ndarray.')

        self._auxiliaries = auxiliaries
        self.n_aux = len(auxiliaries)

        if store and not isinstance(store, int) and not store >= 0:
            raise LinearSolverError('Argument store must be either "False" or positive integer.')

        if store:
            for i in range(len(self._auxiliaries)):
                self._auxiliaries[i] = [self._auxiliaries[i]]

        self._store = 0 if not store else store
        self.n_it = 0

    def update(self, history, auxiliaries):

        history = [history] if not isinstance(history, list) else history

        if len(auxiliaries) != self.n_aux and len(auxiliaries) != 0:
            raise LinearSolverError('Updating a different number of auxiliaries than expected.')

        for i in range(len(auxiliaries)):

            if not isinstance(auxiliaries[i], numpy.ndarray):
                raise LinearSolverError('Auxiliaries must be numpy.ndarray.')

            if not self._store:
                self._auxiliaries[i] = auxiliaries[i]
            else:
                if len(self._auxiliaries[i]) < self._store:
                    self._auxiliaries[i].append(auxiliaries[i])
                else:
                    del self._auxiliaries[i][0]
                    self._auxiliaries[i].append(auxiliaries[i])

        if len(history) != self.n_hist:
            print(len(history), self.n_hist)
            raise LinearSolverError('Updating a different number of history items than expected.')

        for i in range(len(history)):

            if not numpy.isscalar(history[i]):
                raise LinearSolverError('History items must be scalars.')

            self._history[i].append(history[i])

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

        if self.n_aux == 1:
            return ret[0]
        else:
            return ret.__iter__()

    def get_history(self):
        ret = []

        for i in range(self.n_hist):
            history = numpy.stack(self._history[i])
            ret.append(history.T)

        if self.n_hist == 1:
            return ret[0]
        else:
            return ret.__iter__()

    def report(self, solver_name):
        string = '{:8}: run of {:4} iteration(s) | Relative 2-norm residual = {:1.4e}'\
                 .format(solver_name, self.n_it, self._history[0][-1])
        return string


class _SolverResult(object):

    def __init__(self, x_opt, solver_monitor):

        if not isinstance(x_opt, numpy.ndarray) and not isinstance(x_opt, numpy.matrix):
            raise LinearSolverError('x_opt must be numpy.ndarray or numpy.matrix.')

        if not isinstance(solver_monitor, _SolverMonitor):
            raise LinearSolverError('_SolverResult must be defined from _SolverMonitor')

        self.x_opt = x_opt


class ConjugateGradient(_LinearSolver):

    def __init__(self, lin_sys, x_0=None, M=None, store=None, tol=1e-5):
        if not isinstance(lin_sys.lin_op, SelfAdjointMatrix) and not lin_sys.lin_op.def_pos:
            raise LinearSolverError('Conjugate Gradient only apply to s.d.p linear map.')

        if lin_sys.lin_op.shape[0] != lin_sys.lin_op.shape[1]:
            raise LinearSolverError('Conjugate Gradient only apply to square problems.')

        if lin_sys.block:
            raise LinearSolverError('Conjugate Gradient only apply to simple left-hand side.')

        self.store = store
        self.total_cost = 0

        tol = tol * numpy.linalg.norm(lin_sys.lhs)

        super().__init__(lin_sys, x_0, M, tol, lin_sys.shape[0])

        self.iteration_cost = self._iteration_cost()

        if not isinstance(self.M_i, Preconditioner):
            raise LinearSolverError('Conjugate Gradient can handle only one preconditioner.')

    def _initialize(self):
        op_size = self.lin_sys.lin_op.size
        n, _ = self.lin_sys.lhs.shape

        r = - self.lin_sys.get_residual(self.x)
        z = self.M_i.apply(r)
        p = numpy.copy(z)
        auxiliaries = [z, p, r]

        residue = utils.norm(r)
        cost = 2*op_size + n
        history = [residue, cost]

        self.total_cost += cost

        return _SolverMonitor(history, auxiliaries=auxiliaries, store=self.store)

    def _finalize(self):

        residues, cost = self.monitor.get_history()
        z, p, r = self.monitor.get_auxiliaries()

        output = {'report': self.monitor.report('CG'),
                  'x': self.x * self.lin_sys.scale,
                  'n_it': self.monitor.n_it,
                  'residues': residues,
                  'cost': cost,
                  'Z': z,
                  'P': p,
                  'R': r}

        return output

    def _iteration_cost(self):

        op_size = self.lin_sys.lin_op.size
        n, _ = self.lin_sys.lhs.shape
        total_cost = 0

        total_cost += 2 * op_size   # q_k
        total_cost += 4*n + 1       # alpha
        total_cost += 2*n           # self.x
        total_cost += 2*n           # r_k
        total_cost += 2*n           # residue
        total_cost += 0             # z_k
        total_cost += 4*n + 1       # beta
        total_cost += 2*n           # p_k

        return total_cost

    def run(self, verbose=False):
        for k in range(self.maxiter):

            z, p, r = self.monitor.get_previous()

            q_k = self.lin_sys.lin_op.dot(p)

            alpha = z.T.dot(r) / p.T.dot(q_k)
            self.x += alpha * p
            r_k = r - alpha * q_k

            residue = utils.norm(r_k)
            self.total_cost += self.iteration_cost

            if residue < self.tol:
                self.monitor.update([residue, self.total_cost], [])
                break

            z_k = self.M_i.apply(r_k)
            beta = z_k.T.dot(r_k) / z.T.dot(r)
            p_k = z_k + beta * p

            self.monitor.update([residue, self.total_cost], [z_k, p_k, r_k])

        output = self._finalize()

        return output


class BlockConjugateGradient(_LinearSolver):

    def __init__(self, lin_sys, x_0=None, M=None, store=None, tol=1e-5, rank_tol=1e-5):
        if not isinstance(lin_sys.lin_op, SelfAdjointMatrix) and not lin_sys.lin_op.def_pos:
            raise LinearSolverError('Block Conjugate Gradient only apply to s.d.p linear map.')

        if lin_sys.lin_op.shape[0] != lin_sys.lin_op.shape[1]:
            raise LinearSolverError('Block Conjugate Gradient only apply to square problems.')

        if not lin_sys.block:
            raise LinearSolverError('Block Conjugate Gradient only apply to block left-hand side.')

        self.store = store
        self.total_cost = 0
        self.rank_tol = rank_tol

        tol = tol * numpy.linalg.norm(lin_sys.lhs)

        super().__init__(lin_sys, x_0, M, tol, lin_sys.shape[0])

        if not isinstance(self.M_i, Preconditioner):
            raise LinearSolverError('Block Conjugate Gradient can handle only one preconditioner.')

    def _initialize(self):
        op_size = self.lin_sys.lin_op.size
        n, k = self.lin_sys.lhs.shape

        R = - self.lin_sys.get_residual(self.x)
        r = scipy.linalg.qr(R, mode='r')[0]
        s = scipy.linalg.svd(r, compute_uv=False)
        rank = numpy.sum(s * (1 / s[0]) > self.rank_tol)

        Z = self.M_i.apply(R[:, :rank])
        P = numpy.copy(Z)
        auxiliaries = [Z, P, R]

        res = utils.norm(R)
        cost = 2*op_size*k + n*k + 2 * self.lin_sys.lhs.size
        history = [res, cost, rank]

        self.total_cost += cost

        return _SolverMonitor(history, auxiliaries=auxiliaries, store=self.store)

    def _finalize(self):

        residues, cost, rank = self.monitor.get_history()
        Z, P, R = self.monitor.get_auxiliaries()

        output = {'report': self.monitor.report('BlockCG'),
                  'x': self.x * self.lin_sys.scale,
                  'n_it': self.monitor.n_it,
                  'residues': residues,
                  'cost': cost,
                  'rank': rank,
                  'Z': Z,
                  'P': P,
                  'R': R}

        return output

    def _iteration_cost(self, r):

        op_size = self.lin_sys.lin_op.size
        n, k = self.lin_sys.lhs.shape
        total_cost = 0

        total_cost += 2 * op_size * r       # Q_k
        total_cost += 2 * n * r**2          # delta
        total_cost += 4*k*r**2 + 22*k**3    # SVD of delta
        total_cost += r                     # s_inv
        total_cost += 2*(r*n*k + 3*r*r*k)   # alpha
        total_cost += n*k + 2*r*n*k         # self.x
        total_cost += n*k + 2*r*n*k         # R_k
        total_cost += 2*n*r**2              # residue
        total_cost += 0                     # Z_k
        total_cost += 2*(r*n*r + 3*r*r*k)   # beta
        total_cost += 2*n*r*r + n*r         # P_k

        return total_cost

    def run(self, verbose=False):

        for k in range(self.maxiter):

            Z, P, R = self.monitor.get_previous()

            Q_k = self.lin_sys.lin_op.dot(P)

            delta = P.T.dot(Q_k)
            u, s, v = numpy.linalg.svd(delta)
            rank = numpy.sum(s * (1 / s[0]) > self.rank_tol)

            s_inv = numpy.zeros_like(s)
            s_inv[:rank] = 1 / s[:rank]

            alpha = v.T @ numpy.diag(s_inv) @ u.T @ Z.T.dot(R)
            self.x += P.dot(alpha)

            R_k = R - Q_k.dot(alpha)

            residue = utils.norm(R_k[:, :rank])
            self.total_cost += self._iteration_cost(rank)

            if residue < self.tol:
                self.monitor.update([residue, self.total_cost, rank], [])
                break

            Z_k = self.M_i.apply(R_k[:, :rank])

            beta = - v.T @ numpy.diag(s_inv) @ u.T @ Q_k.T.dot(Z_k)
            P_k = Z_k + P.dot(beta)

            self.monitor.update([residue, self.total_cost, rank], [Z_k, P_k, R_k])

        output = self._finalize()

        return output


if __name__ == '__main__':

    from matplotlib import pyplot
    from problems.loader import load_problem, print_problem
    # from preconditioner import LimitedMemoryPreconditioner

    file_name = 'Kuu'

    problem = load_problem(file_name)
    print_problem(problem)

    A = problem['operator']
    B = numpy.random.randn(A.shape[0], 10)
    b = numpy.sum(B, axis=1).reshape(A.shape[0], 1)

    linSys = LinearSystem(SelfAdjointMatrix(A, def_pos=True), b)
    blockLinSys = LinearSystem(SelfAdjointMatrix(A, def_pos=True), B)

    cg_output = ConjugateGradient(linSys, tol=1e-6, store=0).run()
    res1 = numpy.linalg.norm(A.dot(cg_output['x']) - b)

    bcg_output = BlockConjugateGradient(blockLinSys, tol=1e-6, store=0).run()
    res2 = numpy.linalg.norm(A.dot(bcg_output['x']) - B)

    print(cg_output['report'])
    print(bcg_output['report'])

    pyplot.figure()
    pyplot.semilogy(cg_output['cost'], cg_output['residues'])
    pyplot.semilogy(bcg_output['cost'] / B.shape[1], bcg_output['residues'])
    pyplot.xlabel('normalized FLOPs')
    pyplot.ylabel('log relative residual')
    pyplot.legend(['CG', 'BCG'])

    # S = cg_output['p']
    # H = LimitedMemoryPreconditioner(SelfAdjointMatrix(A, def_pos=True, sparse_format='csc'), S)
    #
    # t = time()
    # cg_output = ConjugateGradient(linSys, M=H, tol=1e-5, store=40).run()
    # t1 = time() - t
    # res1 = numpy.linalg.norm(A.dot(cg_output['x']) - b)
    #
    # t = time()
    # bcg_output = BlockConjugateGradient(blockLinSys, M=H, tol=1e-5, store=0).run()
    # t2 = time() - t
    # res2 = numpy.linalg.norm(A.dot(bcg_output['x']) - B)
    #
    # print('                                      time     |    ||r||')
    # print('Algorithm ConjugateGradient:       {:1.4e}  |  {:1.4e}  |  {}'
    #       .format(t1, res1, cg_output['n_it']))
    # print('Algorithm BlockConjugateGradient : {:1.4e}  |  {:1.4e}  |  {}'
    #       .format(t2, res2, bcg_output['n_it']))
    #
    # pyplot.figure()
    # pyplot.semilogy([i for i in range(cg_output['n_it'] + 1)], cg_output['residues'])
    # pyplot.semilogy([i for i in range(cg_output['n_it'] + 1)], bcg_output['residues'])
    # pyplot.legend(['CG', 'BCG'])

    pyplot.figure()
    pyplot.plot(bcg_output['rank'])
    pyplot.show()
