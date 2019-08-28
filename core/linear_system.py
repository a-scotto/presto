#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on April 25, 2019 at 11:02.

@author: a.scotto

Description:
"""

import numpy
import scipy.linalg

from core.linear_operator import LinearOperator, SelfAdjointMatrix
from core.preconditioner import Preconditioner, IdentityPreconditioner


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

        if lhs.shape[0] != lin_op.shape[1]:
            raise LinearSystemError('Linear map and left-hand side shapes do not match.')

        self.lhs, self.scaling = self._normalize_lhs(lhs)
        self.x_sol = x_sol

        self.block = False if self.lhs.shape[1] == 1 else True
        self.shape = self.lin_op.shape
        self.dtype = numpy.find_common_type([self.lin_op.dtype, self.lhs.dtype], [])

    @staticmethod
    def _normalize_lhs(lhs):
        scaling = numpy.linalg.norm(lhs, axis=0)
        lhs = lhs / scaling
        return lhs, scaling

    def get_residual(self, x):
        return self.lin_op.dot(x) - self.lhs * self.scaling

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

        self.x_0 = numpy.copy(x_0)
        self.x = numpy.zeros_like(x_0)

        if M is None:
            M = IdentityPreconditioner(lin_sys.lin_op)

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
        raise LinearSolverError('Linear solvers must implement a _initialize method.')

    def _finalize(self):
        raise LinearSolverError('Linear solvers must implement a _finalize method.')

    def run(self):
        raise LinearSolverError('Linear solvers must implement a run method.')


class _SolverMonitor(object):
    """
    Abstract class for LinearSolver output class. This class is meant to be used by LinearSolver
    object to store the different quantities involved in the LinearSolver run as much as the
    historic of the residues.
    """

    def __init__(self, history, auxiliaries=None, buffer=False):
        """

        :param history:
        :param auxiliaries:
        :param buffer:
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

        if buffer and not isinstance(buffer, int) and not buffer >= 0:
            raise LinearSolverError('Argument buffer must be either "False" or positive integer.')

        if buffer:
            for i in range(len(self._auxiliaries)):
                self._auxiliaries[i] = [self._auxiliaries[i]]

        self._buffer = 0 if not buffer else buffer
        self.n_it = 0

    def update(self, history, auxiliaries):

        history = [history] if not isinstance(history, list) else history

        if len(auxiliaries) != self.n_aux and len(auxiliaries) != 0:
            raise LinearSolverError('Updating a different number of auxiliaries than expected.')

        for i in range(len(auxiliaries)):

            if not isinstance(auxiliaries[i], numpy.ndarray):
                raise LinearSolverError('Auxiliaries must be numpy.ndarray.')

            if not self._buffer:
                self._auxiliaries[i] = auxiliaries[i]
            else:
                if len(self._auxiliaries[i]) < self._buffer:
                    self._auxiliaries[i].append(auxiliaries[i])
                else:
                    del self._auxiliaries[i][0]
                    self._auxiliaries[i].append(auxiliaries[i])

        if len(history) != self.n_hist:
            raise LinearSolverError('Updating a different number of history items than expected.')

        for i in range(len(history)):

            if not numpy.isscalar(history[i]):
                raise LinearSolverError('History items must be scalars.')

            self._history[i].append(history[i])

        self.n_it += 1

    def get_previous(self, index=-1):
        if self._buffer:
            previous_auxiliary = []
            for aux_i in self._auxiliaries:
                previous_auxiliary.append(aux_i[index])
        else:
            previous_auxiliary = self._auxiliaries

        return previous_auxiliary

    def get_auxiliaries(self):
        ret = []
        if not self._buffer:
            return self._auxiliaries

        for i in range(self.n_aux):
            aux_history = self._auxiliaries[i]
            ret.append(aux_history)

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

    def report(self, solver_name, initial_res, final_res):
        string = '{:25}: run of {:4} iteration(s)  |  '.format(solver_name, self.n_it)
        string += 'Absolute 2-norm residual = {:1.4e}  |  '.format(final_res)
        string += 'Relative 2-norm residual = {:1.4e}'.format(final_res / initial_res)
        return string


class ConjugateGradient(_LinearSolver):

    def __init__(self, lin_sys, x_0=None, M=None, tol=1e-5, buffer=None, arnoldi=False):
        if not isinstance(lin_sys.lin_op, SelfAdjointMatrix) and not lin_sys.lin_op.def_pos:
            raise LinearSolverError('Conjugate Gradient only apply to s.d.p linear map.')

        if lin_sys.lin_op.shape[0] != lin_sys.lin_op.shape[1]:
            raise LinearSolverError('Conjugate Gradient only apply to square problems.')

        if lin_sys.block:
            raise LinearSolverError('Conjugate Gradient only apply to simple left-hand side.')

        self.buffer = buffer
        self.total_cost = 0
        self.arnoldi = ([], []) if arnoldi else None

        super().__init__(lin_sys, x_0, M, tol, lin_sys.shape[0])

        self.iteration_cost = self._iteration_cost()

        if not isinstance(self.M_i, Preconditioner):
            raise LinearSolverError('Conjugate Gradient can handle only one preconditioner.')

    def _initialize(self):
        operator_cost = self.lin_sys.lin_op.apply_cost
        n, _ = self.lin_sys.lhs.shape

        r = - self.lin_sys.get_residual(self.x_0) / self.lin_sys.scaling
        z = self.M_i.apply(r)
        p = numpy.copy(z)
        auxiliaries = [z, p, r]

        residue = numpy.linalg.norm(r)
        cost = 2 * operator_cost + n
        history = [residue, cost]

        self.tol *= residue
        self.total_cost += cost

        return _SolverMonitor(history, auxiliaries=auxiliaries, buffer=self.buffer)

    def _finalize(self):
        residues, cost = self.monitor.get_history()
        z, p, r = self.monitor.get_auxiliaries()
        report = self.monitor.report('Conjugate Gradient', residues[0], residues[-1])

        arnoldi = None

        if self.arnoldi is not None:
            print(len(self.arnoldi[1]), len(self.arnoldi[0]))
            if len(self.arnoldi[1]) == len(self.arnoldi[0]):
                del self.arnoldi[1][-1]

            d_k = self.arnoldi[0]
            beta_k = self.arnoldi[1]

            t_diag = [float(d_k[0] / residues[0]**2)]
            t_sub_diag = []

            for i in range(len(self.arnoldi[1])):

                t_diag.append(float((d_k[i + 1] + beta_k[i]**2 * d_k[i]) / residues[i + 1]**2))
                t_sub_diag.append(float(-beta_k[i] * d_k[i] / residues[i] / residues[i + 1]))

            arnoldi = (t_diag, t_sub_diag)

        residues *= self.lin_sys.scaling

        output = {'report': report,
                  'x': self.x_0 + self.x * self.lin_sys.scaling,
                  'n_it': self.monitor.n_it,
                  'residues': residues,
                  'arnoldi': arnoldi,
                  'cost': cost,
                  'z': z,
                  'p': p,
                  'r': r}

        return output

    def _iteration_cost(self):
        operator_cost = self.lin_sys.lin_op.apply_cost
        n, _ = self.lin_sys.lhs.shape
        total_cost = 0

        total_cost += 2 * operator_cost     # q_k
        total_cost += 4*n + 1               # alpha
        total_cost += 2*n                   # self.x
        total_cost += 2*n                   # r_k
        total_cost += 2*n                   # residue
        total_cost += self.M_i.apply_cost   # z_k
        total_cost += 4*n + 1               # beta
        total_cost += 2*n                   # p_k

        return total_cost

    def run(self, verbose=False):
        for k in range(self.maxiter):

            z, p, r = self.monitor.get_previous()

            q_k = self.lin_sys.lin_op.dot(p)
            rho_k = z.T.dot(r)
            d_k = p.T.dot(q_k)

            alpha = rho_k / d_k
            self.x += alpha * p
            r_k = r - alpha * q_k

            residue = numpy.linalg.norm(r_k)
            self.total_cost += self.iteration_cost

            if residue < self.tol:
                self.monitor.update([residue, self.total_cost], [])
                break

            z_k = self.M_i.apply(r_k)
            beta = z_k.T.dot(r_k) / rho_k
            p_k = z_k + beta * p

            self.monitor.update([residue, self.total_cost], [z_k, p_k, r_k])

            if self.arnoldi is not None:
                self.arnoldi[0].append(d_k)
                self.arnoldi[1].append(beta)

        output = self._finalize()

        return output


class BlockConjugateGradient(_LinearSolver):

    def __init__(self, lin_sys, x_0=None, M=None, buffer=None, tol=1e-5, rank_tol=1e-5):
        if not isinstance(lin_sys.lin_op, SelfAdjointMatrix) and not lin_sys.lin_op.def_pos:
            raise LinearSolverError('Block Conjugate Gradient only apply to s.d.p linear map.')

        if lin_sys.lin_op.shape[0] != lin_sys.lin_op.shape[1]:
            raise LinearSolverError('Block Conjugate Gradient only apply to square problems.')

        if not lin_sys.block:
            raise LinearSolverError('Block Conjugate Gradient only apply to block left-hand side.')

        self.buffer = buffer
        self.total_cost = 0
        self.rank_tol = rank_tol

        super().__init__(lin_sys, x_0, M, tol, lin_sys.shape[0])

        if not isinstance(self.M_i, Preconditioner):
            raise LinearSolverError('Block Conjugate Gradient can handle only one preconditioner.')

    def _initialize(self):
        operator_cost = self.lin_sys.lin_op.apply_cost
        n, k = self.lin_sys.lhs.shape

        R = - self.lin_sys.get_residual(self.x_0) / self.lin_sys.scaling
        r = scipy.linalg.qr(R, mode='r')[0]
        s = scipy.linalg.svd(r, compute_uv=False)
        rank = numpy.sum(s * (1 / s[0]) > self.rank_tol)
        eps_rank = s[-1] / s[0]

        Z = self.M_i.apply(R[:, :rank])
        P = numpy.copy(Z)
        auxiliaries = [Z, P, R]

        residue = numpy.linalg.norm(R[:, :rank])
        cost = 2*operator_cost*k + n*k + 2 * self.lin_sys.lhs.size
        history = [residue, cost, rank, eps_rank]

        self.tol *= residue
        self.total_cost += cost

        return _SolverMonitor(history, auxiliaries=auxiliaries, buffer=self.buffer)

    def _finalize(self):
        residues, cost, rank, eps_rank = self.monitor.get_history()
        residues *= numpy.linalg.norm(self.lin_sys.scaling)

        Z, P, R = self.monitor.get_auxiliaries()

        report = self.monitor.report('Block Conjugate Gradient', residues[0], residues[-1])

        output = {'report': report,
                  'x': self.x_0 + self.x * self.lin_sys.scaling,
                  'n_it': self.monitor.n_it,
                  'residues': residues,
                  'cost': cost,
                  'rank': rank,
                  'eps_rank': eps_rank,
                  'Z': Z,
                  'P': P,
                  'R': R}

        return output

    def _iteration_cost(self, r):
        operator_cost = self.lin_sys.lin_op.apply_cost
        n, k = self.lin_sys.lhs.shape
        total_cost = 0

        total_cost += 2 * operator_cost * r     # Q_k
        total_cost += 2 * n * r**2              # delta
        total_cost += 4*k*r**2 + 22*k**3        # SVD of delta
        total_cost += r                         # s_inv
        total_cost += 2*(r*n*k + 3*r*r*k)       # alpha
        total_cost += n*k + 2*r*n*k             # self.x
        total_cost += n*k + 2*r*n*k             # R_k
        total_cost += 2*n*r**2                  # residue
        total_cost += self.M_i.apply_cost       # Z_k
        total_cost += 2*(r*n*r + 3*r*r*k)       # beta
        total_cost += 2*n*r*r + n*r             # P_k

        return total_cost

    def run(self, verbose=False):
        for k in range(self.maxiter):

            Z, P, R = self.monitor.get_previous()

            Q_k = self.lin_sys.lin_op.dot(P)

            delta = P.T.dot(Q_k)
            u, s, v = numpy.linalg.svd(delta)
            rank = numpy.sum(s * (1 / s[0]) > self.rank_tol)
            eps_rank = s[-1] / s[0]

            s_inv = numpy.zeros_like(s)
            s_inv[:rank] = 1 / s[:rank]

            alpha = v.T @ numpy.diag(s_inv) @ u.T @ Z.T.dot(R)
            self.x += P.dot(alpha)

            R -= Q_k.dot(alpha)

            residue = numpy.linalg.norm(R[:, :rank])
            self.total_cost += self._iteration_cost(rank)

            if residue < self.tol:
                self.monitor.update([residue, self.total_cost, rank, eps_rank], [])
                break

            Z = self.M_i.apply(R[:, :rank])

            beta = - v.T @ numpy.diag(s_inv) @ u.T @ Q_k.T.dot(Z)
            P = Z + P.dot(beta)

            self.monitor.update([residue, self.total_cost, rank, eps_rank], [Z, P, R])

        output = self._finalize()

        return output