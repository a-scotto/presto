#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on April 25, 2019 at 11:02.

@author: a.scotto

Description:
"""

import numpy
import scipy.linalg
import scipy.sparse

from typing import Union
from utils.utils import *
from utils.linalg import *
from core.algebra import *
from matplotlib import pyplot
from core.preconditioner import *

__all__ = ['LinearSystem', 'ConjugateGradient', 'DeflatedConjugateGradient', 'BlockConjugateGradient',
           'FlexibleConjugateGradient']

Subspace = Union[numpy.ndarray, scipy.sparse.spmatrix]
# MultiPreconditioner = Union[Preconditioner, List[Preconditioner]]


class LinearSystemError(Exception):
    """
    Exception raised when LinearSystem object encounters specific errors.
    """


class IterativeSolverError(Exception):
    """
    Exception raised when _IterativeSolver object encounters specific errors.
    """


class AssumptionError(Exception):
    """
    Exception raised when assumptions for a given iterative method are not satisfied.
    """


class LinearSystem(object):
    def __init__(self,
                 linear_op: LinearOperator,
                 rhs: numpy.ndarray,
                 M: MultiPreconditioner = None,
                 x_opt: numpy.ndarray = None):
        """
        Abstract representation for linear system of the form Ax = b, where A is a linear operator and b the right-hand
        side. Information related to the nature of the linear operator are provided since the iterative methods for the
        solution of linear systems varies with these properties.

        :param linear_op: Linear operator involved in the linear system.
        :param rhs: Right-hand side associated.
        :param M: Preconditioner or list of Preconditioner to be used to enhance solver convergence.
        :param x_opt: Solution of the linear system if available.
        """
        # Sanitize the linear operator attribute
        if not isinstance(linear_op, LinearOperator):
            raise LinearSystemError('Linear operator must be an instance of LinearOperator.')

        self.linear_op = linear_op

        # Sanitize the right-hand side attribute
        if not isinstance(rhs, numpy.ndarray):
            raise LinearSystemError('Right-hand side must be a numpy.ndarray.')

        if rhs.shape[0] != linear_op.shape[1]:
            raise LinearSystemError('Linear operator and right-hand side have inconsistent shapes.')

        self.rhs = rhs

        # Sanitize the preconditioner attribute
        M = IdentityPreconditioner(linear_op) if M is None else M

        if isinstance(M, list) and not numpy.all([isinstance(M_i, Preconditioner) for M_i in M]):
            raise LinearSystemError('List of preconditioners must solely contain instances of Preconditioner.')

        if not isinstance(M, list) and not isinstance(M, Preconditioner):
            raise LinearSystemError('Preconditioner must be an instance of Preconditioner.')

        if isinstance(M, list) and len(M) != rhs.shape[1]:
            raise LinearSystemError('There must be as many Preconditioners as right-hand side, if several provided.')

        self.M = M

        # Sanitize the solution argument
        if x_opt is not None and not isinstance(x_opt, numpy.ndarray):
            raise LinearSystemError('Exact solution provided must be a numpy.ndarray.')

        if x_opt is not None and x_opt.shape[0] != linear_op.shape[1]:
            raise LinearSystemError('Linear operator and its solution have inconsistent shapes.')

        self.x_opt = x_opt

        # Keep track of whether the system has multiple right-hand sides
        self.block = False if self.rhs.shape[1] == 1 else True

    def get_residual(self, x: numpy.ndarray) -> numpy.ndarray:
        """
        Compute the explicit residual r(x) = b - A(x).

        :param x: Vector on which the residual is computed.
        """
        return self.rhs - self.linear_op.dot(x)

    def get_error_norm(self, x: numpy.ndarray) -> numpy.ndarray:
        """
        Compute the 2-norm error e(x) = ||x - x_opt||_2, if x_opt is available.

        :param x: Vector on which the error is computed.
        """
        if self.x_opt is None:
            raise LinearSystemError('Cannot compute error norm since linear system solution is not known.')

        return numpy.linalg.norm(x - self.x_opt)

    def __repr__(self) -> str:
        """
        Set a user-friendly printing for linear systems.
        """
        _repr = 'Linear system of shape {} with right-hand side of shape {}.'\
                .format(self.linear_op.shape, self.rhs.shape)

        return _repr


class _IterativeSolver(object):
    def __init__(self,
                 linear_system: LinearSystem,
                 x0: numpy.ndarray = None,
                 tol: float = 1e-5,
                 maxiter: int = None):
        """
        Abstract class for iterative methods, i.e. method aimed at finding an approximate solution of a linear system of
        the form Ax = b. The approximate solution is computed through a sequential procedure expected to provide better 
        and better approximates with the iterations, that is, expected to reduce the residual norm at each iteration.
        
        :param linear_system: Linear system to be solved.
        :param x0: Initial guess for the linear system solution.
        :param tol: Relative tolerance threshold under which the algorithm is stopped.
        :param maxiter: Maximum number of iterations affordable before stopping the method.
        """
        # Sanitize the linear system argument
        if not isinstance(linear_system, LinearSystem):
            raise IterativeSolverError('Linear system to be solved must be an instance of LinearSystem.')

        self.linear_system = linear_system
        self.M = self.linear_system.M

        # Sanitize the initial guess argument
        x0 = numpy.zeros_like(linear_system.rhs) if x0 is None else x0

        if x0 is not None and not isinstance(x0, numpy.ndarray):
            raise IterativeSolverError('Initial guess x0 must be of a numpy.ndarray.')

        if x0.shape != linear_system.rhs.shape:
            raise IterativeSolverError('Shapes of initial guess and right-hand side are inconsistent.')

        self.x0 = x0
        self.yk = numpy.zeros_like(x0)
        self.xk = None

        # Sanitize the tolerance argument
        if not isinstance(tol, float) or tol < 0:
            raise IterativeSolverError('Tolerance must be a positive real number.')

        self.tol = tol

        # Sanitize the maximum iteration number argument
        maxiter = self.x0.size if maxiter is None else maxiter

        if not isinstance(maxiter, int) or maxiter < 0:
            raise IterativeSolverError('The maximum number of iteration must be a positive integer.')

        self.maxiter = maxiter

        self.residue_norms = list()
        self.timers = dict(A=Timer('Linear operator'),
                           M=Timer('Preconditioner'),
                           Ro=Timer('Re-orthogonalization'))
        self.N = 0

        self._solve()
        self._finalize()

    def _solve(self) -> None:
        """
        Method to solve the linear system with the iterative method.
        """
        raise NotImplemented('Iterative solver must implement a _solve method.')

    def _finalize(self) -> None:
        """
        Method to post-process results of the iterative method.
        """
        raise NotImplemented('Iterative solver must implement a _finalize method.')

    def plot_convergence(self):
        label = self.__class__.__name__ + ' with ' + self.M.__class__.__name__
        pyplot.gca().plot(self.residue_norms, label=label)


class ConjugateGradient(_IterativeSolver):
    def __init__(self,
                 linear_system: LinearSystem,
                 x0: numpy.ndarray = None,
                 tol: float = 1e-5,
                 maxiter: int = None,
                 store_arnoldi: bool = False) -> None:
        """
        Abstract class for the (Preconditioned) Conjugate Gradient algorithm implementation.

        :param linear_system: Linear system to be solved.
        :param x0: Initial guess for the linear system solution.
        :param tol: Relative tolerance threshold under which the algorithm is stopped.
        :param maxiter: Maximum number of iterations affordable before stopping the method.
        :param store_arnoldi: Whether or not storing the Arnoldi relation elements.
        """
        # Sanitize the linear system argument
        if linear_system.linear_op.shape[0] != linear_system.linear_op.shape[1]:
            raise AssumptionError('Conjugate Gradient only apply to squared operators.')

        if linear_system.block:
            raise AssumptionError('Conjugate Gradient only apply to single right-hand side.')

        # Sanitize the Arnoldi relation storage argument
        self.store_arnoldi = store_arnoldi
        self.H = dict(alpha=[], beta=[]) if store_arnoldi else None
        self.V = list() if store_arnoldi else None

        # Instantiate iterative solver base class
        super().__init__(linear_system,
                         x0=x0,
                         tol=tol,
                         maxiter=maxiter)

    def _solve(self) -> None:
        k = 0

        # Initialize the algorithm
        with self.timers['A']:
            rk = self.linear_system.get_residual(self.x0)

        with self.timers['M']:
            zk = self.M.dot(rk)

        pk = numpy.copy(zk)

        rhos = list([float(zk.T @ rk)])

        self.residue_norms.append(rhos[-1]**0.5)

        if self.store_arnoldi:
            self.V.append(zk / self.residue_norms[-1])

        # Relative tolerance with respect to ||b||_M
        self.tol = self.tol * (norm(self.linear_system.rhs, ip_B=self.M))

        while self.residue_norms[-1] > self.tol and k < self.maxiter:
            with self.timers['A']:
                qk = self.linear_system.linear_op.dot(pk)
            dk = pk.T @ qk
            alpha = rhos[-1] / dk

            self.yk += alpha * pk

            rk = rk - alpha * qk
            with self.timers['M']:
                zk = self.M.dot(rk)

            rhos.append(float(zk.T @ rk))
            self.residue_norms.append(rhos[-1]**0.5)

            beta = rhos[-1] / rhos[-2]
            pk = zk + beta * pk

            if self.store_arnoldi:
                self.H['alpha'].append(alpha)
                self.H['beta'].append(beta)
                self.V.append(zk / self.residue_norms[-1])

            k = k + 1

        self.xk = self.x0 + self.yk
        self.N = k

    def _finalize(self) -> None:
        # Scale the residuals norms
        self.residue_norms = numpy.asarray(self.residue_norms) / self.residue_norms[0]

        # Compute the Arnoldi tridiagonal matrix coefficient if it was required
        if self.store_arnoldi:
            alpha = self.H['alpha']
            beta = self.H['beta']

            self.H = scipy.sparse.dok_matrix((self.N+1, self.N))

            for i in range(self.N):
                if i == 0:
                    self.H[i, i] = 1 / alpha[i]
                    self.H[i+1, i] = -beta[i]**0.5 / alpha[i]
                else:
                    self.H[i-1, i] = self.H[i, i-1]
                    self.H[i, i] = 1 / alpha[i] + beta[i-1] / alpha[i-1]
                    self.H[i+1, i] = -beta[i]**0.5 / alpha[i]

            self.H = self.H.todense()
            self.V = numpy.hstack(self.V)

        for key, value in self.timers.items():
            self.timers[key] = self.timers[key].total_elapsed

    def __repr__(self):
        """
        Provide a readable report of the Conjugate Gradient algorithm results.
        """
        n_iterations = self.N
        final_residual = self.residue_norms[-1]
        residual_decrease = self.residue_norms[-1] / self.residue_norms[0]

        _repr = 'Conjugate Gradient: {:4} iteration(s) |  '.format(n_iterations)
        _repr += '||Axk - b||_M = {:1.4e} | '.format(final_residual)
        _repr += '||Axk - b||_M / ||b||_M = {:1.4e}'.format(residual_decrease)

        return _repr


class FlexibleConjugateGradient(_IterativeSolver):
    def __init__(self,
                 linear_system: LinearSystem,
                 x0: numpy.ndarray = None,
                 m: int = numpy.Inf,
                 tol: float = 1e-5,
                 maxiter: int = None) -> None:
        """
        Abstract class for the (Preconditioned) Conjugate Gradient algorithm implementation.

        :param linear_system: Linear system to be solved.
        :param x0: Initial guess for the linear system solution.
        :param m: Length of the re-orthogonalization window.
        :param tol: Relative tolerance threshold under which the algorithm is stopped.
        :param maxiter: Maximum number of iterations affordable before stopping the method.
        """
        # Sanitize the linear system argument
        if linear_system.linear_op.shape[0] != linear_system.linear_op.shape[1]:
            raise AssumptionError('Conjugate Gradient only apply to squared operators.')

        if linear_system.block:
            raise AssumptionError('Conjugate Gradient only apply to single right-hand side.')

        # Sanitize the Arnoldi relation storage argument
        self.m = m

        # Instantiate iterative solver base class
        super().__init__(linear_system,
                         x0=x0,
                         tol=tol,
                         maxiter=maxiter)

    def _solve(self) -> None:
        k = 0

        # Initialize the algorithm
        with self.timers['A']:
            rk = self.linear_system.get_residual(self.x0)

        with self.timers['M']:
            zk = self.M.dot(rk)

        pk = numpy.copy(zk)

        rhos = list([float(zk.T @ rk)])
        pks, qks, dks = list([pk]), list(), list()

        self.residue_norms.append(rhos[-1]**0.5)

        # Relative tolerance with respect to ||b||_M
        self.tol = self.tol * (norm(self.linear_system.rhs, ip_B=self.M))

        while self.residue_norms[-1] > self.tol and k < self.maxiter:
            with self.timers['A']:
                qk = self.linear_system.linear_op.dot(pk)
            dk = pk.T @ qk
            alpha = rhos[-1] / dk

            if len(qks) == self.m:
                del qks[0], dks[0]
            qks.append(qk)
            dks.append(dk)

            self.yk += alpha * pk

            rk = rk - alpha * qk
            with self.timers['M']:
                zk = self.M.dot(rk)

            rhos.append(float(zk.T @ rk))
            self.residue_norms.append(rhos[-1]**0.5)

            pk = zk
            with self.timers['Ro']:
                for i in range(len(qks)):
                    pk -= (zk.T @ qks[i]) / dks[i] * pks[i]

            if len(pks) == self.m:
                del pks[0]
            pks.append(pk)

            k = k + 1

        self.xk = self.x0 + self.yk
        self.N = k

    def _finalize(self) -> None:
        # Scale the residuals norms
        self.residue_norms = numpy.asarray(self.residue_norms) / self.residue_norms[0]

        for key, value in self.timers.items():
            self.timers[key] = self.timers[key].total_elapsed

    def __repr__(self):
        """
        Provide a readable report of the Conjugate Gradient algorithm results.
        """
        n_iterations = self.N
        final_residual = self.residue_norms[-1]
        residual_decrease = self.residue_norms[-1] / self.residue_norms[0]

        _repr = 'Conjugate Gradient: {:4} iteration(s) |  '.format(n_iterations)
        _repr += '||Axk - b||_M = {:1.4e} | '.format(final_residual)
        _repr += '||Axk - b||_M / ||b||_M = {:1.4e}'.format(residual_decrease)

        return _repr


class DeflatedConjugateGradient(ConjugateGradient):
    def __init__(self,
                 linear_system: LinearSystem,
                 subspace: Subspace,
                 factorized: bool = False,
                 x0: numpy.ndarray = None,
                 tol: float = 1e-5,
                 maxiter: int = None,
                 store_arnoldi: bool = False) -> None:
        """
        Abstract class for the (Preconditioned) Conjugate Gradient algorithm implementation.

        :param linear_system: Linear system to be solved.
        :param subspace: matrix representation of subspace to use for deflation.
        :param factorized: either to process a factorization of the subspace matrix.
        :param x0: Initial guess for the linear system solution.
        :param tol: Relative tolerance threshold under which the algorithm is stopped.
        :param maxiter: Maximum number of iterations affordable before stopping the method.
        :param store_arnoldi: Whether or not storing the Arnoldi relation elements.
        """
        # Instantiate iterative solver base class
        A = linear_system.linear_op
        b = linear_system.rhs
        M = linear_system.M

        # Compute deflated linear system
        self.P = OrthogonalProjector(subspace, ip_B=A, factorized=factorized)
        deflated_A = (IdentityOperator(b.size) - self.P.T) @ A
        deflated_b = b - self.P.T.dot(b)
        deflated_linsys = LinearSystem(deflated_A, deflated_b, M)

        # Adjust tolerance
        tol = tol * norm(b, ip_B=M) / norm(deflated_b, ip_B=M)

        # Instantiate conjugate gradient base class
        super().__init__(deflated_linsys, x0, tol, maxiter, store_arnoldi)

        # Correction of the deflated solution to get initial linear system solution
        correction = self.P.V.T.dot(b)
        scipy.linalg.cho_solve(self.P.L_factor, correction, overwrite_b=True)
        correction = self.P.V.dot(correction)
        self.xk = correction + self.xk - self.P.dot(self.xk)


class BlockConjugateGradient(_IterativeSolver):
    """
    Abstract class to implement the Block Conjugate Gradient method with possible use of
    preconditioner.
    """

    def __init__(self,
                 linear_system: LinearSystem,
                 x0: numpy.ndarray = None,
                 tol: float = 1e-5,
                 maxiter: int = None,
                 store_arnoldi: bool = False,
                 rank_tol: float = 1e-10):
        """
        Constructor of the ConjugateGradient class.

        :param linear_system: LinearSystem to solve.
        :param x0: Initial guess of the linear system.
        :param tol: Relative tolerance to achieve before stating convergence.
        :param rank_tol: Maximum ratio of extrema singular values accepted to consider full rank.
        """
        # Check if Block Conjugate Gradient assumptions are satisfied
        if linear_system.linear_op.shape[0] != linear_system.linear_op.shape[1]:
            raise AssumptionError('Block Conjugate Gradient only apply to squared operators.')

        if not linear_system.block:
            raise AssumptionError('Block Conjugate Gradient only apply to single right-hand side.')

        # Initialize BCG parameters
        self.scaling = None
        self.rank_tol = rank_tol

        # Sanitize the Arnoldi relation storage argument
        self.store_arnoldi = store_arnoldi
        self.H = dict(alpha=[], beta=[]) if store_arnoldi else None
        self.V = list() if store_arnoldi else None

        # Instantiate iterative solver base class
        super().__init__(linear_system,
                         x0=x0,
                         tol=tol,
                         maxiter=maxiter)

    def _solve(self) -> None:
        """
        Run the Block Conjugate Gradient method.
        """
        k = 0

        # Initialize the algorithm with scaled residual R_0, then Z_0 and P_0
        if isinstance(self.M, list):
            self.scaling = numpy.asarray([norm(self.linear_system.rhs[:, [i]], ip_B=self.M[i])
                                          for i in range(len(self.M))])
        else:
            self.scaling = norm(self.linear_system.rhs, ip_B=self.M)

        Rk = self.linear_system.get_residual(self.x0) / self.scaling

        if isinstance(self.M, list):
            Zk = numpy.hstack([self.M[i].dot(Rk[:, [i]]) for i in range(Rk.shape[1])])
        else:
            Zk = self.M.dot(Rk)

        Pk = numpy.copy(Zk)

        rhos = list([Zk.T @ Rk])
        self.residue_norms.append(numpy.diag(rhos[-1])**0.5)
        active_cols = self.residue_norms[-1] > self.tol

        while max(self.residue_norms[-1]) > self.tol and k < self.maxiter:
            Qk = self.linear_system.linear_op.dot(Pk)
            delta = Pk.T.dot(Qk)

            # Computation of A-conjugation corrector alpha
            U, sigma, V = scipy.linalg.svd(delta)
            effective_rank = numpy.sum(sigma / sigma[0] > self.rank_tol)
            sigma_inv = numpy.zeros_like(sigma)
            sigma_inv[:effective_rank] = 1 / sigma[:effective_rank]
            alpha = V.T @ numpy.diag(sigma_inv) @ U.T @ Zk.T @ Rk

            # Update of active iterate
            self.yk += Pk.dot(alpha)
            Rk -= Qk.dot(alpha)

            if isinstance(self.M, list):
                for i in range(Zk.shape[1]):
                    if active_cols[i]:
                        Zk[:, [i]] = self.M[i].dot(Rk[:, [i]])
            else:
                Zk = self.M.dot(Rk)

            rhos.append(Zk.T @ Rk)
            self.residue_norms.append(numpy.diag(rhos[-1]) ** 0.5)

            # print(V.shape, U.shape, Zk[:, active_cols].shape, Qk.shape)
            beta = - V.T @ numpy.diag(sigma_inv) @ U.T @ Zk.T @ Qk
            Pk = Zk + Pk.dot(beta)
            active_cols = self.residue_norms[-1] > self.tol

            k = k + 1

        self.xk = self.x0 + self.yk
        self.N = k

    def _finalize(self) -> None:
        """
        Finalize the Block Conjugate Gradient algorithm by post-processing the data aggregated
        during the run and making up the output dictionary gathering all the final values.
        """

        # Get residuals and iterations cost and scale back the residuals

    def _iteration_cost(self, alpha) -> float:
        """
        Details the Flops counting for one iteration of the Block Conjugate Gradient.

        :param alpha: Matrix equal to (P^T * A * P)^(-1) * (Z^T * R) which size gives the number
        of not yet converged columns and rank deficiency.
        """

        n, k = self.linear_system.rhs.shape
        r, c = alpha.shape
        total_cost = 0

        total_cost += self.linear_system.linear_op.matvec_cost * r    # Q_k
        total_cost += 2*n*r**2                              # delta
        total_cost += 26*r**3                               # SVD of delta
        total_cost += r                                     # s_inv
        total_cost += 2*(r*n*c + 3*r*r*c)                   # alpha
        total_cost += n*c + 2*r*n*c                         # self.x
        total_cost += n*c + 2*r*n*c                         # R_k
        total_cost += 2*n*r**2                              # residue
        total_cost += self.M.matvec_cost                   # Z_k
        total_cost += 2*(r*n*r + 3*r*r*k)                   # beta
        total_cost += 2*n*r*r + n*r                         # P_k

        return total_cost
