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

from core.linop import LinearOperator, MatrixOperator
from core.preconditioner import Preconditioner, IdentityPreconditioner

__all__ = ['LinearSystem', 'ConjugateGradient']


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
                 rhs: numpy.ndarray):
        """
        Abstract representation for linear system of the form Ax = b, where A is a linear operator and b the right-hand
        side. Information related to the nature of the linear operator are provided since the iterative methods for the
        solution of linear systems varies with these properties.

        :param linear_op: Linear operator involved in the linear system.
        :param rhs: Right-hand side associated.
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

        # Initialize the linear system's attributes
        self.rhs = rhs

        self.block = False if self.rhs.shape[1] == 1 else True

    def get_residual(self, x: numpy.ndarray) -> numpy.ndarray:
        """
        Compute the explicit residual r(x) = b - A(x).

        :param x: Vector on which the residual is computed.
        """
        return self.rhs - self.linear_op.dot(x)

    def perturb(self, epsilon):
        """
        Compute a modified linear system via a random perturbation of the linear operator and the right-hand side.

        :param epsilon: Magnitude of the perturbation
        """
        if not isinstance(self.linear_op, MatrixOperator):
            raise LinearSystemError('Perturbation can only be applied to linear systems involving a MatrixOperator.')

        normalization_ = numpy.linalg.norm(self.linear_op.matrix.diagonal())
        perturbed_linop = self.linear_op.matrix + epsilon * normalization_ * scipy.sparse.eye(self.rhs.size)
        perturbed_rhs = self.rhs + epsilon * normalization_ * numpy.random.randn(self.rhs.size, 1)

        A_ = MatrixOperator(perturbed_linop,
                            self_adjoint=self.linear_op.self_adjoint,
                            positive_definite=self.linear_op.positive_definite)

        b_ = perturbed_rhs

        return LinearSystem(A_, b_)

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
                 M: Preconditioner = None,
                 tol: float = 1e-5,
                 maxiter: int = None):
        """
        Abstract class for iterative methods, i.e. method aimed at finding an approximate solution of a linear system of
        the form Ax = b. The approximate solution is computed through a sequential procedure expected to provide better 
        and better approximates with the iterations, that is, expected to reduce the residual norm at each iteration.
        
        :param linear_system: Linear system to be solved.
        :param x0: Initial guess for the linear system solution.
        :param M: Preconditioner to be used to enhance solver convergence.
        :param tol: Relative tolerance threshold under which the algorithm is stopped.
        :param maxiter: Maximum number of iterations affordable before stopping the method.
        """
        # Sanitize the linear system argument
        if not isinstance(linear_system, LinearSystem):
            raise IterativeSolverError('Linear system to be solved must be an instance of LinearSystem.')

        self.linear_system = linear_system

        # Sanitize the initial guess argument
        x0 = numpy.zeros_like(linear_system.rhs) if x0 is None else x0

        if x0 is not None and not isinstance(x0, numpy.ndarray):
            raise IterativeSolverError('Initial guess x0 must be of a numpy.ndarray.')

        if x0.shape != linear_system.rhs.shape:
            raise IterativeSolverError('Shapes of initial guess and right-hand side are inconsistent.')

        self.x0 = x0
        self.yk = numpy.zeros_like(x0)
        self.xk = None

        # Sanitize the preconditioner attribute
        M = IdentityPreconditioner(linear_system.linear_op) if M is None else M

        if not isinstance(M, Preconditioner):
            raise IterativeSolverError('Preconditioner must be an instance of Preconditioner.')

        if not (M.linear_op is self.linear_system.linear_op):
            raise IterativeSolverError('Preconditioner must be related to the same linear operator as the one in the '
                                       'linear system.')

        self.M = M

        # Sanitize the tolerance argument
        if not isinstance(tol, float) or tol < 0:
            raise IterativeSolverError('Tolerance must be a positive real number.')

        self.tol = tol

        # Sanitize the maximum iteration number argument
        maxiter = self.x0.size if maxiter is None else maxiter

        if not isinstance(maxiter, int) or maxiter < 0:
            raise IterativeSolverError('The maximum number of iteration must be a positive integer.')

        self.maxiter = maxiter

    def _initialize(self):
        raise NotImplemented('Iterative solver must implement a _initialize method.')

    def _finalize(self):
        raise NotImplemented('Iterative solver must implement a _finalize method.')

    def _solve(self):
        raise NotImplemented('Iterative solver must implement a _solve method.')


class ConjugateGradient(_IterativeSolver):
    def __init__(self,
                 linear_system: LinearSystem,
                 x0: numpy.ndarray = None,
                 M: Preconditioner = None,
                 tol: float = 1e-5,
                 maxiter: int = None,
                 store_arnoldi: bool = False) -> None:
        """
        Abstract class for the (Preconditioned) Conjugate Gradient algorithm implementation.

        :param linear_system: Linear system to be solved.
        :param x0: Initial guess for the linear system solution.
        :param M: Preconditioner to be used to enhance solver convergence.
        :param tol: Relative tolerance threshold under which the algorithm is stopped.
        :param maxiter: Maximum number of iterations affordable before stopping the method.
        :param store_arnoldi: Whether or not storing the Arnoldi relation elements.
        """
        # Instantiate iterative solver base class
        super().__init__(linear_system,
                         x0=x0,
                         M=M,
                         tol=tol,
                         maxiter=maxiter)

        # Ensure the linear system lies in the assumptions of the conjugate gradient method
        if not linear_system.linear_op.self_adjoint and not linear_system.linear_op.positive_definite:
            raise AssumptionError('Conjugate Gradient only apply to s.d.p linear operators.')

        if linear_system.linear_op.shape[0] != linear_system.linear_op.shape[1]:
            raise AssumptionError('Conjugate Gradient only apply to squared operators.')

        if linear_system.block:
            raise AssumptionError('Conjugate Gradient only apply to single right-hand side.')

        # Get initial residues
        self.residue_norms = list()
        self.N = 0

        # Sanitize the Arnoldi relation storage argument
        self.store_arnoldi = store_arnoldi
        self.H = dict(alpha=[], beta=[]) if store_arnoldi else None
        self.V = list() if store_arnoldi else None

        # Conjugate Gradient method run
        self._solve()
        self._finalize()

    def _solve(self) -> None:
        """
        Run the Conjugate Gradient algorithm.
        """
        k = 0

        # Initialize the algorithm with scaled residual r0, then z0 and p0
        rk = self.linear_system.get_residual(self.x0)
        zk = self.M.dot(rk)
        pk = numpy.copy(zk)

        rhos = list([float(zk.T @ rk)])

        self.residue_norms.append(rhos[-1]**0.5)

        if self.store_arnoldi:
            self.V.append(zk / self.residue_norms[-1])
        
        while self.residue_norms[-1] > self.tol * self.residue_norms[0] and k < self.maxiter:
            qk = self.linear_system.linear_op.dot(pk)
            dk = pk.T @ qk
            alpha = rhos[-1] / dk

            self.yk += alpha * pk

            rk = rk - alpha * qk
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
        """
        Post-process the output of the CG, namely:
         * build the final iterate xk
         * build the Arnoldi matrix H and vectors V
        """

        # Get residuals and iterations cost and scale back the residuals
        self.residue_norms = self.residue_norms

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

    def __repr__(self):
        """
        Provide a readable report of the Conjugate Gradient algorithm results.
        """

        n_iterations = self.N
        final_residual = self.residue_norms[-1]
        residual_decrease = self.residue_norms[-1] / self.residue_norms[0]

        _repr = 'Conjugate Gradient: {:4} iteration(s) |  '.format(n_iterations)
        _repr += '||Axk - b|| = {:1.4e} | '.format(final_residual)
        _repr += '||Axk - b|| / ||Ax0 - b|| = {:1.4e}'.format(residual_decrease)

        return _repr


# class BlockConjugateGradient(_IterativeSolver):
#     """
#     Abstract class to implement the Block Conjugate Gradient method with possible use of
#     preconditioner.
#     """
#
#     def __init__(self,
#                  linear_system: LinearSystem,
#                  x0: numpy.ndarray = None,
#                  M: Preconditioner = None,
#                  buffer: int = 0,
#                  tol: float = 1e-5,
#                  rank_tol: float = 1e-5):
#         """
#         Constructor of the ConjugateGradient class.
#
#         :param linear_system: LinearSystem to solve.
#         :param x0: Initial guess of the linear system.
#         :param M: Preconditioner to use during the resolution.
#         :param tol: Relative tolerance to achieve before stating convergence.
#         :param buffer: Size of the buffer memory.
#         :param rank_tol: Maximum ratio of extrema singular values accepted to consider full rank.
#         """
#
#         # Sanitize the linear system attribute
#         if not hasattr(linear_system.linear_op, 'def_pos'):
#             raise IterativeSolverError('Block Conjugate Gradient only dot to s.d.p linear map.')
#
#         if linear_system.linear_op.shape[0] != linear_system.linear_op.shape[1]:
#             raise IterativeSolverError('Block Conjugate Gradient only dot to square problems.')
#
#         if not linear_system.block:
#             raise IterativeSolverError('Block Conjugate Gradient only dot to block right-hand side.')
#
#         self.buffer = buffer
#         self.total_cost = 0
#         self.rank_tol = rank_tol
#
#         super().__init__(linear_system, x0, M, tol, linear_system.shape[0])
#
#         # Sanitize the preconditioner attribute
#         if not isinstance(self.M_i, Preconditioner):
#             raise IterativeSolverError('Conjugate Gradient can handle only one preconditioner.')
#
#     def _initialize(self):
#         """
#         Initialization of the Block Conjugate Gradient method. Namely compute the initial block
#         residue R_0, block descent direction P_0, and block preconditioned residual Z_0.
#         """
#         n, k = self.x0.shape
#
#         # Initialize the algorithm with scaled residual R_0, then Z_0 and P_0
#         R = self.linear_system.get_residual(self.x0) / self.linear_system.scaling
#
#         # Detect rank deficiency in the initial block residue with QR-SVD
#         r = scipy.linalg.qr(R, mode='r')[0]
#         s = scipy.linalg.svd(r, compute_uv=False)
#         effective_rank = numpy.sum(s * (1 / s[0]) > self.rank_tol)
#         eps_rank = s[-1] / s[0]
#
#         Z = self.M_i.dot(R[:, :effective_rank])
#         P = numpy.copy(Z)
#
#         # Aggregate as auxiliaries quantities
#         auxiliaries = [Z, P, R]
#
#         # Initial residual in max norm
#         residue = max(numpy.linalg.norm(R[:, :effective_rank], axis=0))
#         cost = k * (self.linear_system.linear_op.matvec_cost + self.M_i.matvec_cost)
#         cost += 2*n*k**2 + 2*k**3
#
#         # Aggregate as history quantities
#         history = [residue, cost, effective_rank, eps_rank]
#
#         # Scale the tolerance and update total cost
#         self.tol *= residue
#         self.total_cost += cost
#
#         return _SolverMonitor(history, auxiliaries=auxiliaries, buffer=self.buffer)
#
#     def _finalize(self) -> None:
#         """
#         Finalize the Block Conjugate Gradient algorithm by post-processing the data aggregated
#         during the run and making up the output dictionary gathering all the final values.
#         """
#
#         # Get residuals and iterations cost and scale back the residuals
#         residues, cost, rank, eps_rank = self.monitor.get_history()
#         residues *= numpy.linalg.norm(self.linear_system.scaling)
#
#         # Get the available auxiliaries, depending on the buffer content
#         Z, P, R = self.monitor.get_auxiliaries()
#
#         # Make up the report line to print
#         report = self.monitor.report('Block Conjugate Gradient', residues[0], residues[-1])
#
#         # Make up the output dictionary with all final values
#         output = dict(report=report,
#                       x_opt=self.x0 + self.x * self.linear_system.scaling,
#                       n_iterations=self.monitor.n_it,
#                       residues=residues,
#                       cost=cost,
#                       z=Z,
#                       p=P,
#                       r=R)
#
#         self.output = output
#
#     def _iteration_cost(self, alpha) -> float:
#         """
#         Details the Flops counting for one iteration of the Block Conjugate Gradient.
#
#         :param alpha: Matrix equal to (P^T * A * P)^(-1) * (Z^T * R) which size gives the number
#         of not yet converged columns and rank deficiency.
#         """
#
#         n, k = self.linear_system.rhs.shape
#         r, c = alpha.shape
#         total_cost = 0
#
#         total_cost += self.linear_system.linear_op.matvec_cost * r    # Q_k
#         total_cost += 2*n*r**2                              # delta
#         total_cost += 26*r**3                               # SVD of delta
#         total_cost += r                                     # s_inv
#         total_cost += 2*(r*n*c + 3*r*r*c)                   # alpha
#         total_cost += n*c + 2*r*n*c                         # self.x
#         total_cost += n*c + 2*r*n*c                         # R_k
#         total_cost += 2*n*r**2                              # residue
#         total_cost += self.M_i.matvec_cost                   # Z_k
#         total_cost += 2*(r*n*r + 3*r*r*k)                   # beta
#         total_cost += 2*n*r*r + n*r                         # P_k
#
#         return total_cost
#
#     def run(self, verbose=False):
#         """
#         Method to actually run the Block Conjugate Gradient algorithm. At of the run the end, it
#         update its own output attribute with the final values found.
#         """
#
#         for k in range(self.maxiter):
#             # Retrieve last iterates quantities
#             Z, P, R = self.monitor.get_previous()
#
#             Q_k = self.linear_system.linear_op.dot(P)
#
#             # SVD decomposition of P^T*A*P
#             delta = P.T.dot(Q_k)
#             u, sigma, v = numpy.linalg.svd(delta)
#
#             # Determination of numerical rank with given tolerance
#             rank = numpy.sum((sigma > sigma[0] * self.rank_tol))
#
#             # Pseudo inverse singular values
#             sigma_inv = numpy.zeros_like(sigma)
#             sigma_inv[:rank] = 1 / sigma[:rank]
#
#             # Spot not yet converged columns to determine the remaining active ones
#             residues = numpy.linalg.norm(R, axis=0)
#             active_cols = residues > self.tol
#
#             # Computation of A-conjugation corrector alpha
#             alpha = v.T @ numpy.diag(sigma_inv) @ u.T @ Z.T.dot(R[:, active_cols])
#
#             # Update of active iterate
#             self.x[:, active_cols] += P.dot(alpha)
#             R[:, active_cols] -= Q_k.dot(alpha)
#
#             # Update historic data
#             residue = max(residues)
#             self.total_cost += self._iteration_cost(alpha)
#
#             # Break whenever max norm tolerance is reached
#             if residue < self.tol:
#                 self.monitor.update([residue, self.total_cost, rank, 1], [])
#                 break
#
#             Z = self.M_i.dot(R[:, active_cols])
#
#             beta = - v.T @ numpy.diag(sigma_inv) @ u.T @ Q_k.T.dot(Z)
#             P = Z + P.dot(beta)
#
#             self.monitor.update([residue, self.total_cost, rank, 1], [Z, P, R])
#
#         # Sanitize the output elements
#         self._finalize()
