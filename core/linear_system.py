#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on April 25, 2019 at 11:02.

@author: a.scotto

Description:
"""

import numpy
import scipy.sparse
import scipy.linalg

from core.linear_operator import LinearOperator
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
    Abstract class to model a linear system of the form Ax = b, where A is a linear operator and the
    left-hand side b, a vector of corresponding shape.
    """

    def __init__(self,
                 lin_op: LinearOperator,
                 rhs: numpy.ndarray) -> None:
        """
        Constructor of the LinearSystem class.

        :param lin_op: LinearOperator involved in the linear system.
        :param rhs: Left-hand side associated.
        """

        # Sanitize the linear operator attribute
        if not isinstance(lin_op, LinearOperator):
            raise LinearSystemError('A LinearSystem must be defined with a LinearOperator.')

        self.lin_op = lin_op

        # Sanitize the right-hand side attribute
        if not isinstance(rhs, numpy.ndarray):
            raise LinearSystemError('LinearSystem right-hand side must be numpy.ndarray')

        if rhs.shape[0] != lin_op.shape[1]:
            raise LinearSystemError('Linear map and right-hand side shapes do not match.')

        # Initialize the linear system's attributes
        self.rhs = rhs
        scaling = numpy.linalg.norm(rhs)
        self. scaling = scaling if scaling != 0. else 1.

        self.block = False if self.rhs.shape[1] == 1 else True
        self.shape = self.lin_op.shape
        self.dtype = numpy.find_common_type([self.lin_op.dtype, self.rhs.dtype], [])

    def get_residual(self, x: numpy.ndarray):
        """
        Compute the explicit residual r(x) = Ax - b.

        :param x: Vector on which the residual is computed.
        """

        return self.rhs - self.lin_op.dot(x)

    def __repr__(self):
        """
        Set the printable linear system representation in a suitable manner.
        """

        _repr = 'Linear system of shape {} with right-hand side of shape {}.'\
                .format(self.lin_op.shape, self.rhs.shape)

        return _repr


class _LinearSolver(object):
    """
    Abstract class to build linear solver, i.e. solver aimed at finding the solution of a linear
    system Ax = b.
    """

    def __init__(self,
                 lin_sys: LinearSystem,
                 x_0: numpy.ndarray = None,
                 M: Preconditioner = None,
                 tol: float = 1e-5,
                 maxiter: int = None) -> None:
        """
        Constructor of LinearSolver class.

        :param lin_sys: Linear system to be solved.
        :param x_0: Initial guess for the linear system solution.
        :param M: Preconditioner to be used to enhance solver convergence.
        :param tol: Relative tolerance threshold under which the algorithm is stopped.
        :param maxiter: Maximum number of iterations affordable before stopping the method.
        """

        # Sanitize the linear system argument
        if not isinstance(lin_sys, LinearSystem):
            raise LinearSolverError('LinearSolver requires a LinearSystem.')

        self.lin_sys = lin_sys

        # Sanitize the initial guess argument
        x_0 = numpy.zeros_like(lin_sys.rhs) if x_0 is None else x_0

        if x_0 is not None and not isinstance(x_0, numpy.ndarray):
            raise LinearSolverError('Initial guess x_0 must be a numpy.ndarray.')

        if x_0.shape != lin_sys.rhs.shape:
            raise LinearSolverError('Shapes of initial guess x_0 and right-hand side mismatch.')

        self.x_0 = numpy.copy(x_0)

        # Sanitize the preconditioner attribute
        M = IdentityPreconditioner(lin_sys.lin_op) if M is None else M

        if not isinstance(M, Preconditioner):
            raise LinearSolverError('Preconditioner must be of type Preconditioner.')

        if not (M.lin_op is self.lin_sys.lin_op):
            raise LinearSolverError('Preconditioner must be related to the linear operator '
                                    'provided in the linear system.')

        self.M = M

        # Sanitize the tolerance argument
        if not isinstance(tol, float) or tol < 0:
            raise LinearSolverError('Tolerance must be a positive real number.')

        self.tol = tol

        # Sanitize the maximum iteration number argument
        maxiter = self.x_0.size if maxiter is None else maxiter

        if not isinstance(maxiter, int) or maxiter < 0:
            raise LinearSolverError('The maximum number of iteration must be a positive integer.')

        self.maxiter = maxiter

        # Initialize the solution and the solver monitor
        self.x = numpy.zeros_like(x_0)

    def _initialize(self):
        raise LinearSolverError('LinearSolver must implement a _initialize method.')

    def _finalize(self):
        raise LinearSolverError('LinearSolver must implement a _finalize method.')

    def run(self):
        raise LinearSolverError('LinearSolver must implement a run method.')


class _SolverMonitor(object):
    """
    Abstract class for LinearSolver output class. This class is meant to be used by LinearSolver
    object to store the different quantities involved in the LinearSolver run as much as the
    historic of the residues.
    """

    def __init__(self, history: object, auxiliaries: list = None, buffer: int = False) -> None:
        """
        Constructor of SolverMonitor object.

        :param history: Either scalars or list of scalars related to LinearSolver.
        :param auxiliaries: List of LinearSolver non-scalar iterations quantities.
        :param buffer: Number of iterations quantities to store.
        """

        # Sanitize the historic data attribute
        if isinstance(history, list) and not all(numpy.isscalar(item) for item in history):
            raise LinearSolverError('History quantity must be a scalar or list of scalars.')
        if not isinstance(history, list) and not numpy.isscalar(history):
            raise LinearSolverError('History quantity must be a scalar or list of scalars.')
        
        # Initialize history attribute as list of list
        try:
            self._history = [[h_i] for h_i in history]
        except TypeError:
            self._history = [[history]]

        self.n_hist = len(self._history)

        # Sanitize the auxiliaries quantities attribute
        auxiliaries = [] if auxiliaries is None else auxiliaries
        if not isinstance(auxiliaries, list):
            raise LinearSolverError('Auxiliaries quantities must be provided as a list.')
        if not all(isinstance(a_i, numpy.ndarray) for a_i in auxiliaries):
                raise LinearSolverError('Auxiliary quantity must be of numpy.ndarray type.')

        self._auxiliaries = auxiliaries
        self.n_aux = len(auxiliaries)

        # Sanitize the buffer attribute
        if buffer and not isinstance(buffer, int) and not buffer >= 0:
            raise LinearSolverError('Argument buffer must be either "False" or positive integer.')
        
        # If buffer then turn auxiliaries into list so as to store them
        if buffer:
            for i in range(len(self._auxiliaries)):
                self._auxiliaries[i] = [self._auxiliaries[i]]

        self._buffer = 0 if not buffer else buffer
        self.n_it = 0

    def update(self, history, auxiliaries: list) -> None:
        """
        Update the SolverMonitor history and auxiliaries attributes. This method is called when 
        the update of the different quantities have been done in the LinearSolver, and it updates
        buffer content. 

        :param history: History quantities to add to the SolverMonitor historic.
        :param auxiliaries: Auxiliaries quantity to add/replace in the SolverMonitor auxiliaries 
        list.
        """
        
        history = [history] if not isinstance(history, list) else history
        
        # Check that the number of auxiliary data is consistent through iterations
        if len(auxiliaries) != self.n_aux and len(auxiliaries) != 0:
            raise LinearSolverError('Updating a different number of auxiliaries than expected.')
        
        # Add or replace auxiliaries quantities depending on the buffer size
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

        # Check that the number of historic data is consistent through iterations
        if len(history) != self.n_hist:
            raise LinearSolverError('Updating a different number of history items than expected.')
        
        # Add historic quantities to the solver history
        for i in range(len(history)):
            if not numpy.isscalar(history[i]):
                raise LinearSolverError('History items must be scalars.')

            self._history[i].append(history[i])
        
        # New iteration processed
        self.n_it += 1

    def get_previous(self, index: int = -1) -> list:
        """
        Method to access previous auxiliaries quantities computed and stored in the buffer.
        :param index: Number of iteration back to retrieve from.
        """
        
        # Explore the buffer if there is one, else get the last quantities updated
        if self._buffer:
            previous_auxiliary = []
            for aux_i in self._auxiliaries:
                previous_auxiliary.append(aux_i[index])
        else:
            previous_auxiliary = self._auxiliaries

        return previous_auxiliary

    def get_auxiliaries(self) -> object:
        """
        Arrange the auxiliaries in a suitable manner before returning.
        """
        ret = []
        if not self._buffer:
            return self._auxiliaries
        
        # If the buffer is not empty, go through it
        for i in range(self.n_aux):
            aux_history = self._auxiliaries[i]
            ret.append(aux_history)
        
        # Return a list if necessary
        if self.n_aux == 1:
            return ret[0]
        else:
            return ret.__iter__()

    def get_history(self) -> object:
        """
        Arrange the historic data in a suitable manner before returning.
        """
        ret = []
        
        # Stack the historic data
        for i in range(self.n_hist):
            history = numpy.stack(self._history[i])
            ret.append(history.T)

        # Return a list if necessary
        if self.n_hist == 1:
            return ret[0]
        else:
            return ret.__iter__()


class ConjugateGradient(_LinearSolver):
    """
    Abstract class to implement the (Preconditioned) Conjugate Gradient algorithm.
    """

    def __init__(self,
                 lin_sys: LinearSystem,
                 x_0: numpy.ndarray = None,
                 M: Preconditioner = None,
                 tol: float = 1e-5,
                 maxiter: int = None,
                 buffer: int = None,
                 arnoldi: bool = False) -> None:
        """
        Constructor of the ConjugateGradient class.

        :param lin_sys: Linear system to be solved.
        :param x_0: Initial guess for the linear system solution.
        :param M: Preconditioner to be used to enhance solver convergence.
        :param tol: Relative tolerance threshold under which the algorithm is stopped.
        :param maxiter: Maximum number of iterations affordable before stopping the method.
        :param buffer: Size of the buffer memory.
        :param arnoldi: Whether or not storing the Arnoldi relation elements.
        """

        # Instantiate linear solver attributes
        super().__init__(lin_sys,
                         x_0=x_0,
                         M=M,
                         tol=tol,
                         maxiter=maxiter)

        # Ensure the linear system lies in the range of the conjugate gradient method
        if not hasattr(lin_sys.lin_op, 'def_pos'):
            raise LinearSolverError('Conjugate Gradient only apply to s.d.p linear operators.')

        if lin_sys.lin_op.shape[0] != lin_sys.lin_op.shape[1]:
            raise LinearSolverError('Conjugate Gradient only apply to squared operators.')

        if lin_sys.block:
            raise LinearSolverError('Conjugate Gradient only apply to simple right-hand side.')

        # Sanitize the buffer size argument
        buffer = 0 if buffer is None else buffer

        if not isinstance(buffer, int) or buffer < 0:
            raise LinearSolverError('The buffer size must be a positive integer.')

        self.buffer = buffer

        # Initialize computational costs attributes
        self.total_cost = 0
        self.iteration_cost = self._iteration_cost()

        # Sanitize the Arnoldi relation argument
        self.arnoldi = dict(beta_k=list(), d_k=list(), rho_k=list()) if arnoldi else None

        # Conjugate Gradient method run
        self.monitor = self._initialize()
        self._run()
        self._finalize()

    def _iteration_cost(self) -> float:
        """
        Details the FLOPs counting for one iteration of the Conjugate Gradient.
        """

        n, _ = self.lin_sys.rhs.shape

        total_cost = self.lin_sys.lin_op.apply_cost     # q_k = A * p_k
        total_cost += 4*n + 1                           # alpha_k
        total_cost += 2*n                               # x_k
        total_cost += 2*n                               # r_k
        total_cost += 2*n                               # ||r_k||
        total_cost += self.M.apply_cost               # z_k
        total_cost += 4*n + 1                           # beta
        total_cost += 2*n                               # p_k

        return total_cost

    def _initialize(self) -> _SolverMonitor:
        """
        Initialization of the Conjugate Gradient method. Namely compute the initial residue, descent
        direction, and preconditioned residual.
        """
        
        # Initialize the algorithm with scaled residual r_0, then z_0 and p_0
        r = self.lin_sys.get_residual(self.x_0) / self.lin_sys.scaling
        z = self.M.apply(r)
        p = numpy.copy(z)
        
        # Aggregate as auxiliaries quantities
        auxiliaries = [z, p, r]
        
        # Compute initial cost and residual norm
        residue = numpy.linalg.norm(r)
        cost = self.lin_sys.lin_op.apply_cost + self.M.apply_cost
        
        # Aggregate as history quantities
        history = [residue, cost]
        
        # Scale the tolerance and update total cost
        self.tol *= residue
        self.total_cost += cost

        return _SolverMonitor(history, auxiliaries=auxiliaries, buffer=self.buffer)

    def _run(self) -> None:
        """
        Run the Conjugate Gradient algorithm.
        """
        
        for k in range(self.maxiter):
            # Get the previous iterates quantities
            z, p, r = self.monitor.get_previous()
            
            # Process the recurrence relations of the Conjugate Gradient
            q_k = self.lin_sys.lin_op.dot(p)
            rho_k = z.T.dot(r)
            d_k = p.T.dot(q_k)

            alpha = rho_k / d_k
            self.x += alpha * p
            r_k = r - alpha * q_k

            residue = numpy.linalg.norm(r_k)
            self.total_cost += self.iteration_cost

            # Break whenever the tolerance is reached
            if residue < self.tol:

                # Store the elements required to compute the Arnoldi matrix if required
                if isinstance(self.arnoldi, dict):
                    self.arnoldi['d_k'].append(float(d_k))
                    self.arnoldi['rho_k'].append(float(1 / rho_k**0.5))

                self.monitor.update([residue, self.total_cost], [])
                break

            z_k = self.M.apply(r_k)
            beta = z_k.T.dot(r_k) / rho_k
            p_k = z_k + beta * p

            # Update the historic and vectors quantities with their new values
            self.monitor.update([residue, self.total_cost], [z_k, p_k, r_k])

            # Store the elements required to compute the Arnoldi matrix if required
            if isinstance(self.arnoldi, dict):
                self.arnoldi['d_k'].append(float(d_k))
                self.arnoldi['rho_k'].append(float(1 / rho_k**0.5))
                self.arnoldi['beta_k'].append(float(-beta))

    def _finalize(self) -> None:
        """
        Finalize the Conjugate Gradient algorithm by post-processing the data potentially aggregated
        during the run and making up the output dictionary gathering all the final values.
        """

        # Get residuals and iterations cost and scale back the residuals
        residues, cost = self.monitor.get_history()
        residues *= self.lin_sys.scaling

        # Get the available auxiliaries, depending on the buffer content
        z, p, r = self.monitor.get_auxiliaries()

        arnoldi = None

        # Compute the Arnoldi tridiagonal matrix coefficient if it was required
        if isinstance(self.arnoldi, dict):
            # Remove last beta_k in case convergence was not reached
            if len(self.arnoldi['d_k']) == len(self.arnoldi['beta_k']):
                del self.arnoldi['beta_k'][-1]

            D = scipy.sparse.diags(self.arnoldi['d_k'])
            N = scipy.sparse.diags(self.arnoldi['rho_k'])
            B = scipy.sparse.eye(self.monitor.n_it) + scipy.sparse.diags(self.arnoldi['beta_k'],
                                                                         offsets=1)

            arnoldi = N @ B.T @ D @ B @ N

        # Make up the output dictionary with all final values
        output = dict(x_opt=self.x_0 + self.x * self.lin_sys.scaling,
                      n_iterations=self.monitor.n_it,
                      residues=residues,
                      arnoldi=arnoldi,
                      cost=cost,
                      z=self.lin_sys.scaling * numpy.hstack(z),
                      p=self.lin_sys.scaling * numpy.hstack(p),
                      r=self.lin_sys.scaling * numpy.hstack(r))

        self.output = output

    def __repr__(self):
        """
        Provide a readable report of the Conjugate Gradient algorithm results.
        """

        n_iterations = self.output['n_iterations']
        final_residual = self.output['residues'][-1]
        residual_decrease = self.output['residues'][-1] / self.output['residues'][0]

        _repr = 'Conjugate Gradient run of: {:4} iteration(s)  |  '.format(n_iterations)
        _repr += 'Final absolute 2-norm residual = {:1.4e}  |  '.format(final_residual)
        _repr += 'Relative 2-norm residual reduction = {:1.4e}'.format(residual_decrease)

        return _repr


class BlockConjugateGradient(_LinearSolver):
    """
    Abstract class to implement the Block Conjugate Gradient method with possible use of
    preconditioner.
    """

    def __init__(self,
                 lin_sys: LinearSystem,
                 x_0: numpy.ndarray = None,
                 M: Preconditioner = None,
                 buffer: int = 0,
                 tol: float = 1e-5,
                 rank_tol: float = 1e-5):
        """
        Constructor of the ConjugateGradient class.

        :param lin_sys: LinearSystem to solve.
        :param x_0: Initial guess of the linear system.
        :param M: Preconditioner to use during the resolution.
        :param tol: Relative tolerance to achieve before stating convergence.
        :param buffer: Size of the buffer memory.
        :param rank_tol: Maximum ratio of extrema singular values accepted to consider full rank.
        """

        # Sanitize the linear system attribute
        if not hasattr(lin_sys.lin_op, 'def_pos'):
            raise LinearSolverError('Block Conjugate Gradient only apply to s.d.p linear map.')

        if lin_sys.lin_op.shape[0] != lin_sys.lin_op.shape[1]:
            raise LinearSolverError('Block Conjugate Gradient only apply to square problems.')

        if not lin_sys.block:
            raise LinearSolverError('Block Conjugate Gradient only apply to block right-hand side.')

        self.buffer = buffer
        self.total_cost = 0
        self.rank_tol = rank_tol

        super().__init__(lin_sys, x_0, M, tol, lin_sys.shape[0])

        # Sanitize the preconditioner attribute
        if not isinstance(self.M_i, Preconditioner):
            raise LinearSolverError('Conjugate Gradient can handle only one preconditioner.')

    def _initialize(self):
        """
        Initialization of the Block Conjugate Gradient method. Namely compute the initial block
        residue R_0, block descent direction P_0, and block preconditioned residual Z_0.
        """
        n, k = self.x_0.shape

        # Initialize the algorithm with scaled residual R_0, then Z_0 and P_0
        R = self.lin_sys.get_residual(self.x_0) / self.lin_sys.scaling

        # Detect rank deficiency in the initial block residue with QR-SVD
        r = scipy.linalg.qr(R, mode='r')[0]
        s = scipy.linalg.svd(r, compute_uv=False)
        effective_rank = numpy.sum(s * (1 / s[0]) > self.rank_tol)
        eps_rank = s[-1] / s[0]

        Z = self.M_i.apply(R[:, :effective_rank])
        P = numpy.copy(Z)

        # Aggregate as auxiliaries quantities
        auxiliaries = [Z, P, R]

        # Initial residual in max norm
        residue = max(numpy.linalg.norm(R[:, :effective_rank], axis=0))
        cost = k * (self.lin_sys.lin_op.apply_cost + self.M_i.apply_cost)
        cost += 2*n*k**2 + 2*k**3

        # Aggregate as history quantities
        history = [residue, cost, effective_rank, eps_rank]

        # Scale the tolerance and update total cost
        self.tol *= residue
        self.total_cost += cost

        return _SolverMonitor(history, auxiliaries=auxiliaries, buffer=self.buffer)

    def _finalize(self) -> None:
        """
        Finalize the Block Conjugate Gradient algorithm by post-processing the data aggregated
        during the run and making up the output dictionary gathering all the final values.
        """

        # Get residuals and iterations cost and scale back the residuals
        residues, cost, rank, eps_rank = self.monitor.get_history()
        residues *= numpy.linalg.norm(self.lin_sys.scaling)

        # Get the available auxiliaries, depending on the buffer content
        Z, P, R = self.monitor.get_auxiliaries()

        # Make up the report line to print
        report = self.monitor.report('Block Conjugate Gradient', residues[0], residues[-1])

        # Make up the output dictionary with all final values
        output = dict(report=report,
                      x_opt=self.x_0 + self.x * self.lin_sys.scaling,
                      n_iterations=self.monitor.n_it,
                      residues=residues,
                      cost=cost,
                      z=Z,
                      p=P,
                      r=R)

        self.output = output

    def _iteration_cost(self, alpha) -> float:
        """
        Details the Flops counting for one iteration of the Block Conjugate Gradient.

        :param alpha: Matrix equal to (P^T * A * P)^(-1) * (Z^T * R) which size gives the number
        of not yet converged columns and rank deficiency.
        """

        n, k = self.lin_sys.rhs.shape
        r, c = alpha.shape
        total_cost = 0

        total_cost += self.lin_sys.lin_op.apply_cost * r    # Q_k
        total_cost += 2*n*r**2                              # delta
        total_cost += 26*r**3                               # SVD of delta
        total_cost += r                                     # s_inv
        total_cost += 2*(r*n*c + 3*r*r*c)                   # alpha
        total_cost += n*c + 2*r*n*c                         # self.x
        total_cost += n*c + 2*r*n*c                         # R_k
        total_cost += 2*n*r**2                              # residue
        total_cost += self.M_i.apply_cost                   # Z_k
        total_cost += 2*(r*n*r + 3*r*r*k)                   # beta
        total_cost += 2*n*r*r + n*r                         # P_k

        return total_cost

    def run(self, verbose=False):
        """
        Method to actually run the Block Conjugate Gradient algorithm. At of the run the end, it
        update its own output attribute with the final values found.
        """

        for k in range(self.maxiter):
            # Retrieve last iterates quantities
            Z, P, R = self.monitor.get_previous()

            Q_k = self.lin_sys.lin_op.dot(P)

            # SVD decomposition of P^T*A*P
            delta = P.T.dot(Q_k)
            u, sigma, v = numpy.linalg.svd(delta)

            # Determination of numerical rank with given tolerance
            rank = numpy.sum((sigma > sigma[0] * self.rank_tol))

            # Pseudo inverse singular values
            sigma_inv = numpy.zeros_like(sigma)
            sigma_inv[:rank] = 1 / sigma[:rank]

            # Spot not yet converged columns to determine the remaining active ones
            residues = numpy.linalg.norm(R, axis=0)
            active_cols = residues > self.tol

            # Computation of A-conjugation corrector alpha
            alpha = v.T @ numpy.diag(sigma_inv) @ u.T @ Z.T.dot(R[:, active_cols])

            # Update of active iterate
            self.x[:, active_cols] += P.dot(alpha)
            R[:, active_cols] -= Q_k.dot(alpha)

            # Update historic data
            residue = max(residues)
            self.total_cost += self._iteration_cost(alpha)

            # Break whenever max norm tolerance is reached
            if residue < self.tol:
                self.monitor.update([residue, self.total_cost, rank, 1], [])
                break

            Z = self.M_i.apply(R[:, active_cols])

            beta = - v.T @ numpy.diag(sigma_inv) @ u.T @ Q_k.T.dot(Z)
            P = Z + P.dot(beta)

            self.monitor.update([residue, self.total_cost, rank, 1], [Z, P, R])

        # Sanitize the output elements
        self._finalize()
