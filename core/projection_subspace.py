#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on July 04, 2019 at 11:14.

@author: a.scotto

Description:
"""

import pyamg
import numpy
import scipy.sparse
import scipy.linalg

from core.linear_system import LinearSystem, ConjugateGradient
from core.linear_operator import LinearOperator, MatrixOperator
from core.preconditioner import Preconditioner, IdentityPreconditioner


class RandomSubspaceError(Exception):
    """
    Exception raised when RandomSubspace object encounters specific errors.
    """


class DeterministicSubspaceError(Exception):
    """
    Exception raised when KrylovSubspace object encounters specific errors.
    """


class DeterministicSubspaceFactory(object):
    """
    Abstract class for deterministic subspace factory.
    """

    subspace_type = dict(directions='dense',
                         ritz='dense',
                         harmonic_ritz='dense',
                         amg_spectral='dense')

    def __init__(self,
                 lin_op: MatrixOperator,
                 precond: Preconditioner = None,
                 rhs: numpy.ndarray = None,
                 dtype: object = numpy.float64) -> None:
        """
        Constructor of the DeterministicSubspaceFactory.

        :param lin_op: Linear operator to base the deterministic approaches on.
        :param precond: Preconditioner for a potential preconditioned Krylov subspace.
        :param dtype: Type of the subspace coefficients.
        """

        # Sanitize the linear operator argument
        if not isinstance(lin_op, LinearOperator):
            raise DeterministicSubspaceError('Linear operator must be of type LinearOperator.')

        self.lin_op = lin_op
        n, _ = lin_op.shape

        # Sanitize the preconditioner argument
        self.precond = IdentityPreconditioner(lin_op) if precond is None else precond

        if not isinstance(self.precond, Preconditioner):
            raise DeterministicSubspaceError('Preconditioner must be of type Preconditioner.')

        # Sanitize the right-hand side argument
        self.rhs = lin_op.dot(numpy.random.randn(n, 1)) if rhs is None else rhs

        if not isinstance(self.rhs, numpy.ndarray):
            raise DeterministicSubspaceError('Right-hand side must be of type numpy.ndarray.')

        # Sanitize the dtype attribute
        try:
            self.dtype = numpy.dtype(dtype)
        except TypeError:
            raise DeterministicSubspaceError('dtype provided not understood')

        cg = ConjugateGradient(LinearSystem(self.lin_op, self.rhs),
                               M=precond,
                               x_0=None,
                               tol=1e-10,
                               maxiter=200,
                               buffer=200,
                               arnoldi=True)

        # Get Arnoldi tridiagonal matrix
        self.arnoldi = cg.output['arnoldi'].todense()

        # Get the 3 different basis computed during the run of the Conjugate Gradient
        self.R = cg.output['r']
        self.P = cg.output['p']
        self.Z = cg.output['z']

        # Get final quantities
        self.x_opt = cg.output['x_opt']
        self.final_residual = cg.output['residues'][-1]

    def get(self, subspace: str, k: int, *args) -> object:
        """
        Generic method to get the different deterministic subspaces available.

        :param subspace: Name of the deterministic subspace to build.
        :param k: Size of the subspace to build.
        """

        # Return appropriate subspace
        if subspace == 'directions':
            return self._descent_directions(k, *args)

        elif subspace == 'ritz':
            return self._ritz(k, *args)

        elif subspace == 'harmonic_ritz':
            return self._harmonic_ritz(k, *args)

        elif subspace == 'amg_spectral':
            return self._amg_spectral(k, *args)

        else:
            raise DeterministicSubspaceError('Unknown deterministic subspace name.')

    def _descent_directions(self, k: int, loc: str) -> numpy.ndarray:
        """
        Return 'k' descent directions obtained from the run of the Conjugate Gradient. Either the
        first or the latest obtained depending on the 'loc' parameter.

        :param k: Number of descent directions to return.
        :param loc: Either select the 'first' or 'last' descent directions computed.
        """

        if loc == 'first':
            descent_directions = self.P[:, :k]
        elif loc == 'last':
            descent_directions = self.P[:, -k:]
        else:
            raise ValueError('Only "first" or "last" descent directions can be provided.')

        return descent_directions

    def _ritz(self, k: int, loc: str) -> numpy.ndarray:
        """
        Compute k Ritz vectors chosen either at the upper or lower part of the spectrum.

        :param k: Number of Ritz vectors to compute.
        :param loc: Either 'upper' or 'lower' part of the spectral approximation.
        """

        # Compute the Ritz vectors from the tridiagonal matrix of the Arnoldi relation
        _, eigen_vectors = scipy.linalg.eigh(self.arnoldi)
        ritz_vectors = self.Z.dot(eigen_vectors)

        if loc == 'lower':
            ritz_vectors = ritz_vectors[:, :k]
        elif loc == 'upper':
            ritz_vectors = ritz_vectors[:, -k:]
        else:
            raise ValueError('Only "lower" or "upper" Ritz spectral approximation can be provided.')

        return ritz_vectors

    def _harmonic_ritz(self, k: int, loc: str) -> numpy.ndarray:
        """
        Compute k harmonic Ritz vectors located at the upper or lower part of the spectrum.

        :param k: Number of harmonic Ritz vectors to compute.
        :param loc: Either 'upper' or 'lower' part of the spectral approximation.
        """
        l, _ = self.arnoldi.shape
        e = numpy.zeros((l, 1))
        e[-1, 0] = 1.

        h = self.final_residual
        f = scipy.linalg.solve(self.arnoldi.T, e)

        A = self.arnoldi + h**2 * (f @ e.T)

        _, eigen_vectors = scipy.linalg.eigh(A)

        harmonic_ritz_vectors = self.Z.dot(eigen_vectors)

        if loc == 'lower':
            harmonic_ritz_vectors = harmonic_ritz_vectors[:, :k]
        elif loc == 'upper':
            harmonic_ritz_vectors = harmonic_ritz_vectors[:, -k:]
        else:
            raise ValueError('Only "lower" or "upper" harmonic Ritz spectral approximation can be '
                             'provided.')

        return harmonic_ritz_vectors

    def _amg_spectral(self, k: int, heuristic: str):

        # Sanitize heuristic argument
        if heuristic not in ['ruge_stuben', 'smoothed_aggregated', 'rootnode']:
            raise DeterministicSubspaceError('AMG heuristic {} unknown.'.format(heuristic))

        # Setup multi-grid hierarchical structure with corresponding heuristic
        if heuristic == 'ruge_stuben':
            self.amg = pyamg.ruge_stuben_solver(self.lin_op.matrix.tocsr(), max_coarse=k)

        elif heuristic == 'smoothed_aggregated':
            self.amg = pyamg.smoothed_aggregation_solver(self.lin_op.matrix.tocsr(), max_coarse=k)

        elif heuristic == 'rootnode':
            self.amg = pyamg.rootnode_solver(self.lin_op.matrix.tocsr(), max_coarse=k)

        S = self.amg.levels[0].P
        for i in range(1, len(self.amg.levels) - 1):
            S = S.dot(self.amg.levels[i].P)

        return S


class RandomSubspaceFactory(object):
    """
    Abstract class for a RandomSubspace factory.
    """

    subspace_type = dict(binary_sparse='sparse',
                         gaussian_sparse='sparse',
                         nystrom='dense')

    def __init__(self,
                 lin_op: LinearOperator,
                 dtype: object = numpy.float64) -> None:
        """
        Constructor of the RandomSubspaceFactory.

        :param lin_op: Linear operator to base the deterministic approaches on.
        :param dtype: Type of the subspace coefficients.
        """

        # Sanitize the linear operator argument
        if not isinstance(lin_op, LinearOperator):
            raise DeterministicSubspaceError('Linear operator must be of type LinearOperator.')

        self.lin_op = lin_op
        self.size, _ = lin_op.shape

        # Sanitize the dtype attribute
        try:
            self.dtype = numpy.dtype(dtype)
        except TypeError:
            raise RandomSubspaceError('dtype provided not understood')

    def get(self, sampling_method: str, k: int, *args, **kwargs) -> object:
        """
        Generic method to generate subspaces from various distribution.

        :param sampling_method: Name of the distribution to the draw the subspace from.
        :param k: Size of the subspace to build.
        """

        if sampling_method == 'binary_sparse':
            return self._binary_sparse(k)

        elif sampling_method == 'gaussian_sparse':
            return self._gaussian_sparse(k)

        elif sampling_method == 'nystrom':
            return self._nystrom(k, *args, **kwargs)

        else:
            raise RandomSubspaceError('Sampling method {} unknown.'.format(sampling_method))

    def _binary_sparse(self, k: int) -> scipy.sparse.csc_matrix:
        """
        Draw a subspace from the Binary Sparse distribution.

        :param k: Size of the subspace to build.
        """

        # Initialize subspace in dok format to allow easy update
        subspace = scipy.sparse.dok_matrix((self.size, k))

        # Draw columns index and verify corresponding rank consistency
        cols = numpy.random.randint(k, size=self.size)
        _, counts = numpy.unique(cols, return_counts=True)

        while len(counts) != k:
            cols = numpy.random.randint(k, size=self.size)
            _, counts = numpy.unique(cols, return_counts=True)

        index = numpy.arange(self.size)

        # Fill-in with coefficients in {-1, 1}
        subspace[index, cols] = (2 * numpy.random.randint(0, 2, size=self.size) - 1)

        return subspace.tocsr()

    def _gaussian_sparse(self, k: int) -> scipy.sparse.csc_matrix:
        """
        Draw a subspace from the Gaussian Sparse distribution.

        :param k: Size of the subspace to build.
        """

        # Initialize subspace in dok format to allow easy update
        subspace = scipy.sparse.dok_matrix((self.size, k))

        # Draw columns index and verify corresponding rank consistency
        cols = numpy.random.randint(k, size=self.size)
        _, counts = numpy.unique(cols, return_counts=True)

        while len(counts) != k:
            cols = numpy.random.randint(k, size=self.size)
            _, counts = numpy.unique(cols, return_counts=True)

        index = numpy.arange(self.size)

        # Fill-in with coefficients drawn from standard normal distribution
        subspace[index, cols] = numpy.random.randn(self.size)

        return subspace.tocsr()

    def _nystrom(self, k: int, lin_op: scipy.sparse.spmatrix, p: int = 10) -> numpy.ndarray:
        """
        Compute a spectral approximation of the higher singular vectors of a linear operator using
        the Nyström method. It is a stochastic method for low-rank approximation here utilized to
        generate approximate spectral information.

        :param k: Size of the subspace to build.
        :param lin_op : Sparse matrix to process the spectral approximation on.
        :param p: Over-sampling parameter meant to increase the approximation accuracy.
        """

        # Draw a Gaussian block of appropriate size
        G = numpy.random.randn(self.size, k + p)

        # Form the sample matrix Y
        Y = lin_op.dot(G)

        # Orthogonalize the columns of the sample matrix
        Q, _ = scipy.linalg.qr(Y, mode='economic')

        # Form B1 oh shape (m, k + p) and B2 of shape (k + p, k + p)
        B1 = lin_op.dot(Q)
        B2 = Q.T.dot(B1)

        # Perform a Cholesky factorization of B2 as B2 = C^T * C where C is upper triangular
        C = scipy.linalg.cholesky(B2)

        # Form F = B1 C^{-1} by solving C^T F^T = B1^T, F is of size (m,k+p)
        H = scipy.linalg.solve_triangular(C, B1.transpose(), 'T')
        F = H.transpose()

        # Perform the economical SVD decomposition of F
        U, _, _ = scipy.linalg.svd(F, full_matrices=False)

        return U[:, :k]
