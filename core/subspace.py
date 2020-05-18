#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on July 04, 2019 at 11:14.

@author: a.scotto

Description:
"""

import pyamg
import numpy
import warnings
import scipy.stats
import scipy.sparse
import scipy.linalg

from typing import Tuple, Union
from core.linop import Subspace
from utils.utils import random_surjection
from core.linsys import LinearSystem, ConjugateGradient

__all__ = ['SubspaceGenerator']

SubspaceType = Union[Subspace, numpy.ndarray, scipy.sparse.spmatrix]


class SubspaceGeneratorError(Exception):
    """
    Exception raised when KrylovSubspace object encounters specific errors.
    """


class SubspaceError(Exception):
    """
    Exception raised when KrylovSubspace object encounters specific errors.
    """


class SubspaceGenerator(object):

    output_format = dict(ritz='dense',
                         harmonic_ritz='dense',
                         amg_spectral='dense',
                         binary_sparse='sparse',
                         random_split='sparse')

    def __init__(self,
                 linear_system: LinearSystem,
                 x0: numpy.ndarray = None,
                 recycle: LinearSystem = None):
        """
        Generator for subspaces aimed at being used for deflation, recycling or projections.

        :param linear_system: Linear system to be solved to obtain Krylov subspace bases.
        :param x0: Initial guess for the linear system.
        :param recycle: Additional linear system to recycle from.
        """

        self.linear_op = linear_system.linear_op
        self.rhs = linear_system.rhs
        self.M = linear_system.M

        # Sanitize recycling argument
        if recycle is not None:
            if not isinstance(recycle, LinearSystem):
                raise SubspaceGeneratorError('Recycling argument must be a pair')

            # Retrieve information from previous system solution via Arnoldi relation
            cg = ConjugateGradient(recycle, x0=x0, store_arnoldi=True)
            self.arnoldi_matrix = cg.H
            self.arnoldi_vectors = cg.V

            self.linear_op_ = recycle.linear_op
            self.rhs_ = recycle.rhs
            self.M_ = recycle.M

        else:
            self.arnoldi_matrix = None
            self.arnoldi_vectors = None

            self.linear_op_ = None
            self.rhs_ = None
            self.M_ = None

    def get(self, subspace: str, k: int, *args, **kwargs) -> Subspace:
        """
        Generic method to get the different deterministic subspaces available.

        :param subspace: Name of the deterministic subspace to build.
        :param k: Size of the subspace to build.
        """

        # Root to method corresponding to provided subspace label
        if subspace == 'ritz':
            return self._ritz(k, *args, **kwargs)

        elif subspace == 'harmonic_ritz':
            return self._harmonic_ritz(k, *args, **kwargs)

        elif subspace == 'amg_spectral':
            return self._amg_spectral(k, *args, **kwargs)

        elif subspace == 'binary_sparse':
            return self._binary_sparse(k)

        elif subspace == 'random_split':
            return self._random_split(k)

        elif subspace == 'random_amg':
            return self._random_amg(k, *args, **kwargs)

        elif subspace == 'nystrom':
            return self._nystrom(k, **kwargs)

        else:
            raise SubspaceGeneratorError('Unknown deterministic subspace name.')

    def _ritz(self, k: int,
              select: str = 'smallest',
              return_values: bool = False) -> Union[Subspace, Tuple[Subspace, numpy.ndarray]]:
        """
        Compute k Ritz vectors and select as required the ones of interest.

        :param k: Number of Ritz vectors to compute.
        :param select: Which part of the approximate eigen-space to select, one of 'smallest', 'largest', 'extremal'.
        :param return_values: Whether to return the harmonic Ritz values.
        """
        if self.linear_op_ is None:
            raise SubspaceError('Cannot compute Ritz vectors from recycled linear system since not provided.')

        if k > self.arnoldi_matrix.shape[1]:
            warnings.warn('Required more Ritz vectors than available, hence truncation.')
            k = self.arnoldi_matrix.shape[1]

        # Compute the Ritz vectors from the tridiagonal matrix of the Arnoldi relation
        H_sq = self.arnoldi_matrix[:-1, :]

        ritz_values, vectors = scipy.linalg.eigh(H_sq)
        ritz_vectors = self.arnoldi_vectors[:, :-1].dot(vectors)
        res = self.M.dot(self.linear_op.dot(ritz_vectors)) - ritz_vectors.dot(numpy.diag(ritz_values))

        if select == 'smallest':
            indices = numpy.argsort(ritz_values)[:k]
        elif select == 'largest':
            indices = numpy.argsort(ritz_values)[-k:]
        elif select == 'extremal':
            indices = numpy.argsort(numpy.linalg.norm(res, axis=0))[:k]
        else:
            raise SubspaceError('Select must be one of ["largest", "smallest", "extremal"], received {}'.format(select))

        subspace = Subspace(ritz_vectors[:, indices])

        if return_values:
            return subspace, ritz_values[indices]
        else:
            return subspace

    def _harmonic_ritz(self, k: int,
                       select: str = 'smallest',
                       return_values: bool = False) -> Union[Subspace, Tuple[Subspace, numpy.ndarray]]:
        """
        Compute k harmonic Ritz vectors and select as required the ones of interest.

        :param k: Number of Ritz vectors to compute.
        :param select: Either 'upper' or 'lower' part of the spectral approximation.
        :param return_values: Return the harmonic Ritz values.
        """
        if self.linear_op_ is None:
            raise SubspaceError('Cannot compute harmonic Ritz vectors from recycled linear system since not provided.')

        if k > self.arnoldi_matrix.shape[1]:
            warnings.warn('Required more harmonic Ritz vectors than available, hence truncation.')
            k = self.arnoldi_matrix.shape[1]

        # Compute the Ritz vectors from the tridiagonal matrix of the Arnoldi relation
        H_sq = self.arnoldi_matrix[:-1, :]
        h_ll = self.arnoldi_matrix[-1, -1]

        el = numpy.zeros((self.arnoldi_matrix.shape[1], 1))
        el[-1, 0] = 1.
        fl = scipy.linalg.solve(H_sq, el)

        H_eq = H_sq + numpy.abs(h_ll)**2 * (fl.T @ el)

        h_ritz_values, vectors = scipy.linalg.eigh(H_eq)
        h_ritz_vectors = self.arnoldi_vectors[:, :-1].dot(vectors)
        res = self.M.dot(self.linear_op.dot(h_ritz_vectors)) - h_ritz_vectors.dot(numpy.diag(h_ritz_values))

        if select is 'smallest':
            indices = numpy.argsort(h_ritz_values)[:k]
        elif select == 'largest':
            indices = numpy.argsort(h_ritz_values)[-k:]
        elif select == 'extremal':
            indices = numpy.argsort(numpy.linalg.norm(res, axis=0))[:k]
        else:
            raise SubspaceError('Select must be one of ["largest", "smallest", "extremal"].')

        subspace = Subspace(h_ritz_vectors[:, indices])

        if return_values:
            return subspace, h_ritz_values[indices]
        else:
            return subspace

    def _amg_spectral(self, k: int,
                      heuristic: str,
                      select: str = 'smallest',
                      return_values: bool = False,
                      **kwargs) -> Union[Subspace, Tuple[Subspace, numpy.ndarray]]:
        """
        Compute approximate eigen-vectors via algebraic multi-grid architecture and then select k vectors from the
        obtained approximate eigen-space.

        :param k: Number of approximate eigen-vectors to compute.
        :param heuristic: Name of the algebraic multi-grid heuristic used to construct the hierarchy.
        :param select: Which part of the approximate eigen-space to select, one of 'smallest', 'largest', 'extremal'.
        :param return_values: Whether to return the approximate eigen-values.
        :param kwargs: complementary arguments for algebraic multi-grid construction, see PyAMG library for details.
        """
        # Sanitize heuristic argument
        if heuristic not in ['ruge_stuben', 'smoothed_aggregated', 'rootnode']:
            raise SubspaceError('Algebraic multi-grid heuristic {} unknown.'.format(heuristic))

        # Setup multi-grid hierarchical structure with corresponding heuristic
        if heuristic == 'ruge_stuben':
            self.amg = pyamg.ruge_stuben_solver(self.linear_op.matrix.tocsr(), **kwargs)

        elif heuristic == 'smoothed_aggregated':
            self.amg = pyamg.smoothed_aggregation_solver(self.linear_op.matrix.tocsr(), **kwargs)

        elif heuristic == 'rootnode':
            self.amg = pyamg.rootnode_solver(self.linear_op.matrix.tocsr(), **kwargs)

        # Retrieve coarse operator and resulting prolongation operator
        A_ = self.amg.levels[-1].A

        S = self.amg.levels[0].P
        for i in range(1, len(self.amg.levels) - 1):
            S = S.dot(self.amg.levels[i].P)

        if k > S.shape[1]:
            warnings.warn('Required more AMG approximate eigen-vectors than available, hence truncation.')
            k = S.shape[1]

        # Compute eigen information
        eig_values, vectors = scipy.linalg.eigh(A_.todense())
        eig_vectors = S.dot(vectors)
        res = self.M.dot(self.linear_op.dot(eig_vectors)) - eig_vectors.dot(numpy.diag(eig_values))

        if select is 'smallest':
            indices = numpy.argsort(eig_values)[:k]
        elif select == 'largest':
            indices = numpy.argsort(eig_values)[-k:]
        elif select == 'extremal':
            indices = numpy.argsort(numpy.linalg.norm(res, axis=0))[:k]
        else:
            raise SubspaceError('Select must be one of ["largest", "smallest", "extremal"].')

        subspace = Subspace(eig_vectors[:, indices])

        if return_values:
            return subspace, eig_values[indices]
        else:
            return subspace

    def _binary_sparse(self, k: int) -> Subspace:
        """
        Draw a subspace from the Binary Sparse distribution.

        :param k: Size of the subspace to build.
        """
        n_, _ = self.linear_op.shape

        # Initialize subspace in dok format to allow easy update
        subspace = scipy.sparse.dok_matrix((n_, k))

        # Draw columns indices
        cols = random_surjection(n_, k)
        index = numpy.arange(n_)

        # Fill-in with coefficients in {-1, 1}
        subspace[index, cols] = (2 * numpy.random.randint(0, 2, size=n_) - 1)

        return Subspace(subspace.tocsr())

    def _random_split(self, k: int, x: numpy.ndarray = None) -> Subspace:
        """
        Draw a subspace from the random split distribution.

        :param k: Size of the subspace to build.
        """
        n_, _ = self.linear_op.shape

        # Initialize subspace in dok format to allow easy update
        subspace = scipy.sparse.dok_matrix((n_, k))

        # Draw columns indices
        cols = random_surjection(n_, k)
        index = numpy.arange(n_)

        # Fill-in with coefficients of linear system's right-hand side or provided vector x
        content = self.rhs.reshape(-1) if x is None else x
        subspace[index, cols] = content

        return Subspace(subspace.tocsr())

    def _random_amg(self, k: int,
                    heuristic: str,
                    randomization: str,
                    density: float = None,
                    **kwargs) -> Subspace:
        """
        Compute a subspace in two levels, a first one obtained via an algebraic multi-grid method, and the second one of
        random type. The resulting subspace is then in the form of a product.

        :param k: Number of approximate eigen-vectors to compute.
        :param heuristic: Name of the algebraic multi-grid heuristic used to construct the hierarchy.
        :param randomization: Name of the random distribution to use for the second level projection.
        :param kwargs: complementary arguments for algebraic multi-grid construction, see PyAMG library for details.
        """
        # Sanitize heuristic argument
        if heuristic not in ['ruge_stuben', 'smoothed_aggregated', 'rootnode']:
            raise SubspaceError('Algebraic multi-grid heuristic {} unknown.'.format(heuristic))

        # Setup multi-grid hierarchical structure with corresponding heuristic
        if heuristic == 'ruge_stuben':
            self.amg = pyamg.ruge_stuben_solver(self.linear_op.matrix.tocsr(), **kwargs)

        elif heuristic == 'smoothed_aggregated':
            self.amg = pyamg.smoothed_aggregation_solver(self.linear_op.matrix.tocsr(), **kwargs)

        elif heuristic == 'rootnode':
            self.amg = pyamg.rootnode_solver(self.linear_op.matrix.tocsr(), **kwargs)

        P = Subspace(self.amg.levels[0].P)
        G = None

        _, k_ = P.shape

        if k > k_:
            warnings.warn('Random restriction size superior to coarse operator size, hence truncated.')
            k = k_

        # Sanitize random distribution argument
        if randomization not in ['gaussian']:
            raise SubspaceError('Randomization name {} unknown.'.format(heuristic))

        # Initialize subspace in dok format to allow easy update
        if randomization == 'gaussian':
            if density is not None:
                rvs = scipy.stats.norm().rvs
                G = Subspace(scipy.sparse.random(k_, k, density=density, data_rvs=rvs))
            else:
                G = Subspace(numpy.random.randn(k_, k))

        return P @ G

    def _nystrom(self, k: int, p: int = 10) -> Subspace:
        """
        Compute a spectral approximation of the higher singular vectors of a linear operator using the Nyström method.
        It is a stochastic method for low-rank approximation here utilized to generate approximate spectral information.

        :param k: Size of the subspace to build.
        :param p: Over-sampling parameter meant to increase the approximation accuracy.
        """
        n_, _ = self.linear_op.shape

        # Draw a Gaussian block of appropriate size
        G = numpy.random.randn(n_, k + p)

        # Form the sample matrix Y
        Y = self.linear_op.dot(G)

        # Orthogonalize the columns of the sample matrix
        Q, _ = scipy.linalg.qr(Y, mode='economic')

        # Form B1 oh shape (m, k + p) and B2 of shape (k + p, k + p)
        B1 = self.linear_op.dot(Q)
        B2 = Q.T.dot(B1)

        # Perform a Cholesky factorization of B2 as B2 = C^T * C where C is upper triangular
        C = scipy.linalg.cholesky(B2)

        # Form F = B1 C^{-1} by solving C^T F^T = B1^T, F is of size (m,k+p)
        H = scipy.linalg.solve_triangular(C, B1.transpose(), 'T')
        F = H.transpose()

        # Perform the economical SVD decomposition of F
        U, _, _ = scipy.linalg.svd(F, full_matrices=False)

        return Subspace(U[:, :k])
