#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on July 04, 2019 at 11:14.

@author: a.scotto

Description:
"""

import numpy
import random
import scipy.stats
import scipy.sparse
import scipy.linalg

from core.linear_system import LinearSystem, ConjugateGradient
from core.preconditioner import Preconditioner, IdentityPreconditioner


class RandomSubspaceError(Exception):
    """
    Exception raised when RandomSubspace object encounters specific errors.
    """


class KrylovSubspaceError(Exception):
    """
    Exception raised when KrylovSubspace object encounters specific errors.
    """


class KrylovSubspaceFactory(object):
    """
    Abstract class for a Krylov subspace factory.
    """

    krylov_type = dict(directions='dense',
                       residuals='dense',
                       precond_residuals='dense',
                       ritz='dense',
                       harmonic_ritz='dense',
                       random_xopt='sparse',
                       random_res='sparse')

    def __init__(self,
                 lin_sys: LinearSystem,
                 precond: Preconditioner = None,
                 dtype: object = numpy.float64) -> None:
        """
        Constructor of the RandomSubspaceFactory.

        :param lin_sys: Linear system to build the Krylov subspace from.
        :param precond: Preconditioner for a potential preconditioned Krylov subspace.
        :param dtype: Type of the subspace coefficients.
        """

        # Sanitize the linear system attribute
        if not isinstance(lin_sys, LinearSystem):
            raise KrylovSubspaceError('KrylovSubspace requires a LinearSystem object.')

        n, _ = lin_sys.lin_op.shape

        # Sanitize the preconditioner attribute

        precond = IdentityPreconditioner(lin_sys.lin_op) if precond is None else precond

        if not isinstance(precond, Preconditioner):
            raise KrylovSubspaceError('Preconditioned Krylov Subspace requires a Preconditioner '
                                      'object.')

        # Sanitize the dtype attribute
        try:
            self.dtype = numpy.dtype(dtype)
        except TypeError:
            raise KrylovSubspaceError('dtype provided not understood')

        cg = ConjugateGradient(lin_sys,
                               M=precond,
                               x_0=None,
                               tol=1e-6,
                               buffer=n,
                               arnoldi=True)
        cg.run()

        # Get Arnoldi tridiagonal matrix
        self.arnoldi = cg.output['arnoldi'].todense()

        # Get the 3 different basis computed during the run of the Conjugate Gradient
        self.R = cg.output['residues'][0] * numpy.hstack(cg.output['r'])
        self.P = cg.output['residues'][0] * numpy.hstack(cg.output['p'])
        self.Z = cg.output['residues'][0] * numpy.hstack(cg.output['z'])

        # Get final quantities
        self.x_opt = cg.output['x_opt']
        self.final_res = cg.output['residues'][-1]

    def get(self, krylov_subspace: str, k: int, *args) -> object:
        """
        Generic method to get the different subspaces possibly

        :param krylov_subspace: Name of the Krylov subspace basis required.
        :param k: Size of the subspace to build.
        """

        # Sanitize Krylov subspace name parameter
        if krylov_subspace not in self.krylov_type.keys():
            raise KrylovSubspaceError('Krylov subspace {} unknown.'.format(krylov_subspace))

        # Return appropriate subspace
        if krylov_subspace == 'directions':
            return self.P[:, :k]

        elif krylov_subspace == 'residuals':
            return self.R[:, :k]

        elif krylov_subspace == 'precond_residuals':
            return self.Z[:, :k]

        elif krylov_subspace == 'ritz':
            return self._ritz(k, *args)

        elif krylov_subspace == 'harmonic_ritz':
            return self._harmonic_ritz(k, *args)

        elif krylov_subspace == 'random_xopt':
            return self._random_x_opt(k)

        elif krylov_subspace == 'random_res':
            return self._random_res(k)

    def _ritz(self, k: int, loc: str) -> numpy.ndarray:
        """
        Compute k Ritz vectors located at the upper or lower part of the spectrum.

        :param k: Number of Ritz vectors to compute.
        :param loc: Either 'upper' or 'lower' part of the spectral approximation.
        """

        # Compute the Ritz vectors from the tridiagonal matrix of the Arnoldi relation
        _, eigen_vectors = scipy.linalg.eigh(self.arnoldi)
        ritz_vectors = self.Z.dot(eigen_vectors)

        if loc == 'lower':
            ritz_vectors = ritz_vectors[:, -k:]
        elif loc == 'upper':
            ritz_vectors = ritz_vectors[:, :k]
        else:
            raise ValueError('Unknown localization parameter {}'.format(loc))

        return ritz_vectors

    def _harmonic_ritz(self, k: int, loc: str) -> numpy.ndarray:
        """
        Compute k harmonic Ritz vectors located at the upper or lower part of the spectrum.

        :param k: Number of harmonic Ritz vectors to compute.
        :param loc: Either 'upper' or 'lower' part of the spectral approximation.
        """

        e = numpy.zeros((1, self.arnoldi.shape[0]))
        e[0, -1] = self.final_res

        H_bar = numpy.vstack([self.arnoldi, e])
        H_square = self.arnoldi

        A = H_bar.T @ H_bar
        B = H_square.T

        _, eigen_vectors = scipy.linalg.eigh(A, B)

        harmonic_ritz_vectors = self.Z.dot(eigen_vectors)

        if loc == 'lower':
            harmonic_ritz_vectors = harmonic_ritz_vectors[:, -k:]
        elif loc == 'upper':
            harmonic_ritz_vectors = harmonic_ritz_vectors[:, :k]
        else:
            raise ValueError('Unknown localization parameter {}'.format(loc))

        return harmonic_ritz_vectors

    def _random_x_opt(self, k: int) -> scipy.sparse.spmatrix:
        """
        Process a random subspace decomposition of linear system solution x_opt.

        :param k: Size of the random subspace to decompose x_opt on.
        """
        # Initialize subspace in lil format to allow easy update
        n = self.x_opt.size
        subspace = scipy.sparse.lil_matrix((n, k))

        # Random indexes
        index = [i for i in range(n)]
        random.shuffle(index)

        for i in range(n):
            subspace[index[i], i % k] = self.x_opt[i]

        return subspace.tocsc()

    def _random_res(self, k: int) -> scipy.sparse.spmatrix:
        """
        Process a random subspace decomposition of linear system solution x_opt.

        :param k: Size of the random subspace to decompose x_opt on.
        """
        # Initialize subspace in lil format to allow easy update
        n = self.x_opt.size
        subspace = scipy.sparse.lil_matrix((n, k))

        # Random indexes
        index = [i for i in range(n)]
        random.shuffle(index)

        for i in range(n):
            subspace[index[i], i % k] = self.R[i, -1]

        return subspace.tocsc()


class RandomSubspaceFactory(object):
    """
    Abstract class for a RandomSubspace factory.
    """

    sampling_type = dict(binary_sparse='sparse',
                         gaussian_sparse='sparse',
                         nystrom='dense')

    def __init__(self,
                 size: int,
                 dtype: object = numpy.float64,
                 sparse_tol: float = 5e-2) -> None:
        """
        Constructor of the RandomSubspaceFactory.

        :param size: Size of the subspaces to build.
        :param dtype: Type of the subspace coefficients.
        :param sparse_tol: Tolerance below which a subspace is considered as sparse.
        """

        # Sanitize the size attribute
        if not isinstance(size, int):
            raise RandomSubspaceError('Size must be an integer.')

        self.size = size

        # Sanitize the dtype attribute
        try:
            self.dtype = numpy.dtype(dtype)
        except TypeError:
            raise RandomSubspaceError('dtype provided not understood')

        self.sparse_tol = sparse_tol

    def get(self, sampling_method: str, k: int, *args, **kwargs) -> object:
        """
        Generic method to generate subspaces from various distribution.

        :param sampling_method: Name of the distribution to the draw the subspace from.
        :param k: Size of the subspace to build.
        """

        if sampling_method == 'binary_sparse':
            return self._binary_sparse(k, *args, **kwargs)

        elif sampling_method == 'gaussian_sparse':
            return self._gaussian_sparse(k, *args, **kwargs)

        elif sampling_method == 'nystrom':
            return self._nystrom(k, *args, **kwargs)

        else:
            raise RandomSubspaceError('Sampling method {} unknown.'.format(sampling_method))

    def _binary_sparse(self, k: int, d: float) -> scipy.sparse.csc_matrix:
        """
        Draw a subspace from the Binary Sparse distribution.

        :param k: Size of the subspace to build.
        :param d: Control the sparsity of the subspace. The subspace will contain p = k + d(n - k)
        non-zeros elements.
        """

        # Initialize subspace in lil format to allow easy update
        subspace = scipy.sparse.lil_matrix((self.size, k))

        # Number of non-zeros elements
        p = int(k + (self.size - k) * d)
        p = p - (p % k)

        # Random rows selection
        rows = [i for i in range(self.size)]
        random.shuffle(rows)

        for i in range(p):
            subspace[rows[i], i % k] = (2 * numpy.random.randint(0, 2) - 1) / numpy.sqrt(p / k)

        return subspace.tocsc()

    def _gaussian_sparse(self, k: int, d: float) -> scipy.sparse.csc_matrix:
        """
        Draw a subspace from the Gaussian Sparse distribution.

        :param k: Size of the subspace to build.
        :param d: Control the sparsity of the subspace. The subspace will contain p = k + d(n - k)
        non-zeros elements.
        """

        # Initialize subspace in lil format to allow easy update
        subspace = scipy.sparse.lil_matrix((self.size, k))

        # Number of non-zeros elements
        p = int(k + (self.size - k) * d)
        p = p - (p % k)

        # Random rows selection
        rows = [i for i in range(self.size)]
        random.shuffle(rows)

        for i in range(p):
            subspace[rows[i], i % k] = numpy.random.randn()

        return subspace.tocsc()

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
