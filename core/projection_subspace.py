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

    basis = ['directions',
             'residuals',
             'precond_residuals',
             'ritz']

    def __init__(self,
                 shape: tuple,
                 lin_sys: LinearSystem,
                 M: Preconditioner = None,
                 dtype: object = numpy.float64) -> None:
        """
        Constructor of the RandomSubspaceFactory.

        :param shape: Shape of the subspaces to build.
        :param lin_sys: Linear system to build the Krylov subspace from.
        :param M: Preconditioner for a potential preconditioned Krylov subspace.
        :param dtype: Type of the subspace coefficients.
        """

        # Sanitize the shape attribute
        if not isinstance(shape, tuple) or len(shape) != 2:
            raise KrylovSubspaceError('Shape must be a tuple of the form (n, p).')

        self.shape = shape

        # Sanitize the linear system attribute
        if not isinstance(lin_sys, LinearSystem):
            raise KrylovSubspaceError('KrylovSubspace requires a LinearSystem object.')

        self.lin_sys = lin_sys

        # Sanitize the preconditioner attribute

        M = IdentityPreconditioner(lin_sys.lin_op) if M is None else M

        if not isinstance(M, Preconditioner):
            raise KrylovSubspaceError('Preconditioned Krylov Subspace requires a Preconditioner '
                                      'object.')

        self.M = M

        # Sanitize the dtype attribute
        try:
            self.dtype = numpy.dtype(dtype)
        except TypeError:
            raise KrylovSubspaceError('dtype provided not understood')

        cg = ConjugateGradient(self.lin_sys,
                               M=M,
                               x_0=None,
                               tol=1e-6,
                               buffer=self.shape[0],
                               arnoldi=True)
        cg.run()

        self.R, self.P, self.Z, self.ritz = self.process(cg)

    def process(self, cg: ConjugateGradient) -> tuple:
        """
        Process the different basis of the Krylov subspace computed during the run of the conjugate
        gradient. Aggregate the A-conjugate, M-conjugate, and M^(-1)-conjugate basis as well as the
        Ritz vectors from the tridiagonal matrix.

        :param cg: Conjugate gradient converged to process the krylov subspace built.
        """

        # Stack the descent directions, residuals and preconditioned residuals
        P = numpy.hstack(cg.output['p'])
        Z = numpy.hstack(cg.output['p'])
        R = numpy.hstack(cg.output['p'])

        # Compute the Ritz vectors from the tridiagonal matrix of the Arnoldi relation
        _, eigen_vectors = numpy.linalg.eig(cg.output['arnoldi'].todense())
        ritz_vectors = Z.dot(eigen_vectors)

        return R, P, Z, ritz_vectors

    def get(self, krylov_basis: str, *args, **kwargs) -> numpy.ndarray:
        """
        Generic method to get the different subspaces possibly
        :param krylov_basis: Name of the Krylov subspace basis required
        """

        n, k = self.shape

        if krylov_basis == 'directions':
            return self.P[:, :k]

        elif krylov_basis == 'residuals':
            return self.R[:, :k]

        elif krylov_basis == 'precond_residuals':
            return self.Z[:, :k]

        elif krylov_basis == 'ritz':
            return self.ritz[:, :k]

        else:
            raise KrylovSubspaceError('Krylov basis {} unknown.'.format(krylov_basis))


class RandomSubspaceFactory(object):
    """
    Abstract class for a RandomSubspace factory.
    """

    samplings = ['binary_sparse',
                 'gaussian_sparse',
                 'nystrom']

    def __init__(self,
                 shape: tuple,
                 dtype: object = numpy.float64,
                 sparse_tol: float = 5e-2) -> None:
        """
        Constructor of the RandomSubspaceFactory.

        :param shape: Shape of the subspaces to build.
        :param dtype: Type of the subspace coefficients.
        :param sparse_tol: Tolerance below which a subspace is considered as sparse.
        """

        # Sanitize the shape attribute
        if not isinstance(shape, tuple) or len(shape) != 2:
            raise RandomSubspaceError('Shape must be a tuple of the form (n, p).')

        self.shape = shape

        # Sanitize the dtype attribute
        try:
            self.dtype = numpy.dtype(dtype)
        except TypeError:
            raise RandomSubspaceError('dtype provided not understood')

        self.sparse_tol = sparse_tol

    def get(self, sampling_method: str, *args, **kwargs) -> object:
        """
        Generic method to generate subspaces from various distribution.

        :param sampling_method: Name of the distribution to the draw the subspace from.
        :param args: Optional arguments for distributions.
        """

        if sampling_method == 'binary_sparse':
            return self._binary_sparse(*args, **kwargs)

        elif sampling_method == 'gaussian_sparse':
            return self._gaussian_sparse(*args, **kwargs)

        elif sampling_method == 'nystrom':
            return self._nystrom(*args, **kwargs)

        else:
            raise RandomSubspaceError('Sampling method {} unknown.'.format(sampling_method))

    def _binary_sparse(self, d: float) -> scipy.sparse.csc_matrix:
        """
        Draw a subspace from the Binary Sparse distribution.

        :param d: Control the sparsity of the subspace. The subspace will contain p = k + d(n - k)
        non-zeros elements.
        """

        # Initialize subspace in lil format to allow easy update
        n, k = self.shape
        subspace = scipy.sparse.lil_matrix(self.shape)

        # Number of non-zeros elements
        p = int(k + (n - k) * d)
        p = p - (p % k)

        # Random rows selection
        rows = [i for i in range(n)]
        random.shuffle(rows)

        # Random column selection
        columns = [i % k for i in range(p)]
        random.shuffle(columns)

        for i in range(p):
            subspace[rows[i], columns[i]] = (2 * numpy.random.randint(0, 2) - 1) / numpy.sqrt(p / k)

        return subspace.tocsc()

    def _gaussian_sparse(self, d: float) -> scipy.sparse.csc_matrix:
        """
        Draw a subspace from the Gaussian Sparse distribution.

        :param d: Control the sparsity of the subspace. The subspace will contain p = k + d(n - k)
        non-zeros elements.
        """

        # Initialize subspace in lil format to allow easy update
        n, k = self.shape
        subspace = scipy.sparse.lil_matrix(self.shape)

        # Number of non-zeros elements
        p = int(k + (n - k) * d)

        # Random rows selection
        rows = [i % n for i in range(p)]
        random.shuffle(rows)

        # Random column selection
        columns = [i % k for i in range(k * (p // k + 1))]
        random.shuffle(columns)

        for i in range(p):
            subspace[rows[i], columns[i]] = numpy.random.randn()

        return subspace.tocsc()

    def _nystrom(self, lin_op: scipy.sparse.spmatrix, p: int = 10) -> numpy.ndarray:
        """
        Compute a spectral approximation of the higher singular vectors of a linear operator using
        the Nyström method. It is a stochastic method for low-rank approximation here utilized to
        generate approximate spectral information.

        :param lin_op : Sparse matrix to process the spectral approximation on.
        :param p: Over-sampling parameter meant to increase the approximation accuracy.
        """
        # Retrieve the problem dimension
        n, k = self.shape

        # Draw a Gaussian block of appropriate size
        G = numpy.random.randn(n, k + p)

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
