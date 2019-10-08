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


class RandomSubspaceError(Exception):
    """
    Exception raised when RandomSubspace object encounters specific errors.
    """


class RandomSubspaceFactory(object):
    """
    Abstract class for a RandomSubspace factory.
    """

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

    def generate(self, sampling_method: str, *args, **kwargs) -> object:
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
            raise ValueError('Sampling method {} unknown.'.format(sampling_method))

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

        # Random rows selection
        rows = [i % n for i in range(p)]
        random.shuffle(rows)

        # Random column selection
        columns = [i % k for i in range(k * (p // k + 1))]
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
