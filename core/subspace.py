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
import scipy.sparse.linalg

from typing import Union
from core.algebra import *
from utils.utils import random_surjection
from core.linsolve import LinearSystem, ConjugateGradient

__all__ = ['SubspaceGenerator', 'BinarySparse', 'RandomSplit']

MatrixType = Union[numpy.ndarray, scipy.sparse.spmatrix]
SubspaceType = Union[Subspace, numpy.ndarray, scipy.sparse.spmatrix]


class SubspaceGeneratorError(Exception):
    """
    Exception raised when KrylovSubspace object encounters specific errors.
    """


class SubspaceError(Exception):
    """
    Exception raised when KrylovSubspace object encounters specific errors.
    """


class _SubspaceGenerator(object):

    def get(self, *args, **kwargs) -> Subspace:
        """
        Method to create a subspace from the subspace generator, given an arbitrary number of arguments.
        """
        raise NotImplementedError('Method get not implemented for class objects {}.'.format(self.__class__))

    def cost(self, k: int, *args, **kwargs) -> float:
        """
        Compute the computational cost of a dot-product by the subspace as a block of columns, that is when acting as a
        prolongation operator from R^k to R^n.

        :param k: Number of vectors in the subspace to compute the dot-product from.
        """
        raise NotImplementedError('Method cost not implemented for class objects {}.'.format(self.__class__))

    def rcost(self, k: int, *args, **kwargs) -> float:
        """
        Compute the computational cost of a dot-product by the subspace as a block of rows, that is when acting as a
        restriction operator from R^n to R^k.

        :param k: Number of vectors in the subspace to compute the adjoint dot-product from.
        """
        raise NotImplementedError('Method rcost not implemented for class objects {}.'.format(self.__class__))


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

        self.linear_system = linear_system
        self.x0 = x0

        self.linear_op = linear_system.linear_op
        self.rhs = linear_system.rhs
        self.M = linear_system.M

        # Sanitize recycling argument
        if recycle is not None:
            if not isinstance(recycle, LinearSystem):
                raise SubspaceGeneratorError('Recycling argument must be an instance of LinearSystem.')

            # Retrieve information from previous system solution via Arnoldi relation
            cg = ConjugateGradient(recycle, x0=x0, maxiter=150, store_arnoldi=True)
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

    def get(self, subspace: str) -> _SubspaceGenerator:
        """
        Generic method to get the different deterministic subspaces available.

        :param subspace: Name of the deterministic subspace to build.
        """

        # Root to method corresponding to provided subspace label
        if subspace == 'ritz':
            # return self._harmonic_ritz(k, *args, **kwargs)
            return Ritz(self.M @ self.linear_op, self.arnoldi_matrix, self.arnoldi_vectors)

        elif subspace == 'harmonic_ritz':
            return HarmonicRitz(self.M @ self.linear_op, self.arnoldi_matrix, self.arnoldi_vectors)

        elif subspace == 'eigen_vectors':
            return EigenVectors(self.linear_op, self.M)

        elif subspace == 'binary_sparse':
            return BinarySparse(self.rhs.size)

        elif subspace == 'random_split':
            return RandomSplit(self.linear_system, self.x0)

        elif subspace == 'random_amg':
            return RandomAMG(self.linear_op)

        elif subspace == 'nystrom':
            return Nystrom(self.linear_op)

        elif subspace == 'multi_level_rs':
            return MultiLevelRandomSplit(self.linear_system, self.x0)

        else:
            raise SubspaceGeneratorError('Unknown deterministic subspace name.')


class Ritz(_SubspaceGenerator):
    def __init__(self, linear_op: LinearOperator, arnoldi_matrix: MatrixType, arnoldi_vectors: numpy.ndarray):
        """
        Generator of Ritz subspace, that is linear subspace made up of Ritz vectors.

        :param linear_op: Linear operator from which Arnoldi relation is related.
        :param arnoldi_matrix: Matrix of the Arnoldi relation obtained via Krylov method.
        :param arnoldi_vectors: Vectors of the Arnoldi relation obtained via Krylov method.
        """
        # Sanitize arguments
        if not isinstance(linear_op, LinearOperator):
            raise SubspaceGeneratorError('Linear operator must be an instance of LinearOperator.')

        self.linear_op = linear_op

        if not isinstance(arnoldi_matrix, (numpy.ndarray, scipy.sparse.spmatrix)):
            raise SubspaceGeneratorError('Arnoldi matrix must be of matrix-like type.')

        if not isinstance(arnoldi_vectors, numpy.ndarray):
            raise SubspaceGeneratorError('Arnoldi vectors must be numpy.ndarray.')

        # Process the spectral analysis and compute Ritz information
        values, vectors = scipy.linalg.eigh(arnoldi_matrix[:-1, :])
        self.ritz_values = values
        self.ritz_vectors = arnoldi_vectors[:, :-1].dot(vectors)

        self.residues_norms = None

    def get(self, k: int, select: str = 'sm') -> Subspace:
        """
        Method to return a selection of k Ritz vectors. The selection is one of:
            * 'sm' for the Ritz vectors associated to Ritz values of Smallest Magnitude
            * 'lm' for the Ritz vectors associated to Ritz values of Largest Magnitude
            * 'sr' for the Ritz vectors leading to Smallest Residues
            * 'lr' for the Ritz vectors leading to Largest Residues

        :param k: Number of vectors making up the expected subspace.
        :param select: Criterion for the selection of the vectors.
        """
        if k > self.ritz_values.size:
            warnings.warn('Required more Ritz vectors than available, hence truncated from {} down to {}.'
                          .format(k, self.ritz_values.size))
            k = self.ritz_values.size

        if select in ['sm', 'lm']:
            if select == 'sm':
                selection = self.ritz_vectors[:, :k]
            else:
                selection = self.ritz_vectors[:, -k:]

        elif select in ['sr', 'lr']:
            if self.residues_norms is None:
                residues = self.linear_op.dot(self.ritz_vectors) - self.ritz_vectors @ numpy.diag(self.ritz_values)
                self.residues_norms = numpy.linalg.norm(residues, axis=0)

            sorted_residues_norms = numpy.argsort(self.residues_norms)

            if select == 'sr':
                indices = sorted_residues_norms[:k]
            else:
                indices = sorted_residues_norms[-k:]

            selection = self.ritz_vectors[:, indices]

        else:
            raise SubspaceError('Select must be one of "sm", "lm", "sr" or "lr", received {}'.format(select))

        return Subspace(selection)

    def cost(self, k: int, *args, **kwargs):
        return 2. * self.ritz_vectors.shape[0] * k

    def rcost(self, k: int, *args, **kwargs):
        return 2. * self.ritz_vectors.shape[0] * k


class HarmonicRitz(_SubspaceGenerator):
    def __init__(self, linear_op: LinearOperator, arnoldi_matrix: MatrixType, arnoldi_vectors: numpy.ndarray):
        """
        Generator of Harmonic Ritz subspace, that is linear subspace made up of Harmonic Ritz vectors.

        :param linear_op: Linear operator from which Arnoldi relation is related.
        :param arnoldi_matrix: Matrix of the Arnoldi relation obtained via Krylov method.
        :param arnoldi_vectors: Vectors of the Arnoldi relation obtained via Krylov method.
        """
        # Sanitize arguments
        if not isinstance(linear_op, LinearOperator):
            raise SubspaceGeneratorError('Linear operator must be an instance of LinearOperator.')

        self.linear_op = linear_op

        if not isinstance(arnoldi_matrix, (numpy.ndarray, scipy.sparse.spmatrix)):
            raise SubspaceGeneratorError('Arnoldi matrix must be of matrix-like type.')

        if not isinstance(arnoldi_vectors, numpy.ndarray):
            raise SubspaceGeneratorError('Arnoldi vectors must be numpy.ndarray.')

        # Process the spectral analysis and compute harmonic Ritz information
        H_sq = arnoldi_matrix[:-1, :]
        h_ll = arnoldi_matrix[-1, -1]

        el = numpy.zeros((arnoldi_matrix.shape[1], 1))
        el[-1, 0] = 1.
        fl = scipy.linalg.solve(H_sq, el)

        H_eq = H_sq + numpy.abs(h_ll)**2 * (fl.T @ el)

        values, vectors = scipy.linalg.eigh(H_eq)
        self.harmonic_ritz_values = values
        self.harmonic_ritz_vectors = arnoldi_vectors[:, :-1].dot(vectors)

        self.residues_norms = None

    def get(self, k: int, select: str = 'sm') -> Subspace:
        """
        Method to return a selection of k harmonic Ritz vectors. The selection is one of:
            * 'sm' for the harmonic Ritz vectors associated to harmonic Ritz values of Smallest Magnitude
            * 'lm' for the harmonic Ritz vectors associated to harmonic Ritz values of Largest Magnitude
            * 'sr' for the harmonic Ritz vectors leading to Smallest Residues
            * 'lr' for the harmonic Ritz vectors leading to Largest Residues

        :param k: Number of vectors making up the expected subspace.
        :param select: Criterion for the selection of the vectors.
        """
        if k > self.harmonic_ritz_values.size:
            warnings.warn('Required more harmonic Ritz vectors than available, hence truncated from {} down to {}.'
                          .format(k, self.harmonic_ritz_values.size))
            k = self.harmonic_ritz_values.size

        if select in ['sm', 'lm']:
            if select == 'sm':
                selection = self.harmonic_ritz_vectors[:, :k]
            else:
                selection = self.harmonic_ritz_vectors[:, -k:]

        elif select in ['sr', 'lr']:
            if self.residues_norms is None:
                residues = self.linear_op.dot(self.harmonic_ritz_vectors) - \
                           self.harmonic_ritz_vectors @ numpy.diag(self.harmonic_ritz_values)
                self.residues_norms = numpy.linalg.norm(residues, axis=0)

            sorted_residues_norms = numpy.argsort(self.residues_norms)

            if select == 'sr':
                indices = sorted_residues_norms[:k]
            else:
                indices = sorted_residues_norms[-k:]

            selection = self.harmonic_ritz_vectors[:, indices]

        else:
            raise SubspaceError('Select must be one of "sm", "lm", "sr" or "lr", received {}'.format(select))

        return Subspace(selection)

    def cost(self, k: int, *args, **kwargs):
        return 2. * self.harmonic_ritz_vectors.shape[0] * k

    def rcost(self, k: int, *args, **kwargs):
        return 2. * self.harmonic_ritz_vectors.shape[0] * k


class EigenVectors(_SubspaceGenerator):
    def __init__(self, linear_op: LinearOperator, M: Preconditioner):
        """
        Generator of eigen vectors based on the LOBPCG method.

        :param linear_op: Linear operator from which to compute the eigen vectors.
        :param M: Preconditioner for the LOBPCG method.
        """
        # Sanitize arguments
        if not isinstance(linear_op, LinearOperator):
            raise SubspaceGeneratorError('Linear operator must be an instance of LinearOperator.')

        self.linear_op = linear_op

        if not isinstance(M, Preconditioner):
            raise SubspaceGeneratorError('Preconditioner must be an instance of Preconditioner.')

        self.M = M
        self.low_end_vectors = None
        self.low_end_values = None
        self.high_end_vectors = None
        self.high_end_values = None

    def get(self, k: int, select: str = 's', tol: float = None) -> Subspace:
        """
        Method to return a selection of k eigen vectors. The selection is one of:
            * 's' for the k eigen vectors associated to the smallest eigen values
            * 'l' for the k eigen vectors associated to the largest eigen values

        :param k: Number of vectors making up the expected subspace.
        :param select: Criterion for the selection of the vectors.
        :param tol: Tolerance in the convergence of the LOBPCG method.
        """

        if select == 's':
            if self.low_end_vectors is not None and self.low_end_vectors.shape[0] > k:
                selection = self.low_end_vectors[:, :k]
            else:
                self.low_end_vectors = numpy.random.rand(self.linear_op.shape[0], k)
                output = scipy.sparse.linalg.lobpcg(self.linear_op,
                                                    self.low_end_vectors,
                                                    M=self.M,
                                                    tol=tol,
                                                    maxiter=None,
                                                    largest=False)

                self.low_end_values = output[0]
                self.low_end_vectors = output[1]
                selection = self.low_end_vectors[:, :k]

        elif select == 'l':
            if self.high_end_vectors is not None and self.high_end_vectors.shape[0] > k:
                selection = self.high_end_vectors[:, :k]
            else:
                self.high_end_vectors = numpy.random.rand(self.linear_op.shape[0], k)
                output = scipy.sparse.linalg.lobpcg(self.linear_op,
                                                    self.high_end_vectors,
                                                    M=self.M,
                                                    tol=tol,
                                                    maxiter=None,
                                                    largest=True)

                self.high_end_values = output[0]
                self.high_end_vectors = output[1]
                selection = self.high_end_vectors[:, :k]

        else:
            raise SubspaceGeneratorError('Selection of eigen vectors must be either "s" or "l".')

        return Subspace(selection)

    def cost(self, k: int, *args, **kwargs):
        return 2. * self.linear_op.shape[0] * k

    def rcost(self, k: int, *args, **kwargs):
        return 2. * self.linear_op.shape[0] * k


class Nystrom(_SubspaceGenerator):
    def __init__(self, linear_op: LinearOperator):
        """
        Generator of Nyström subspace, i.e. a random subspace acting an approximation of the linear subspace generated
        by the leading singular vectors of given linear operator.

        :param linear_op: Linear operator to approximate eigen space of.
        """
        # Sanitize arguments
        if not isinstance(linear_op, LinearOperator):
            raise SubspaceGeneratorError('Linear operator must be an instance of LinearOperator.')

        self.linear_op = linear_op

    def get(self, k: int, p: int = 10) -> Subspace:
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

    def cost(self, k: int, *args, **kwargs):
        return 2. * self.linear_op.shape[0] * k

    def rcost(self, k: int, *args, **kwargs):
        return 2. * self.linear_op.shape[0] * k


class BinarySparse(_SubspaceGenerator):
    def __init__(self, n: int):
        """
        Generator of binary sparse subspace, that is a distribution of very sparse random column block.

        :param n: Dimension of the vector space.
        """
        # Sanitize arguments
        if not isinstance(n, int):
            raise SubspaceGeneratorError('Dimension of the vector space must ba natural number.')

        self.n = n

    def get(self, k: int) -> Subspace:
        """
        Draw a subspace from the binary sparse distribution of dimension k. The block vector is made up of coefficient
        among {-1, 1}.

        :param k: Number of vectors making up the expected subspace.
        """
        # Initialize subspace in dok format to allow easy update
        subspace = scipy.sparse.dok_matrix((self.n, k))

        # Draw columns indices
        cols = random_surjection(self.n, k)
        index = numpy.arange(self.n)

        # Fill-in with coefficients in {-1, 1}
        subspace[index, cols] = (2 * numpy.random.randint(0, 2, size=self.n) - 1)

        return Subspace(subspace.tocsr())

    def cost(self, k: int, *args, **kwargs):
        return self.n

    def rcost(self, k: int, *args, **kwargs):
        return 2 * self.n


class RandomSplit(_SubspaceGenerator):
    def __init__(self, linear_system: LinearSystem, x0: numpy.ndarray = None):
        """
        Generator of random split subspace, that is a distribution of very sparse random column block.

        :param linear_system: Linear system from which the initial residual is extracted.
        :param x0: Initial guess to compute the initial residual.
        """
        # Sanitize arguments
        if not isinstance(linear_system, LinearSystem):
            raise SubspaceGeneratorError('Linear system must be an instance of LinearSystem.')

        if x0 is not None and not isinstance(x0, numpy.ndarray):
            raise SubspaceGeneratorError('Initial guess must be a numpy.ndarray.')

        self.r0 = linear_system.rhs if x0 is None else linear_system.get_residual(x0)
        self.d = linear_system.linear_op.mat.diagonal()

    def get(self, k: int, x: numpy.ndarray, seed: int =None) -> Subspace:
        """
        Draw a subspace from the random split distribution of dimension k. The subspace is by default generated from the
        initial residual r0 of the linear system, but can be constructed from any n-dimensional vector x.

        :param k: Number of vectors making up the expected subspace.
        :param x: Vector to optionally build the subspace from.
        """
        x = x.reshape(-1, 1) if x.ndim == 1 else x
        n, d = x.shape

        if seed is not None:
            numpy.random.seed(seed)

        # Initialize subspace in dok format to allow easy update
        subspace = scipy.sparse.dok_matrix((n, k * d))

        # Draw columns indices
        cols = random_surjection(n, k)
        rows = numpy.arange(n)

        # Fill-in with coefficients of linear system's right-hand side or provided vector x
        for i in range(d):
            subspace[rows, (k * i) + cols] = x[:, i].reshape(-1)

        return Subspace(subspace.tocsr())

    def cost(self, k: int, *args, **kwargs):
        return self.r0.size

    def rcost(self, k: int, *args, **kwargs):
        return 2 * self.r0.size


class RandomAMG(_SubspaceGenerator):
    def __init__(self, linear_op: LinearOperator):
        """
        Generator of two stage subspace made up of a deterministic multi-level restriction and a random block.

        :param linear_op: Linear operator to base the algebraic multi-level hierarchy on.
        """
        # Sanitize arguments
        if not isinstance(linear_op, LinearOperator):
            raise SubspaceGeneratorError('Linear operator must be an instance of LinearOperator.')

        self.linear_op = linear_op
        self.P = None

    def get(self, k: int, heuristic: str, randomization: str, density: float = None, **kwargs) -> Subspace:
        """
        Compute a subspace in two levels, a first one obtained via an algebraic multi-grid method, and the second one of
        random type. The resulting subspace is then in the form of a product.

        :param k: Number of approximate eigen-vectors to compute.
        :param heuristic: Name of the algebraic multi-grid heuristic used to construct the hierarchy.
        :param randomization: Name of the random distribution to use for the second level projection.
        :param density: If provided, the random operator is sparse with given density.
        :param kwargs: complementary arguments for algebraic multi-grid construction, see PyAMG library for details.
        """
        # Setup multi-grid hierarchical structure with corresponding heuristic
        if heuristic == 'ruge_stuben':
            amg = pyamg.ruge_stuben_solver(self.linear_op.mat.tocsr(), max_levels=2, **kwargs)

        elif heuristic == 'smoothed_aggregated':
            amg = pyamg.smoothed_aggregation_solver(self.linear_op.mat.tocsr(), max_levels=2, **kwargs)

        elif heuristic == 'rootnode':
            amg = pyamg.rootnode_solver(self.linear_op.mat.tocsr(), max_levels=2, **kwargs)
        else:
            raise SubspaceError('Algebraic multi-grid heuristic {} unknown.'.format(heuristic))

        self.P = Subspace(amg.levels[0].P)

        G = None

        _, k_ = self.P.shape

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

        return self.P @ G

    def cost(self, k: int, *args, density: float = 1.):
        p = self.P.shape[1]
        k = min(k, p)
        return self.P.matvec_cost + 2*density*k*p

    def rcost(self, k: int, *args, density: float = 1.):
        p = self.P.shape[1]
        k = min(k, p)
        return self.P.T.matvec_cost + 2*density*k*p


class MultiLevelRandomSplit(_SubspaceGenerator):
    def __init__(self, linear_system: LinearSystem, x0: numpy.ndarray = None):
        """
        Generator of random split subspace, that is a distribution of very sparse random column block.

        :param linear_system: Linear system from which the initial residual is extracted.
        :param x0: Initial guess to compute the initial residual.
        """
        # Sanitize arguments
        if not isinstance(linear_system, LinearSystem):
            raise SubspaceGeneratorError('Linear system must be an instance of LinearSystem.')

        if x0 is not None and not isinstance(x0, numpy.ndarray):
            raise SubspaceGeneratorError('Initial guess must be a numpy.ndarray.')

        self.r0 = linear_system.rhs if x0 is None else linear_system.get_residual(x0)
        self.d = linear_system.linear_op.mat.diagonal()

    def get(self, k: int, n_levels: int, x: numpy.ndarray = None) -> Subspace:
        """
        Draw a subspace from the random split distribution of dimension k. The subspace is by default generated from the
        initial residual r0 of the linear system, but can be constructed from any n-dimensional vector x.

        :param k: Number of vectors making up the expected subspace.
        :param n_levels: Number of levels needed.
        :param x: Vector to optionally build the subspace from.
        """
        # Senitize arguments
        if not isinstance(n_levels, int) and n_levels < 1:
            raise SubspaceGeneratorError('Number of levels must be a strictly positive integer.')

        n = self.r0.size

        step = (n - k) / n_levels
        levels = list()

        for i in range(n_levels):
            # Initialize subspace in dok format to allow easy update
            subspace = scipy.sparse.dok_matrix((int(n - i * step), int(n - (i+1) * step)))

            # Draw columns indices
            cols = random_surjection(int(n - i * step), int(n - (i+1) * step))
            index = numpy.arange(int(n - i * step))

            # Fill-in with coefficients of linear system's right-hand side or provided vector x
            if i == 0:
                content = self.r0.T / self.d.T if x is None else x.T
            else:
                # content = levels[-1].T.dot(numpy.ones(int(n - (i-1) * step))).T
                content = numpy.ones(int(n - i * step)).T

            subspace[index, cols] = content

            levels.append(Subspace(subspace.tocsr()))

        S = levels[0]

        for i in range(1, n_levels):
            S = S @ levels[i]

        return S

    def cost(self, k: int, *args, **kwargs):
        levels = args[0]
        return self.r0.size * levels - 0.5 * (self.r0.size - k) * (levels - 1)

    def rcost(self, k: int, *args, **kwargs):
        return 2 * self.r0.size
