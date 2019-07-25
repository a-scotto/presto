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

    def __init__(self, shape, dtype=numpy.float64, sparse_tol=5e-2):
        if not isinstance(shape, tuple) or len(shape) != 2:
            raise RandomSubspaceError('Shape must be a tuple of the form (n, p).')

        if not isinstance(shape[0], int) or not isinstance(shape[1], int):
            raise RandomSubspaceError('Shape must me a tuple of integers.')

        self.shape = shape

        try:
            self.dtype = numpy.dtype(dtype)
        except TypeError:
            raise RandomSubspaceError('dtype provided not understood')

        self.sparse_tol = sparse_tol

    def gaussian(self, density, loc=0, scale=None):
        n, k = self.shape

        if scale is None:
            try:
                scale = 1 / (1 - density)
            except ZeroDivisionError:
                scale = 1.

        rvs = scipy.stats.norm(loc=loc, scale=scale)

        if not 0. < density <= 1.:
            raise RandomSubspaceError('Density must be float between 0 and 1.')

        subspace = scipy.sparse.random(n, k, density=density, format='csr', data_rvs=rvs.rvs)

        if density > self.sparse_tol:
            subspace = subspace.todense()

        return subspace

    def sparse_embedding(self):
        n, k = self.shape
        subspace = scipy.sparse.lil_matrix(self.shape)

        p = (n // k + 1) * k
        index = [i % k for i in range(p)]
        random.shuffle(index)

        for i in range(n):
            e = 2 * numpy.random.randint(0, 2) - 1
            subspace[i, index[i]] = e

        return subspace.tocsc()

    def sparse_embedding_mod(self):
        n, k = self.shape
        subspace = scipy.sparse.lil_matrix(self.shape)

        p = (n // k + 1) * k
        index = [i % k for i in range(p)]
        random.shuffle(index)

        for i in range(n):
            e = numpy.random.randn()
            subspace[i, index[i]] = e

        return subspace.tocsc()


if __name__ == "__main__":

    from tqdm import tqdm
    from matplotlib import pyplot
    from linear_operator import SelfAdjointMatrix
    from preconditioner import LimitedMemoryPreconditioner
    from problems.loader import load_problem, print_problem

    file_name = 'msc04515'
    problem = load_problem(file_name)
    print_problem(problem)

    if problem['singular_values'] is None:
        u = numpy.random.randn(problem['rank'], 1)
        gamma = float(u.T.dot(u)) / float(u.T.dot(problem['operator'].dot(u)))
    elif problem['singular_values'][0] > 1:
        gamma = float(2 / (problem['singular_values'][0] + problem['singular_values'][-1]))
    elif problem['singular_values'][-1] < 1:
        gamma = float(0.5 * problem['singular_values'][0] + problem['singular_values'][-1])
    else:
        gamma = 1.

    A = SelfAdjointMatrix(gamma * problem['operator'], def_pos=True)

    n, _ = A.shape
    delta = 0.005

    density = []
    nnz = []
    cond = []
    size = []

    k_ = [10 * i for i in range(10, 1000)]
    k_lim = [ki * (1 - 1/ki)**n for ki in k_]

    pyplot.semilogy(k_, k_lim)

    for k in tqdm(range(1, n // 150)):
        s = 100 * k
        S = RandomSubspaceFactory((n, s), sparse_tol=0.15).sparse_embedding()
        A_tilde = S.T.dot(A.dot(S))
        if isinstance(A_tilde, numpy.ndarray):
            density.append(numpy.sum(A_tilde != 0) / s**2)
            nnz.append(numpy.sum(A_tilde != 0))
        else:
            density.append(A_tilde.size / s**2)
            nnz.append(A_tilde.size)
            A_tilde = A_tilde.todense()

        cond.append(numpy.linalg.cond(A_tilde))
        size.append(s)

    pyplot.figure()
    pyplot.semilogy(size, density)
    pyplot.title('Density of reduced operator')

    pyplot.figure()
    pyplot.semilogy(size, cond)
    pyplot.title('Density of reduced operator')

    pyplot.figure()
    pyplot.plot(size, nnz)
    pyplot.title('Non-zeros entries of reduced operator')
    pyplot.hlines(problem['non_zeros'], xmin=0, xmax=n)
    pyplot.show()




