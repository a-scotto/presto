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

    def generate(self, sampling, *args):
        if sampling == 'bs':
            return self._binary_sparse(*args)
        elif sampling == 'gs':
            return self._gaussian_sparse(*args)
        elif sampling == 'gd':
            return self._gaussian_dense()
        elif sampling == 't':
            return self._test(*args)
        else:
            raise ValueError('Sampling strategy unknown.')

    def _binary_sparse(self, d=1.):
        n, k = self.shape
        subspace = scipy.sparse.lil_matrix(self.shape)

        r = max(int(n * d), k)

        rows = [i % n for i in range(r)]
        random.shuffle(rows)

        for i in range(r):
            subspace[rows[i], i % k] = (2 * numpy.random.randint(0, 2) - 1) / numpy.sqrt(d)

        return subspace.tocsc()

    def _gaussian_sparse(self, d=1.):
        n, k = self.shape
        subspace = scipy.sparse.lil_matrix(self.shape)

        r = max(int(n * d), k)

        rows = [i % n for i in range(r)]
        random.shuffle(rows)

        for i in range(r):
            subspace[rows[i], i % k] = numpy.random.randn()

        return subspace.tocsc()

    def _gaussian_dense(self, density=1., loc=0, scale=None):
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

    def _test(self):
        n, k = self.shape
        subspace = scipy.sparse.lil_matrix(self.shape)

        target = [i for i in range(n)]
        random.shuffle(target)

        alpha = 100
        beta = 0.1 ** 2 / 4 / numpy.log(alpha)
        theta = beta * n ** 2
        c = int(numpy.sqrt(theta * numpy.log(alpha)))

        for j in range(k):
            p = target[j]
            # x = numpy.asarray([i for i in range(max(p - c, 0), min(p + c + 1, n))])
            # P = numpy.exp(-(x - p)**2 / theta)
            #
            # distribution = scipy.stats.rv_discrete(1, n, values=(x, P / numpy.sum(P)))
            #
            # n_points = n_points
            # n_points = int(numpy.log(n)) + 1

            # for i in range(n_points):
            for i in range(max(p - c, 0) - p, min(p + c + 1, n) - p):

                # subspace[distribution.rvs(), j] = 1 / n_points**0.5
                # subspace[distribution.rvs(), j] = numpy.random.randn() / n_points**0.5
                # subspace[distribution.rvs(), j] = 2 * numpy.random.randint(0, 2) - 1 / n_points**0.5

                subspace[p + i, j] = (2 * numpy.random.randint(0, 2) - 1) / (numpy.abs(i) + 1)**0.5

        return subspace.tocsc()
