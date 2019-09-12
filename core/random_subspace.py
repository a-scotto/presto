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
        else:
            raise ValueError('Sampling strategy unknown.')

    def _binary_sparse(self, d):
        n, k = self.shape
        subspace = scipy.sparse.lil_matrix(self.shape)

        r = int(k + (n - k) * d)
        rows = [i % n for i in range(r)]
        random.shuffle(rows)

        for i in range(r):
            subspace[rows[i], i % k] = (2 * numpy.random.randint(0, 2) - 1) / numpy.sqrt(r / k)

        return subspace.tocsc()

    def _gaussian_sparse(self, d):
        n, k = self.shape
        subspace = scipy.sparse.lil_matrix(self.shape)

        r = int(k + (n - k) * d)
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
