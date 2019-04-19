#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on April 19, 2019 at 09:37.

@author: a.scotto

Description:
"""

import numpy

from linear_map import LinearMap
from utils import qr, convert_to_col


class SubspaceError(Exception):
    """
    Raise error relative to Subspace objects.
    """


class ProjectorError(Exception):
    """
    Raise error relative to Projector objects.
    """


class Projector(LinearMap):

    def __init__(self, X, Y=None, orthogonalize=True, ip_B=None):

        Y = X if Y is None else Y

        if not isinstance(X, numpy.ndarray) or not isinstance(Y, numpy.ndarray):
            raise ProjectorError('Projector objects must be defined from numpy.ndarray')

        X = convert_to_col(X)
        Y = convert_to_col(Y)

        if X.shape != Y.shape:
            raise ProjectorError('Projector range and kernel orthogonal must have same dimension.')

        shape = (X.shape[0], X.shape[0])
        dtype = numpy.find_common_type([X.dtype, Y.dtype], [])

        super().__init__(shape, dtype, self._dot, self._dot_adj)

        if orthogonalize:
            X, R_x = qr(X, ip_B=ip_B)
        if Y is not X:
            print('lol')
            Y, R_y = qr(Y, ip_B=ip_B)

        self._range = X
        self._kernel_ortho = Y
        self._R = None
        self._A = None

    def _dot(self, X):
        return self._A @ X

    def _dot_adj(self, X):
        return self._A.T.conj() @ X

    def __repr__(self):
        return self._A.__repr__()


if __name__ == "__main__":

    P = Projector(numpy.ones((2, 100)))

    print(P.shape, P.dtype)