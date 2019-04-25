#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on April 25, 2019 at 11:02.

@author: a.scotto

Description:
"""

import numpy
import utils

from matrix_operator import MatrixOperator, SelfAdjointMap


class LinearSystemError(Exception):
    """
    Exception raised when LinearSystem object encounters specific errors.
    """


class LinearSystem(object):

    def __init__(self, A, b,
                 x_k=None,
                 err_norm=None,
                 symmetric=False,
                 definite_positive=False):

        self.symmetric = symmetric
        self.definite_positive = definite_positive

        if self.symmetric:
            self.A = SelfAdjointMap(A)
        else:
            self.A = MatrixOperator(A)

        if not isinstance(b, numpy.ndarray):
            raise LinearSystemError('Linear system left-hand side must be numpy.ndarray')

        self.b = b

        x_k = numpy.zeros_like(b) if x_k is None else x_k

        if not isinstance(x_k, numpy.ndarray):
            raise LinearSystemError('Linear system initial guess must be numpy.ndarray')

        self.x_k = x_k

        if err_norm is not None and not isinstance(err_norm, numpy.ndarray):
            raise LinearSystemError('Linear system error norm must be numpy.ndarray')

        self.err_norm = err_norm
        self.res = utils.norm(self.A.dot(self.x_k) - self.b, ip_B=err_norm)

    def get_residual(self):
        return self.res

    def left_precond(self, M):
        self
        return None

    def right_precond(self, M):
        self
        return None
