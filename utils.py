#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on April 19, 2019 at 08:59.

@author: a.scotto

Description:
"""

import numpy
import scipy.linalg


class InnerProductError(Exception):
    """
    Raised when the inner product is indefinite.
    """


def convert_to_col(X):
    """
    Convert numpy.ndarray to column stack format, i.e. of shape (m, n) with m > n.

    :param X: numpy.ndarray with arbitrary shape
    :return: numpy.ndarray with shape (m, n) with m > n.
    """

    if len(X.shape) == 1:
        X = numpy.atleast_2d(X).T

    if len(X.shape) != 2:
        raise ValueError('Impossible to convert to columns numpy.ndarray of dimension =/= 2.')

    if X.shape[1] > X.shape[0]:
        X = X.T

    return X


def inner(X, Y, ip_B=None):
    """
    Euclidean and non-Euclidean inner product.

    Extension of numpy.vdot and numpy.dot which respectively only works for vectors and does not
    use the conjugate transpose.

    :param X: numpy array with ``shape==(N,m)``
    :param Y: numpy array with ``shape==(N,n)``
    :param ip_B: (optional) May be one of the following

        * ``None``: Euclidean inner product.
        * a self-adjoint and positive definite operator :math:`B` (as ``numpy.array`` or
        ``LinearOperator``). Then :math:`X^*B Y` is returned.
        * a callable which takes 2 arguments X and Y and returns :math:`\\langle X,Y\\rangle`.

    **Caution:** a callable should only be used if necessary. The choice
    potentially has an impact on the round-off behavior, e.g. of projections.

    :return: numpy array :math:`\\langle X,Y\\rangle` with ``shape==(m,n)``.
    """

    if ip_B is None:
        return numpy.dot(X.T.conj(), Y)

    (N, m) = X.shape
    (_, n) = Y.shape

    if m > n:
        return numpy.dot((ip_B.dot(X)).T.conj(), Y)
    else:
        return numpy.dot(X.T.conj(), ip_B.dot(Y))


def norm(x, y=None, ip_B=None):
    """
    Compute norm (Euclidean and non-Euclidean).

    :param x: a 2-dimensional ``numpy.array``.
    :param y: a 2-dimensional ``numpy.array``.
    :param ip_B: see :py:meth:`inner`.

    Compute :math:`\sqrt{\langle x,y\rangle}` where the inner product is defined via ``ip_B``.
    """

    # Euclidean inner product?
    if y is None and ip_B is None:
        return numpy.linalg.norm(x, 2)

    if y is None:
        y = x

    ip = inner(x, y, ip_B=ip_B)

    norm_diag = numpy.linalg.norm(numpy.diag(ip), 2)
    norm_diag_imag = numpy.linalg.norm(numpy.imag(numpy.diag(ip)), 2)

    if norm_diag_imag > norm_diag*1e-10:
        raise InnerProductError('Inner product defined by ip_B is potentially not positive definite'
                                '||diag(ip).imag||/||diag(ip)||={0}'
                                .format(norm_diag_imag / norm_diag))

    return numpy.sqrt(numpy.linalg.norm(ip, 2))


def qr(X, ip_B=None, reorthos=1):
    """
    QR factorization with customizable inner product.

    :param X: array with ``shape==(N,k)``
    :param ip_B: (optional) inner product, see :py:meth:`inner`.
    :param reorthos: (optional) number of reorthogonalizations. Defaults to 1 (i.e. 2 runs of
    modified Gram-Schmidt) which should be enough in most cases (TODO: add reference).

    :return: Q, R where :math:`X=QR` with :math:`\\langle Q,Q \\rangle=I_k` and R upper triangular.
    """

    if ip_B is None and X.shape[1] > 0:
        pass
        return scipy.linalg.qr(X, mode='economic')

    else:
        (N, k) = X.shape
        Q = X.copy()
        R = numpy.zeros((k, k), dtype=X.dtype)

        for i in range(k):

            for reortho in range(reorthos+1):
                for j in range(i):
                    alpha = inner(Q[:, [j]], Q[:, [i]], ip_B=ip_B)[0, 0]
                    R[j, i] += alpha
                    Q[:, [i]] -= alpha * Q[:, [j]]

            R[i, i] = norm(Q[:, [i]], ip_B=ip_B)

            if R[i, i] >= 1e-15:
                Q[:, [i]] /= R[i, i]

        return Q, R
