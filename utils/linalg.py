#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on April 13, 2020 at 11:51.

@author: a.scotto

Description:
"""

import numpy
import scipy

from typing import Union, Tuple
from scipy.sparse.linalg import LinearOperator

__all__ = ['inner', 'norm', 'qr', 'angles']

MatrixType = Union[numpy.ndarray, scipy.sparse.spmatrix]


class LinearAlgebraError(Exception):
    """
    Exception to handle errors specific to linear algebra methods.
    """


def inner(V: MatrixType, W: MatrixType, ip_B: LinearOperator = None) -> numpy.ndarray:
    """
    Euclidean and non-Euclidean inner product.

    :param V: matrix with shape (n, m).
    :param W: matrix with shape (n, p).
    :param ip_B: inner product, a self-adjoint, positive-definite linear operator.
    """
    if not V.shape[0] == W.shape[0]:
        raise LinearAlgebraError('V and W have inconsistent shapes: {} and {}'.format(V.shape, W.shape))

    if ip_B is not None and not isinstance(ip_B, LinearOperator):
        raise LinearAlgebraError('Inner product must be a LinearOperator instance.')

    if ip_B is not None:
        inner_product = ip_B.dot(V).T @ W
    else:
        inner_product = V.T @ W

    if scipy.sparse.isspmatrix(inner_product):
        inner_product = inner_product.todense()

    return inner_product


def norm(x: MatrixType, ip_B: LinearOperator = None) -> Union[float, numpy.ndarray]:
    """
    Compute norm (Euclidean and non-Euclidean).

    :param x: a 2-dimensional matrix object.
    :param ip_B: inner product, a self-adjoint, positive-definite linear operator.
    """
    if x.ndim == 1 or x.shape[1] == 1:
        return float(inner(x, x, ip_B=ip_B))**0.5
    else:
        norms = [float(inner(x[:, [i]], x[:, [i]], ip_B=ip_B)**0.5) for i in range(x.shape[1])]
        return numpy.asarray(norms)


def qr(X: MatrixType, ip_B: LinearOperator = None, passes: int = 1) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    QR factorization with customizable inner product.

    :param X: matrix with shape (N,k).
    :param ip_B: inner product, a self-adjoint, positive-definite linear operator.
    :param passes: number of re-orthogonalizations.
    """
    if scipy.sparse.isspmatrix(X):
        X = X.todense()

    if ip_B is None and X.shape[1] > 0:
        return scipy.linalg.qr(X, mode='economic')
    else:
        _, k = X.shape
        Q = X.copy()
        R = numpy.zeros((k, k), dtype=X.dtype)
        for i in range(k):
            for _ in range(passes+1):
                for j in range(i):
                    alpha = inner(Q[:, [j]], Q[:, [i]], ip_B=ip_B)[0, 0]
                    R[j, i] += alpha
                    Q[:, [i]] -= alpha * Q[:, [j]]
            R[i, i] = norm(Q[:, [i]], ip_B=ip_B)
            if R[i, i] >= 1e-15:
                Q[:, [i]] /= R[i, i]
        return Q, R


def angles(F, G, ip_B=None, compute_vectors=False, degree=False):
    """Principal angles between two subspaces.

    This algorithm is based on algorithm 6.2 in `Knyazev, Argentati. Principal
    angles between subspaces in an A-based scalar product: algorithms and
    perturbation estimates. 2002.` This algorithm can also handle small angles
    (in contrast to the naive cosine-based svd algorithm).

    :param F: array with ``shape==(N,k)``.
    :param G: array with ``shape==(N,l)``.
    :param ip_B: (optional) angles are computed with respect to this
      inner product. See :py:meth:`inner`.
    :param compute_vectors: (optional) if set to ``False`` then only the angles
      are returned (default). If set to ``True`` then also the principal
      vectors are returned.
    :param degree: Whether to convert angles from radians to degrees.

    :return:

      * ``theta`` if ``compute_vectors==False``
      * ``theta, U, V`` if ``compute_vectors==True``

      where

      * ``theta`` is the array with ``shape==(max(k,l),)`` containing the
        principal angles
        :math:`0\\leq\\theta_1\\leq\\ldots\\leq\\theta_{\\max\\{k,l\\}}\\leq
        \\frac{\\pi}{2}`.
      * ``U`` are the principal vectors from F with
        :math:`\\langle U,U \\rangle=I_k`.
      * ``V`` are the principal vectors from G with
        :math:`\\langle V,V \\rangle=I_l`.

    The principal angles and vectors fulfill the relation
    :math:`\\langle U,V \\rangle = \
    \\begin{bmatrix} \
    \\cos(\\Theta) & 0_{m,l-m} \\\\ \
    0_{k-m,m} & 0_{k-m,l-m} \
    \\end{bmatrix}`
    where :math:`m=\\min\\{k,l\\}` and
    :math:`\\cos(\\Theta)=\\operatorname{diag}(\\cos(\\theta_1),\\ldots,\\cos(\\theta_m))`.
    Furthermore,
    :math:`\\theta_{m+1}=\\ldots=\\theta_{\\max\\{k,l\\}}=\\frac{\\pi}{2}`.
    """
    # make sure that F.shape[1]>=G.shape[1]
    reverse = False
    if F.shape[1] < G.shape[1]:
        reverse = True
        F, G = G, F

    QF, _ = qr(F, ip_B=ip_B)
    QG, _ = qr(G, ip_B=ip_B)

    # one or both matrices empty? (enough to check G here)
    if G.shape[1] == 0:
        theta = numpy.ones(F.shape[1])*numpy.pi/2
        U = QF
        V = QG
    else:
        Y, s, Z = scipy.linalg.svd(inner(QF, QG, ip_B=ip_B))
        Vcos = numpy.dot(QG, Z.T.conj())
        n_large = numpy.flatnonzero((s**2) < 0.5).shape[0]
        n_small = s.shape[0] - n_large
        theta = numpy.r_[
            numpy.arccos(s[n_small:]),  # [-i:] does not work if i==0
            numpy.ones(F.shape[1]-G.shape[1])*numpy.pi/2]
        if compute_vectors:
            Ucos = numpy.dot(QF, Y)
            U = Ucos[:, n_small:]
            V = Vcos[:, n_small:]

        if n_small > 0:
            RG = Vcos[:, :n_small]
            S = RG - numpy.dot(QF, inner(QF, RG, ip_B=ip_B))
            _, R = qr(S, ip_B=ip_B)
            Y, u, Z = scipy.linalg.svd(R)
            theta = numpy.r_[
                numpy.arcsin(u[::-1][:n_small]),
                theta]
            if compute_vectors:
                RF = Ucos[:, :n_small]
                Vsin = numpy.dot(RG, Z.T.conj())
                # next line is hand-crafted since the line from the paper does
                # not seem to work.
                Usin = numpy.dot(RF, numpy.dot(
                    numpy.diag(1/s[:n_small]),
                    numpy.dot(Z.T.conj(), numpy.diag(s[:n_small]))))
                U = numpy.c_[Usin, U]
                V = numpy.c_[Vsin, V]

    if compute_vectors:
        if reverse:
            U, V = V, U

        if not degree:
            return theta[:G.shape[1]], U, V
        else:
            return numpy.rad2deg(theta[:G.shape[1]]), U, V
    else:
        if not degree:
            return theta[:G.shape[1]]
        else:
            return numpy.rad2deg(theta[:G.shape[1]])
