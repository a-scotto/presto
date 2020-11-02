#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on April 11, 2020 at 10:01.

@author: a.scotto

Description:
"""

import os
import time
import numpy
import pickle
import os.path
import scipy.sparse
import scipy.optimize

from presto.algebra import *
from typing import List, Tuple

__all__ = ['Timer', 'compute_subspace_dim', 'report_init', 'merge_reports', 'load_operator', 'random_surjection', 'qr',
           'subspace_angles']


class UtilsError(Exception):
    """
    Exception raised when utils method encounters specific errors.
    """


class Timer:
    """
    Timer class allowing to time block of code in a context manager. The opening of the context manager takes a label as
    argument so as to gather different timings in a single object.
    """
    def __init__(self, name: str):
        self.name = name
        self._start_time = None
        self.total_elapsed = 0.

    def start(self):
        """
        Begin to time.
        """
        self._start_time = time.perf_counter()

    def stop(self):
        """
        End to time.
        """
        elapsed_time = time.perf_counter() - self._start_time

        self._start_time = None
        self.total_elapsed += elapsed_time

    def __enter__(self):
        """
        Start a new timer as a context manager.
        """
        self.start()
        return self

    def __exit__(self, *exc_info):
        """
        Stop the context manager timer.
        """
        self.stop()


def compute_subspace_dim(max_budget: float,
                         N: int,
                         linear_operator: LinearOperator,
                         subspace_format: str) -> Tuple[List[int], List[int]]:
    """
    Method dedicated to Limited Memory Preconditioner (LMP) benchmark. Thus, compute subspace dimensions list linearly
    distributed from 0 to the limit dimensions allowed within the maximum budget for a matrix-product with the
    corresponding LMP.

    :param max_budget: Maximum computational budget in FLOPs.
    :param N: Number of dimensions to return.
    :param linear_operator: Linear operator involved in the underlying linear system to tests.
    :param subspace_format: Format of the subspace generated, either dense or sparse.
    """
    n, _ = linear_operator.shape
    a = linear_operator.matvec_cost

    # Compute the maximum subspace size depending of the subspace type
    if subspace_format == 'dense':
        k_max = max_budget / (8 * n)
    elif subspace_format == 'sparse':
        k_max = scipy.optimize.fsolve(lambda k: 4*k**2 + 2*a + 6*n - max_budget, x0=n)
    else:
        raise UtilsError('Format of subspace must be either dense or sparse.')

    step = float(k_max / N)
    subspace_dims, computational_cost = list(), list()

    for i in range(N):
        subspace_dims.append(int((i + 1)*step))
        if subspace_format == 'dense':
            computational_cost.append(8*subspace_dims[-1]*n)
        elif subspace_format == 'sparse':
            computational_cost.append(4*subspace_dims[-1]**2 + 2*a + 6*n)

    return subspace_dims, computational_cost


def report_init(config: dict,
                PRECONDITIONER: dict,
                SUBSPACE: dict,
                OPERATOR_PATH: str):
    """
    Initialize report of benchmark with metadata content and define unique name.

    :param config: General configuration of the benchmark.
    :param PRECONDITIONER: Preconditioner configuration of the benchmark.
    :param SUBSPACE: Subspace configuration of the benchmark.
    :param OPERATOR_PATH: Path to the operator file.
    """

    content = dict(tol=config["tol"],
                   maxiter=config["maxiter"],
                   MEMORY_LIMIT=config["MEMORY_LIMIT"],
                   perturbation=config["perturbation"],
                   operator=OPERATOR_PATH,
                   preconditioner=PRECONDITIONER,
                   subspace=SUBSPACE)

    OPERATOR_NAME = os.path.basename(OPERATOR_PATH)

    datetime = time.strftime('%d') + time.strftime('%m') + time.strftime('%Y') + '_'
    datetime += time.strftime('%H') + time.strftime('%M') + time.strftime('%S')

    FILE_NAME = '_'.join([OPERATOR_NAME,
                          PRECONDITIONER['name'],
                          SUBSPACE['name'],
                          datetime])

    FILE_NAME += '.json'

    return FILE_NAME, content


def merge_reports(REPORT_PATHS: str) -> dict:
    """
    Gathers report JSON files corresponding to the same operator tested.

    :param REPORT_PATHS: List of reports files to make the gathering on.
    """
    merged_reports = dict()

    for REPORT_PATH in REPORT_PATHS:
        OPERATOR_NAME = os.path.basename(REPORT_PATH).split('_')[0]
        PRECONDITIONER = os.path.basename(REPORT_PATH).split('_')[1]

        if OPERATOR_NAME in merged_reports.keys():
            if PRECONDITIONER in merged_reports[OPERATOR_NAME].keys():
                merged_reports[OPERATOR_NAME][PRECONDITIONER].append(REPORT_PATH)
            else:
                merged_reports[OPERATOR_NAME][PRECONDITIONER] = list([REPORT_PATH])
        else:
            merged_reports[OPERATOR_NAME] = dict()
            merged_reports[OPERATOR_NAME][PRECONDITIONER] = list([REPORT_PATH])

    return merged_reports


def load_operator(OPERATOR_FILE_PATH: str, display: bool = False) -> LinearOperator:
    """
    Load the binary operator file located at the specified path and return it as dictionary.

    :param OPERATOR_FILE_PATH: Path to the operator file containing LinearOperator instance.
    :param display: Whether to display operator's characteristics or not.
    """

    # Open the operator binary file and load the content
    with open(OPERATOR_FILE_PATH, 'rb') as file:
        p = pickle.Unpickler(file)
        operator = p.load()

    # Display the operator characteristics if required
    if display:
        print(operator)

    return operator


def random_surjection(n: int, k: int) -> numpy.ndarray:
    """
    Generate a random surjection output from {1, n} into {1, k}. Return an array of shape (n,) with all elements in
    {1, k} with integers from 1 to k present at least once.

    :param n: Number of elements in the initial set
    :param k: Number of elements in the target set
    """
    # Draw random map from n to k
    output = numpy.hstack([numpy.arange(k), numpy.random.randint(k, size=n - k)])
    numpy.random.shuffle(output)
    return output


def qr(V, ip_A=None):
    """
    Process to the QR decomposition in the inner product provided. Namely for a (n, p) matrix V, the algorithm provides
    two matrices Q and R such that V = QR with Q of shape (n, p) with its columns being conjugate with respect to the
    inner product ip_A and R of shape (p, p) upper triangular.

    :param V: Block columns to orthogonalize
    :param ip_A: Symmetric positive-definite operator as the inner product
    """

    # Sanitize the arguments of the method
    if not isinstance(V, (numpy.ndarray, scipy.sparse.spmatrix, LinearSubspace)):
        raise UtilsError('Cannot process to the orthogonalization because of format problem.')

    V = V.mat if isinstance(V, LinearSubspace) else V
    V = V.todense() if scipy.sparse.isspmatrix(V) else V
    ip_A = IdentityOperator(V.shape[0]) if ip_A is None else ip_A

    # Process to the orthogonalization
    A_ = V.T @ (ip_A @ V)
    L_factor, _ = scipy.linalg.cho_factor(A_, lower=True, overwrite_a=False)

    Q = scipy.linalg.solve_triangular(L_factor, V.T, lower=True).T
    R = numpy.tril(L_factor).T

    return Q, R


def subspace_angles(F, G, ip_A=None, compute_vectors=False, degree=False):
    """
    Principal angles between two subspaces.

    This algorithm is based on algorithm 6.2 in `Knyazev, Argentati. Principal angles between subspaces in an A-based
    scalar product: algorithms and perturbation estimates. 2002.` This algorithm can also handle small angles (in
    contrast to the naive cosine-based svd algorithm).

    :param F: array with ``shape==(N,k)``.
    :param G: array with ``shape==(N,l)``.
    :param ip_A: (optional) angles are computed with respect to this inner product. See :py:meth:`inner`.
    :param compute_vectors: (optional) if set to ``False`` then only the angles are returned (default). If set to
    ``True`` then also the principal vectors are returned.
    :param degree: Whether to convert angles from radians to degrees.
    """

    # Make sure that F has more columns than G
    reverse = False
    if F.shape[1] < G.shape[1]:
        reverse = True
        F, G = G, F

    _, p = G.shape

    # Set the inner product to work with
    ip_A = IdentityOperator(F.shape[0]) if ip_A is None else ip_A

    QF, _ = qr(F, ip_A=ip_A)
    QG, _ = qr(G, ip_A=ip_A)

    u, s, v = scipy.linalg.svd(QF.T @ (ip_A @ QG))
    V_cos = QG @ v.T
    n_large = numpy.flatnonzero((s**2) < 0.5).shape[0]
    n_small = s.shape[0] - n_large
    theta = numpy.r_[numpy.arccos(s[n_small:]), numpy.ones(F.shape[1] - p) * numpy.pi/2]

    # Deal with small angles using sine measure
    if n_small != 0:
        RG = V_cos[:, :n_small]
        S = RG - QF @ QF.T @ (ip_A @ RG)
        _, R = qr(S, ip_A=ip_A)
        u_, s_, v_ = scipy.linalg.svd(R)
        theta = numpy.r_[numpy.arcsin(s_[::-1][:n_small]), theta]
    else:
        RG, v_ = None, None

    # Compute vectors related to principal angles
    if compute_vectors:
        U_cos = numpy.dot(QF, u)
        U = U_cos[:, n_small:]
        V = V_cos[:, n_small:]

        if RG is not None and v_ is not None:
            RF = U_cos[:, :n_small]
            V_sin = RG @ v_.T
            U_sin = RF @ numpy.diag(1 / s[:n_small]) @ v_.T @ numpy.diag(s[:n_small])
            U = numpy.c_[U_sin, U]
            V = numpy.c_[V_sin, V]

        if reverse:
            U, V = V, U

        if not degree:
            return theta[:p], U, V
        else:
            return numpy.rad2deg(theta[:p]), U, V
    else:
        if not degree:
            return theta[:p]
        else:
            return numpy.rad2deg(theta[:p])
