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
import warnings
import scipy.optimize

from typing import List, Tuple
from core.algebra import LinearOperator

__all__ = ['Timer', 'compute_subspace_dim', 'report_init', 'merge_reports', 'load_operator', 'random_surjection']


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

