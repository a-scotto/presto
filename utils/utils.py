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
