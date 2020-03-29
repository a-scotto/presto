#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on September 06, 2019 at 09:43.

@author: a.scotto

Description:
"""

import os
import time
import tqdm
import numpy
import pickle
import fnmatch
import scipy.sparse
import scipy.optimize

from utils.operator import TestOperator
from core.linear_operator import SelfAdjointMatrix
from core.linear_system import ConjugateGradient, LinearSystem
from core.projection_subspace import RandomSubspaceFactory, DeterministicSubspaceFactory
from core.preconditioner import LimitedMemoryPreconditioner, AlgebraicPreconditionerFactory


OPERATORS_ROOT_PATH = 'operators/'
REFERENCES_RUN_ROOT_PATH = 'runs/'
REPORTS_ROOT_PATH = 'reports/'

MAX_BUDGET_RATIO = 8*100


def process_precond_data(precond_parameters: str) -> dict:
    """
    Method to process data concerning the preconditioner in setup text file lines.

    :param precond_parameters: Parameters of the preconditioner to benchmark.
    """

    subspace, first_precond = precond_parameters.split(',')

    # Retrieve subspace information
    try:
        subspace_name, parameters = subspace.split(':')

        # Check if parameters is a float
        try:
            parameters = float(parameters)
        except ValueError:
            pass

    except ValueError:
        subspace_name = subspace
        parameters = None

    subspace = dict(name=subspace_name, parameters=parameters)

    # Retrieve firs-level preconditioner information
    try:
        first_precond, parameters = first_precond.split(':')

    except ValueError:
        first_precond = first_precond
        parameters = None

    first_precond = dict(name=first_precond, parameters=parameters)

    return dict(subspace=subspace, first_precond=first_precond)


def process_benchmark_data(benchmark_data: str) -> dict:
    """
    Method to process data concerning the benchmark in setup text file lines.

    :param benchmark_data: Data of the benchmark itself.
    """

    n_tests, n_subspaces = benchmark_data.split(',')

    return dict(n_tests=int(n_tests), n_subspaces=int(n_subspaces))


def process_operators_data(operators_data: str) -> dict:
    """
    Method to process data concerning the operators in setup text file lines.

    :param operators_data: Data of the operators to benchmark on.
    """

    operators_list = os.listdir(OPERATORS_ROOT_PATH)
    operators_list = fnmatch.filter(operators_list, '*.opr')

    if operators_data == '*':
        operators = [os.path.join(OPERATORS_ROOT_PATH, op) for op in operators_list]

    else:
        operators = [os.path.join(OPERATORS_ROOT_PATH, op + '.opr')
                     for op in operators_data.split(',')
                     if op + '.opr' in operators_list]

    return dict(operators=operators)


def read_setup(SETUP_FILE_PATH: str) -> list:
    """
    Method to read the benchmark setup text file and return the different setup in a suitable
    format to run the benchmark.

    :param SETUP_FILE_PATH: Path of the setup text file.
    """

    # Initialize the list of benchmarks
    benchmarks = list()

    # Read the setup text file
    with open(SETUP_FILE_PATH, 'r') as setup_text_file:
        for line in setup_text_file.readlines():
            # Skip user dedicated header
            if line.startswith('>'):
                continue

            # Remove useless spacings and newline character
            line = line.replace(' ', '')
            line = line.replace('\n', '')

            precond_data, benchmark_data, operators_data = line.split('|')

            # Parsing the data
            precond_data = process_precond_data(precond_data)
            benchmark_data = process_benchmark_data(benchmark_data)
            operators = process_operators_data(operators_data)

            # Merge data
            benchmark_setup = dict(**precond_data, **benchmark_data, **operators)

            benchmarks.append(benchmark_setup)

    return benchmarks


def compute_subspace_sizes(n_subspaces: int,
                           linear_operator: scipy.sparse.spmatrix,
                           subspace_type: str = 'dense') -> numpy.asarray:
    """
    Method to compute the list of subspaces size to be tested, regarding the number of size to be
    tested, the order of the linear operator, and the type of subspace either dense or sparse.

    :param n_subspaces: Number of subspaces to test.
    :param linear_operator: Linear operator tested.
    :param subspace_type: Either dense or sparse leading to a different cost computation.
    """

    # Initialise parameters
    n, _ = linear_operator.shape
    a = linear_operator.size

    # Maximum budget
    budget = MAX_BUDGET_RATIO * n

    # Compute the maximum subspace size depending of the subspace type
    if subspace_type == 'dense':
        k_max = budget / (8 * n)
    elif subspace_type == 'sparse':
        k_max = scipy.optimize.fsolve(lambda k: 4*k**2 + 4*a + 6*n - budget, x0=n)
    else:
        raise ValueError('Subspace type must be either dense or sparse.')

    step = float(k_max / n_subspaces)
    subspace_sizes = numpy.asarray([(i + 1) * step for i in range(n_subspaces)], dtype=numpy.int32)

    return subspace_sizes


def set_reference_run(operator: TestOperator, precond: dict, n_runs: int = 50) -> dict:
    """
    Method to run the Conjugate Gradient a representative number of times, to select the right-hand 
    side corresponding to the average run.

    :param operator: TestOperator object to run the Conjugate Gradient on.
    :param precond: First-level preconditioner to run the CG with.
    :param n_runs: Number of runs to compute the average on.
    """

    # Initialize parameters
    n, _ = operator['shape']
    PATH = os.path.join(REFERENCES_RUN_ROOT_PATH, operator['name'] + '.ref')
    precond_label = precond['name'] + '_' + str(precond['parameters'])

    # Create file if not existing
    if not os.path.isfile(PATH):
        with open(PATH, 'wb') as file:
            pickle.Pickler(file).dump(dict())

    # Open file corresponding to the operator under test
    with open(PATH, 'rb') as file:
        operators_run = pickle.Unpickler(file).load()

        if precond_label not in operators_run.keys():

            # Set a CG run of reference
            rhs, iterations = list(), list()
            lin_op = SelfAdjointMatrix(operator['operator'], def_pos=True)
            M = AlgebraicPreconditionerFactory(lin_op).get(precond['name'], precond['parameters'])

            # Run the CG with randomly generated right-hand sides
            for i in range(n_runs):
                # Create random linear system
                lin_sys = LinearSystem(lin_op, lin_op.dot(numpy.random.randn(n, 1)))

                # Run the Conjugate Gradient
                cg = ConjugateGradient(lin_sys, x_0=None, M=M)

                # Store results
                rhs.append(lin_sys.rhs)
                iterations.append(cg.output['n_iterations'])

            # Get the index of the average run
            n_iterations = numpy.asarray(iterations)
            average_run = int(numpy.argmin(numpy.abs(n_iterations - numpy.mean(n_iterations))))

            operators_run[precond_label] = dict(n_iterations=n_iterations[average_run],
                                                rhs=rhs[average_run])

            # Save results obtained
            with open(PATH, 'wb') as new_file:
                pickle.Pickler(new_file).dump(operators_run)

    return operators_run[precond_label]


def initialize_report(operator: TestOperator, setup: dict, n_iterations: int) -> str:
    """
    Method to create a report file and write the header.

    :param operator: TestOperator containing the operator to benchmark on and its metadata.
    :param setup: dictionary containing the setup parameters.
    :param n_iterations: number of iteration of the reference run with the first-level
    preconditioner alone.
    """
    # Date and time fore report identification
    date_ = ''.join(time.strftime("%x").split('/'))
    time_ = ''.join(time.strftime("%X").split(':'))

    # Sampling method and its parameters
    sampling = setup['subspace']['name'] + '_' + str(setup['subspace']['parameters'])
    precond = setup['first_precond']['name'] + '_' + str(setup['first_precond']['parameters'])

    # Set the report name
    report_name = '_'.join([operator['name'], date_, time_, precond, sampling]) + '.rpt'

    # Aggregate metadata of both the operator tested and the benchmark itself
    operator_metadata = ', '.join([str(operator['rank']),
                                   str(operator['non_zeros']),
                                   str(operator['conditioning']),
                                   operator['source']])

    benchmark_metadata = ', '.join([setup['first_precond']['name'],
                                    str(setup['first_precond']['parameters']),
                                    setup['subspace']['name'],
                                    str(setup['subspace']['parameters']),
                                    str(n_iterations)])

    # Header writing
    with open('reports/' + report_name, 'w') as report_file:
        report_file.write('>>   REPORT OF PRECONDITIONING STRATEGIES BENCHMARK   << \n')
        report_file.write('> \n')
        report_file.write('>  SOLVER ............. Conjugate Gradient \n')
        report_file.write('>  SUBSPACES TESTED ... ' + str(setup['n_subspaces']) + '\n')
        report_file.write('>  RUNS ............... ' + str(setup['n_tests']) + '\n')
        report_file.write('>  PROBLEM NAME ....... ' + operator['name'] + '\n')
        report_file.write('>  REFERENCE RUN ...... ' + str(n_iterations) + '\n')
        report_file.write('> \n')
        report_file.write('>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<< \n')
        report_file.write('~operator_metadata: ' + operator_metadata + '\n')
        report_file.write('~benchmark_metadata: ' + benchmark_metadata + '\n')
        report_file.write('>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<< \n')

    return report_name


def benchmark(operator: TestOperator, setup: dict, subspaces: list) -> None:
    """
    Process the benchmark of an operator with the specification contained in the setup
    dictionary. Compare the operator to a reference run and store the results in text file.

    :param operator: TestOperator containing the operator to benchmark on and its metadata.
    :param setup: Dictionary with benchmark parameters.
    :param subspaces: List of subspaces size to test.
    """

    # Initialize variables
    lin_op = SelfAdjointMatrix(operator['operator'], def_pos=True)
    lin_sys = LinearSystem(lin_op, operator['rhs'])
    n, _ = lin_op.shape

    # Proceed to reference run
    first_precond = AlgebraicPreconditionerFactory(lin_op).get(setup['first_precond']['name'],
                                                               setup['first_precond']['parameters'])

    ref_cg = ConjugateGradient(lin_sys, x_0=None, M=first_precond)
    n_iterations = ref_cg.output['n_iterations']

    # Initialize report text file
    report_name = initialize_report(operator, setup, n_iterations)

    # Process the benchmark
    for k in tqdm.tqdm(subspaces):
        report_line = '{:4} | '.format(k)

        # Create the subspace factory producing subspaces of shape (n, k)
        if setup['subspace']['name'] in DeterministicSubspaceFactory.subspace_type.keys():
            perturbation = 1e-2

            # Diagonal perturbation of linear operator
            mat_op = SelfAdjointMatrix(operator['operator'] + perturbation * scipy.sparse.eye(n), def_pos=True)

            # Perturbation of right-hand side
            rhs = operator['rhs'] + perturbation * operator['operator'].dot(numpy.random.randn(n, 1))

            precond = AlgebraicPreconditionerFactory(mat_op).get(setup['first_precond']['name'],
                                                                 setup['first_precond']['parameters'])

            factory = DeterministicSubspaceFactory(mat_op, precond=precond, rhs=rhs)

        elif setup['subspace']['name'] in RandomSubspaceFactory.subspace_type.keys():
            factory = RandomSubspaceFactory(lin_op)
        else:
            raise ValueError('Subspace type unrecognized.')

        # Process the required number of runs
        for i in range(setup['n_tests']):
            subspace = factory.get(setup['subspace']['name'], k, setup['subspace']['parameters'])
            lmp = LimitedMemoryPreconditioner(lin_op, subspace, M=first_precond)

            cg = ConjugateGradient(lin_sys, x_0=None, M=lmp)

            report_line += str(cg.output['n_iterations'] / n_iterations) + ','

        # Write result in report text file
        with open(os.path.join(REPORTS_ROOT_PATH, report_name), 'a') as report_file:
            report_file.write(report_line[:-1] + '\n')
