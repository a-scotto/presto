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

from core.linear_operator import SelfAdjointMatrix
from core.linear_system import ConjugateGradient, LinearSystem
from core.projection_subspace import RandomSubspaceFactory, KrylovSubspaceFactory
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

    subspace, first_level_precond = precond_parameters.split(',')

    # Retrieve subspace information
    try:
        subspace_type, parameters = subspace.split(':')

        # Check if parameters is a float
        try:
            parameters = float(parameters)
        except ValueError:
            pass

    except ValueError:
        subspace_type = subspace
        parameters = None

    subspace = dict(type=subspace_type, parameters=parameters)

    return dict(subspace=subspace, first_level_precond=first_level_precond)


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
        k_max = scipy.optimize.fsolve(lambda k: 4*k**2 + 4*a + n - budget, x0=n)
    else:
        raise ValueError('Subspace type must be either dense or sparse.')

    step = float(k_max / n_subspaces)
    subspace_sizes = numpy.asarray([(i + 1) * step for i in range(n_subspaces)], dtype=numpy.int32)

    return subspace_sizes


def set_reference_run(operator: dict, precond: str, n_runs: int = 100) -> ConjugateGradient:
    """
    Method to run the Conjugate Gradient a representative number of times, to select the right-hand 
    side corresponding to the average run.

    :param operator: LinearOperator object to run the Conjugate Gradient on.
    :param precond: First-level preconditioner to run the CG with.
    :param n_runs: Number of runs to compute the average on.
    """

    # Initialize parameters
    n, _ = operator.get('shape')
    PATH = os.path.join(REFERENCES_RUN_ROOT_PATH, operator.get('name') + '.ref')

    if not os.path.isfile(PATH):
        with open(PATH, 'wb') as file:
            pickle.Pickler(file).dump(dict())

    with open(PATH, 'rb') as file:
        operators_run = pickle.Unpickler(file).load()

        if precond not in operators_run.keys():
            # Set a CG run of reference
            rhs = list()
            iterations = list()
            lin_op = SelfAdjointMatrix(operator.get('operator'), def_pos=True)
            M = AlgebraicPreconditionerFactory(lin_op).get(precond)

            # Run the CG with several left-hand sides
            for i in range(n_runs):
                # Create random linear system
                lin_sys = LinearSystem(lin_op, lin_op.dot(numpy.random.randn(n, 1)))

                # Run the Conjugate Gradient
                cg = ConjugateGradient(lin_sys, x_0=None, M=M)
                cg.run()

                # Store results
                rhs.append(lin_sys.rhs)
                iterations.append(cg.output['n_iterations'])

            # Get the index of the average run
            n_iterations = numpy.asarray(iterations)
            average_run = int(numpy.argmin(numpy.abs(n_iterations - numpy.mean(n_iterations))))

            # Set the average linear system
            lin_sys = LinearSystem(lin_op, rhs[average_run])

            # Run the Conjugate Gradient on this average linear system
            reference_cg = ConjugateGradient(lin_sys, x_0=None, M=M, arnoldi=True, buffer=numpy.Inf)
            reference_cg.run()

            operators_run[precond] = reference_cg

            # Save results obtained
            with open(PATH, 'wb') as new_file:
                pickle.Pickler(new_file).dump(operators_run)

    return operators_run[precond]


def initialize_report(operator: dict, setup: dict, reference: ConjugateGradient) -> str:
    """
    Method to create a report file and write the header.

    :param operator: dictionary containing the operator to benchmark on and its metadata.
    :param setup: dictionary containing the setup parameters.
    :param reference: ConjugateGradient object with convergence data of the reference run.
    """
    # Date and time fore report identification
    date_ = ''.join(time.strftime("%x").split('/'))
    time_ = ''.join(time.strftime("%X").split(':'))

    # Sampling method and its parameters
    sampling = setup['subspace']['type'] + '#' + str(setup['subspace']['parameters'])

    # Set the report name
    report_name = '_'.join([operator.get('name'), date_, time_, sampling]) + '.rpt'

    print(report_name)

    # Aggregate metadata of both the operator tested and the benchmark itself
    operator_metadata = ', '.join([str(operator.get('rank')),
                                  str(operator.get('non_zeros')),
                                  str(operator.get('conditioning')),
                                  operator.get('source')])

    benchmark_metadata = ', '.join([setup['first_level_precond'],
                                   str(reference.output['n_iterations']),
                                   str(setup['subspace']['type']),
                                    str(setup['subspace']['parameters'])])

    # Header writing
    with open('reports/' + report_name, 'w') as report_file:
        report_file.write('>>   REPORT OF PRECONDITIONING STRATEGIES BENCHMARK   << \n')
        report_file.write('> \n')
        report_file.write('>  SOLVER ............. Conjugate Gradient \n')
        report_file.write('>  SUBSPACES TESTED ... ' + str(setup['n_subspaces']) + '\n')
        report_file.write('>  RUNS ............... ' + str(setup['n_tests']) + '\n')
        report_file.write('>  PROBLEM NAME ....... ' + operator.get('name') + '\n')
        report_file.write('>  REFERENCE RUN ...... ' + str(reference.output['n_iterations']) + '\n')
        report_file.write('> \n')
        report_file.write('>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<< \n')
        report_file.write('~operator_metadata: ' + operator_metadata + '\n')
        report_file.write('~benchmark_metadata: ' + benchmark_metadata + '\n')
        report_file.write('>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<< \n')

    return report_name


def benchmark(setup: dict,
              subspaces: list,
              reference: ConjugateGradient) -> None:
    """
    Process the benchmark of an operator with the specification contained in the setup
    dictionary. Compare the operator to a reference run and store the results in text file.
    
    :param setup: Dictionary with benchmark parameters.
    :param subspaces: List of subspaces size to test.
    :param reference: ConjugateGradient object meant to be the reference to compare with.
    """

    # Initialize variables
    lin_sys = reference.lin_sys
    lin_op = lin_sys.lin_op
    first_level_precond = reference.M_i
    n, _ = lin_op.shape

    krylov_lin_sys = LinearSystem(lin_op, lin_op.dot(numpy.random.randn(n, 1)))

    # Process the benchmark
    for k in tqdm.tqdm(subspaces):
        report_line = '{:4} | '.format(k)

        # Create the subspace factory producing subspaces of shape (n, k)
        if setup['subspace']['type'] in KrylovSubspaceFactory.basis:
            factory = KrylovSubspaceFactory((n, k), krylov_lin_sys, M=first_level_precond)

        elif setup['subspace']['type'] in RandomSubspaceFactory.samplings:
            factory = RandomSubspaceFactory((n, k))
        else:
            raise ValueError('Subspace type unrecognized.')

        for i in range(setup['n_tests']):
            subspace = factory.get(setup['subspace']['type'], setup['subspace']['parameters'])
            lmp = LimitedMemoryPreconditioner(lin_op, subspace, M=first_level_precond)

            cg = ConjugateGradient(lin_sys, x_0=None, M=lmp)
            cg.run()

            report_line += str(cg.output['n_iterations'] / reference.output['n_iterations']) + ','

        # Write result in report text file
        with open(os.path.join(REPORTS_ROOT_PATH, setup['report_name']), 'a') as report_file:
            report_file.write(report_line[:-1] + '\n')
