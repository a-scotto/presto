#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on September 06, 2019 at 09:43.

@author: a.scotto

Description:
"""

import os
import tqdm
import time
import numpy
import scipy.sparse

from core.linear_system import ConjugateGradient, LinearSystem
from core.random_subspace import RandomSubspaceFactory
from core.preconditioner import LimitedMemoryPreconditioner, DiagonalPreconditioner
from core.linear_operator import SelfAdjointMatrix


def read_setup(OPERATOR_ROOT_PATH, setup_file_path):
    """
    Method to read the benchmark setup text file and return the different setup in a suitable
    format to run the benchmark.

    :param OPERATOR_ROOT_PATH: Path of folder containing the operators
    :param setup_file_path: Path of the setup text file
    """

    # Get the list of all available operators in OPERATOR_ROOT_PATH folder
    operators_list = os.listdir(OPERATOR_ROOT_PATH)
    operators_list.remove('__pycache__')

    setups = dict()

    # Read the setup text file
    with open(setup_file_path, 'r') as setup_text_file:
        # Get the list of text lines
        content = setup_text_file.readlines()

        # Go through all the lines
        for line in content:
            # Skip user dedicated header
            if line.startswith('>'):
                continue

            # Turn spaced separated data into list
            operators, n_subspaces, max_ratio, n_tests, sampling = line.split(' ')

            # Get the sampling method and optionally arguments
            sampling_method, sampling_parameter = sampling.split(',')

            setup = dict(n_subspaces=int(n_subspaces),
                         max_ratio=float(max_ratio),
                         n_tests=int(n_tests),
                         sampling_method=sampling_method,
                         sampling_parameter=float(sampling_parameter))

            # Either add all operators or the desired one to the setups dictionary
            if operators == '*':
                for operator in operators_list:
                    OPERATOR_PATH = os.path.join(OPERATOR_ROOT_PATH, operator)

                    # Add the operator path if already existing, else create the list
                    try:
                        setups[OPERATOR_PATH].append(setup)
                    except KeyError:
                        setups[OPERATOR_PATH] = [setup]
            else:
                OPERATOR_PATH = os.path.join(OPERATOR_ROOT_PATH, operators)

                if operators not in operators_list:
                    raise FileNotFoundError('Not such operator file {}.'.format(OPERATOR_PATH))

                # Add the operator path if already existing, else create the list
                try:
                    setups[OPERATOR_PATH].append(setup)
                except KeyError:
                    setups[OPERATOR_PATH] = [setup]

    return setups


def set_reference_run(RUN_STORAGE_PATH, lin_op, precond, n_runs=50):
    """
    Method to run the Conjugate Gradient a given number of times, to select the left-hand side corresponding to the
    average run.

    :param RUN_STORAGE_PATH: Path to store the results.
    :param lin_op: LinearOperator object to run the Conjugate Gradient on.
    :param n_runs: Number of runs to compute the average on.
    :return:
    """

    # Either load left-hand side from existing file or compute it
    try:
        reference = numpy.loadtxt(RUN_STORAGE_PATH)
        n_iterations = int(reference[0])
        lhs = reference[1:].reshape(-1, 1)

    except OSError:
        # Set a CG run of reference
        lhs = list()
        n_iterations = list()

        # Run the CG with several left-hand sides
        for i in range(n_runs):
            # Create random left-hand side
            y = numpy.random.randn(lin_op['shape'][0], 1)
            lin_sys = LinearSystem(lin_op, lin_op.dot(y))

            # Run the Conjugate Gradient
            cg = ConjugateGradient(lin_sys, x_0=None, M=precond).run()

            # Store results
            lhs.append(lin_sys.lhs)
            n_iterations.append(cg['n_it'])

        # Get the left-hand side the closest to the average run
        average_run = int(numpy.argmin(numpy.asarray(n_iterations) - numpy.mean(n_iterations)))
        lhs = lhs[average_run]
        n_iterations = n_iterations[average_run]

        # Store left-hand side and iterations number in a single stacked array
        reference = numpy.vstack([n_iterations, lhs])
        numpy.savetxt(RUN_STORAGE_PATH, reference)

    reference_run = dict(lhs=lhs, n_iterations=n_iterations)

    return reference_run


def initialize_report(operator, setup):
    """
    Method to create a report file and write the header.

    :param operator: LinearOperator object to benchmark on.
    :param setup: dictionary containing the setup parameters.
    """
    # Date and time fore report identification
    date_ = ''.join(time.strftime("%x").split('/'))
    time_ = ''.join(time.strftime("%X").split(':'))

    # Sampling method and eventual parameters
    param = '' if setup['sampling_parameter'] is None else str(setup['sampling_parameter'])
    sampling = setup['sampling_method'] + param

    # Set the report name from metadata above
    report_name = '_'.join([operator['name'], date_, time_, sampling])

    operator_metadata = '#'.join([str(operator['rank']),
                                  str(operator['non_zeros']),
                                  str(operator['conditioning']),
                                  operator['source']])

    benchmark_metadata = '#'.join([str(setup['reference']),
                                   str(param)])

    # Header writing
    with open('reports/' + report_name, 'w') as report_file:
        report_file.write('>>   REPORT OF PRECONDITIONING STRATEGIES BENCHMARK   << \n')
        report_file.write('> \n')
        report_file.write('>  SOLVER ................. Conjugate Gradient \n')
        report_file.write('>  NUMBER OF TESTS ........ ' + str(setup['n_subspaces']) + '\n')
        report_file.write('>  RUNS PER TEST .......... ' + str(setup['n_tests']) + '\n')
        report_file.write('>  MINIMAL SUBSPACE SIZE .. ' + str(setup['sto_subspaces'][0]) + '\n')
        report_file.write('>  MAXIMAL SUBSPACE SIZE .. ' + str(setup['sto_subspaces'][-1]) + '\n')
        report_file.write('>  PROBLEM NAME ........... ' + operator['name'] + '\n')
        report_file.write('>  REFERENCE RUN ...........' + str(setup['reference']) + '\n')
        report_file.write('> \n')
        report_file.write('>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<< \n')
        report_file.write('~operator_metadata#' + operator_metadata + '\n')
        report_file.write('~benchmark_metadata#' + benchmark_metadata + '\n')
        report_file.write('>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<< \n')

    return report_name


def benchmark(lin_sys, setup):
    lin_op = lin_sys.lin_op
    n, _ = lin_op.shape

    lin_sys_lmp = LinearSystem(lin_op, lin_op.dot(numpy.random.randn(n, 1)))
    cg = ConjugateGradient(lin_sys_lmp, x_0=None, M=setup['first_level_precond'], tol=1e-8).run()

    # Test all the subspace sizes
    for i in tqdm.tqdm(range(setup['n_subspaces'])):

        k_sto = setup['sto_subspaces'][i]
        k_det = setup['det_subspaces'][i]

        factory = RandomSubspaceFactory((n, k_sto))

        subspace = numpy.hstack(cg['p'][:k_det])
        lmp_det = LimitedMemoryPreconditioner(lin_op, subspace, M=setup['first_level_precond'])
        pcg = ConjugateGradient(lin_sys, x_0=None, M=lmp_det).run()

        deterministic_report = str(k_det) + '_' + str(pcg['n_it'] / setup['reference'])[:4]
        stochastic_report = str(k_sto) + '_'

        best = numpy.Inf
        worse = 0

        # Run the number of tests specified
        for i in range(setup['n_tests']):
            # Generate the subspace from the factory
            subspace = factory.generate(setup['sampling'], setup['args'])

            # Build the LMP from the subspace generated
            lmp = LimitedMemoryPreconditioner(lin_op, subspace, M=setup['first_level_precond'])

            # Run the Preconditioned Conjugate Gradient
            pcg = ConjugateGradient(lin_sys, x_0=None, M=lmp).run()

            # Add result to the report data line
            deterministic_report += str(pcg['n_it'] / setup['reference'])[:6] + ','

            if k == subspace_sizes[-1]:
                if pcg_score > worse:
                    scipy.sparse.save_npz('reports/' + report_file_name + '_worse_' + str(k),
                                          subspace)
                    worse = pcg_score

                elif pcg_score < best:
                    scipy.sparse.save_npz('reports/' + report_file_name + '_best_' + str(k),
                                          subspace)
                    best = pcg_score

        # Remove last coma from report line
        REPORT_LINE = REPORT_LINE[:-1]

        # Write data line in report file
        with open('reports/' + report_file_name, 'a') as report_file:
            report_file.write(REPORT_LINE + '\n')
