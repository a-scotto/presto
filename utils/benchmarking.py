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

from core.linear_operator import LinearOperator
from core.random_subspace import RandomSubspaceFactory
from core.linear_system import ConjugateGradient, LinearSystem
from core.preconditioner import LimitedMemoryPreconditioner, Preconditioner


def read_setup(OPERATOR_ROOT_PATH: str, SETUP_FILE_PATH: str) -> dict:
    """
    Method to read the benchmark setup text file and return the different setup in a suitable
    format to run the benchmark.

    :param OPERATOR_ROOT_PATH: Path of folder containing the operators.
    :param SETUP_FILE_PATH: Path of the setup text file.
    """

    # Get the list of all available operators in OPERATOR_ROOT_PATH folder
    operators_list = os.listdir(OPERATOR_ROOT_PATH)
    operators_list = fnmatch.filter(operators_list, '*.opr')

    # Initialize the dictionary
    setups = dict()

    # Read the setup text file
    with open(SETUP_FILE_PATH, 'r') as setup_text_file:
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

                    # Append the operator path if already existing, else create the list
                    try:
                        setups[OPERATOR_PATH].append(setup)
                    except KeyError:
                        setups[OPERATOR_PATH] = [setup]
            else:
                operators = operators + '.opr'
                OPERATOR_PATH = os.path.join(OPERATOR_ROOT_PATH, operators)

                if operators not in operators_list:
                    raise FileNotFoundError('Not such operator file {}.'.format(OPERATOR_PATH))

                # Append the operator path if already existing, else create the list
                try:
                    setups[OPERATOR_PATH].append(setup)
                except KeyError:
                    setups[OPERATOR_PATH] = [setup]

    return setups


def set_reference_run(REFERENCE_RUN_PATH: str,
                      lin_op: LinearOperator,
                      precond: Preconditioner,
                      n_runs: int = 100) -> ConjugateGradient:
    """
    Method to run the Conjugate Gradient a given number of times, to select the left-hand side
    corresponding to the average run.

    :param REFERENCE_RUN_PATH: Path to store the results.
    :param lin_op: LinearOperator object to run the Conjugate Gradient on.
    :param precond: First-level preconditioner to run the CG with.
    :param n_runs: Number of runs to compute the average on.
    """

    # Either load left-hand side from existing file or compute it
    try:
        with open(REFERENCE_RUN_PATH, 'rb') as file:
            reference_run = pickle.Unpickler(file).load()

    except OSError:
        # Set a CG run of reference
        lhs = list()
        n_iterations = list()

        # Run the CG with several left-hand sides
        for i in range(n_runs):
            # Create random left-hand side
            y = numpy.random.randn(lin_op.shape[0], 1)
            lin_sys = LinearSystem(lin_op, lin_op.dot(y))

            # Run the Conjugate Gradient
            cg = ConjugateGradient(lin_sys, x_0=None, M=precond)
            cg.run()

            # Store results
            lhs.append(lin_sys.lhs)
            n_iterations.append(cg.output['n_iterations'])

        # Get the index of the average run
        n_iterations = numpy.asarray(n_iterations)
        average_run = int(numpy.argmin(numpy.abs(n_iterations - numpy.mean(n_iterations))))

        # Set the average linear system
        lin_sys = LinearSystem(lin_op, lhs[average_run])

        # Run the Conjugate Gradient on this average linear system
        reference_run = ConjugateGradient(lin_sys,
                                          x_0=None,
                                          M=precond,
                                          tol=1e-8,
                                          arnoldi=True,
                                          buffer=numpy.Inf)
        reference_run.run()

        # Save results obtained
        with open(REFERENCE_RUN_PATH, 'wb') as file:
            pickle.Pickler(file).dump(reference_run)

    return reference_run


def initialize_report(operator: dict, setup: dict) -> str:
    """
    Method to create a report file and write the header.

    :param operator: dictionary containing the operator to benchmark on and its metadata.
    :param setup: dictionary containing the setup parameters.
    """
    # Date and time fore report identification
    date_ = ''.join(time.strftime("%x").split('/'))
    time_ = ''.join(time.strftime("%X").split(':'))

    # Sampling method and its parameters
    sampling = setup['sampling_method'] + str(setup['sampling_parameter'])

    # Set the report name
    report_name = '_'.join([operator['name'], date_, time_, sampling])

    # Aggregate metadata of both the operator tested and the benchmark itself
    operator_metadata = '#'.join([str(operator['rank']),
                                  str(operator['non_zeros']),
                                  str(operator['conditioning']),
                                  operator['source']])

    benchmark_metadata = '#'.join([setup['first_lvl_preconditioner'],
                                   str(setup['reference']),
                                   str(setup['sampling_parameter'])])

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


def benchmark(reference_run: ConjugateGradient, setup: dict) -> None:
    """
    Process the benchmark of an operator with the specification contained in the setup
    dictionary. Compare the operator to a reference run and store the results in text file.

    :param reference_run: ConjugateGradient object meant to be the reference to compare with.
    :param setup: Dictionary with parameters of the benchmark.
    """

    # Get LinearOperator and Preconditioner from the reference run
    lin_op = reference_run.lin_sys.lin_op
    first_level_precond = reference_run.M_i
    n, _ = lin_op.shape

    # Run another Conjugate Gradient to build a non-informed LMP
    lin_sys_lmp_det = LinearSystem(lin_op, lin_op.dot(numpy.random.randn(n, 1)))
    cg = ConjugateGradient(lin_sys_lmp_det,
                           x_0=None,
                           M=first_level_precond,
                           tol=1e-5,
                           arnoldi=True,
                           buffer=numpy.Inf)
    cg.run()

    # Compute the Ritz vectors from the Arnoldi relation, i.e. the tridiagonal matrix
    _, eig_vectors = numpy.linalg.eig(cg.output['arnoldi'].todense())
    ritz_vectors = numpy.hstack(cg.output['z']).dot(eig_vectors)

    # Initialize variables to avoid code inspection troubles
    k_det, k_sto = None, None
    best_score, worst_score = numpy.Inf, 0.
    best_subspace, worst_subspace = None, None
    directions_subspace, ritz_subspace = None, None

    # Test all the subspace sizes provided
    for i in tqdm.tqdm(range(setup['n_subspaces'])):

        # Deterministic reporting
        k_det = setup['det_subspaces'][i]
        deterministic_report = list()

        # LMP with the k_det first informed descent directions
        directions_subspace = numpy.hstack(reference_run.output['p'][:k_det])
        lmp_det = LimitedMemoryPreconditioner(lin_op, directions_subspace, M=first_level_precond)
        pcg = ConjugateGradient(reference_run.lin_sys, x_0=None, M=lmp_det)
        pcg.run()

        deterministic_report.append(str(pcg.output['n_iterations'] / setup['reference']))

        # LMP with the k_det first non-informed Ritz vectors
        ritz_subspace = ritz_vectors[:, :k_det]
        lmp_det = LimitedMemoryPreconditioner(lin_op, ritz_subspace, M=first_level_precond)
        pcg = ConjugateGradient(reference_run.lin_sys, x_0=None, M=lmp_det)
        pcg.run()

        deterministic_report.append(str(pcg.output['n_iterations'] / setup['reference']))

        # Finalize deterministic report
        deterministic_report = str(k_det) + '_' + ','.join(deterministic_report)

        # Stochastic reporting
        k_sto = setup['sto_subspaces'][i]
        stochastic_report = list()

        factory = RandomSubspaceFactory((n, k_sto))

        # Run the number of tests specified
        for j in range(setup['n_tests']):
            # Generate the subspace from the factory
            subspace = factory.generate(setup['sampling_method'], setup['sampling_parameter'])

            # Build the LMP from the subspace generated
            lmp_sto = LimitedMemoryPreconditioner(lin_op, subspace, M=first_level_precond)

            # Run the Preconditioned Conjugate Gradient
            pcg = ConjugateGradient(reference_run.lin_sys, x_0=None, M=lmp_sto)
            pcg.run()

            # Add result to the report data line
            lmp_sto_score = pcg.output['n_iterations'] / setup['reference']
            stochastic_report.append(str(lmp_sto_score))

            # At last subspace size tested, store the best and worst subspaces
            if i == setup['n_subspaces'] - 1:
                if lmp_sto_score > worst_score:
                    worst_score = lmp_sto_score
                    worst_subspace = subspace

                if lmp_sto_score < best_score:
                    best_score = lmp_sto_score
                    best_subspace = subspace

        # Finalize deterministic report
        stochastic_report = str(k_sto) + '_' + ','.join(stochastic_report)

        # Remove last coma from report line
        REPORT_LINE = '#'.join([deterministic_report, stochastic_report])

        # Write data line in report file
        with open('reports/' + setup['report_path'], 'a') as report_file:
            report_file.write(REPORT_LINE + '\n')

    # Save the best and worst subspace encountered for further analysis
    scipy.sparse.save_npz('reports/' + setup['report_path'] + '_worst_' + str(k_sto),
                          worst_subspace)

    scipy.sparse.save_npz('reports/' + setup['report_path'] + '_best_' + str(k_sto),
                          best_subspace)

    # Save the corresponding deterministic subspaces with Ritz vectors and descent directions
    numpy.save('reports/' + setup['report_path'] + '_ritz_' + str(k_det), ritz_subspace)
    numpy.save('reports/' + setup['report_path'] + '_dir_' + str(k_det), directions_subspace)
