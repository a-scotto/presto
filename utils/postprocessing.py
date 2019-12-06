#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on September 23, 2019 at 15:22.

@author: a.scotto

Description:
"""

import os
import numpy


def arrange_report(reports: list) -> dict:
    """
    Method to arrange the reports file so as to gather the reports concerning the same operators.

    :param reports: List of reports paths to sort.
    """
    # Initialize variable
    arranged_reports = dict()

    for REPORT_PATH in reports:
        # Skip not concerned files
        if not REPORT_PATH.endswith('.rpt'):
            continue

        _, report_name = os.path.split(REPORT_PATH)

        operator = report_name.split('_')[0]

        # Aggregate the reports path according to the problem tested
        if operator not in arranged_reports.keys():
            arranged_reports[operator] = [REPORT_PATH]
        else:
            arranged_reports[operator].append(REPORT_PATH)

    return arranged_reports


def read_report(REPORT_PATH: str) -> tuple:
    """
    Method to read report text files (.rpt files) to extract metadata and data from a benchmark.

    :param REPORT_PATH: Path of the report to read.
    """
    # Initialize variables
    metadata = dict()
    data = dict()

    # Open up the report text file
    with open(REPORT_PATH, 'r') as report:
        # Get the list of text lines

        for line in report.readlines():
            # Skip report header
            if line.startswith('>'):
                continue

            # Read and store operator metadata
            elif line.startswith('~operator_metadata'):
                _metadata = line.split(': ')[1]
                size, non_zeros, conditioning, source = _metadata.split(', ')

                metadata['size'] = int(size)
                metadata['nnz'] = int(non_zeros)
                try:
                    metadata['cond'] = float(conditioning)
                except ValueError:
                    metadata['cond'] = None

                metadata['source'] = source

            # Read and store benchmark setup metadata
            elif line.startswith('~benchmark_metadata'):
                _metadata = line.split(': ')[1]
                first_lvl_precond, reference, subspace_type, subspace_param = _metadata.split(', ')

                metadata['first_lvl_preconditioner'] = first_lvl_precond
                metadata['reference'] = int(reference)
                metadata['subspace_type'] = subspace_type
                try:
                    metadata['subspace_parameter'] = float(subspace_param)
                except ValueError:
                    metadata['subspace_parameter'] = None

            # Read algorithm runs data
            else:
                # Remove spaces
                line = line.replace(' ', '')

                # Retrieve and convert data
                subspace_size, performances = line.split('|')
                performances = performances.split(',')
                performances = [float(p_i) for p_i in performances]

                data[subspace_size] = numpy.asarray(performances)

    return metadata, data


def convert_to_dense(k: numpy.array, lin_op_size: int, lin_op_order: int) -> numpy.array:
    """
    Method to convert a sparse subspace size to a dense subspace size keeping the criterion chosen
    constant (either memory requirements or application cost).

    :param k: Sparse subspace size to convert.
    :param lin_op_size: Linear operator involved in the conversion.
    :param lin_op_order: Linear operator involved in the conversion.
    """

    # Sanitize argument
    if isinstance(k, int) or isinstance(k, float) or isinstance(k, list):
        k = numpy.asarray(k)
    elif not isinstance(k, numpy.ndarray):
        raise ValueError('Subspaces must be of type numpy.array or of convertible type.')

    # Process conversion
    application_cost = 4*k**2 + 4*lin_op_size + lin_op_order

    converted_subspaces = numpy.asarray(application_cost / (8*lin_op_order), dtype=numpy.int32)

    return converted_subspaces


def process_data(data: dict) -> tuple:
    """
    Method to process the data from a report. Namely, compute statistics if necessary and convert to
    a suitable format for plots.

    :param data: Dictionary of data extracted from a report text file.
    """
    # Initialize variables
    subspace_sizes = list()
    mean = list()
    standard_deviation = list()

    for subspace_size, performances in data.items():
        subspace_sizes.append(float(subspace_size))
        mean.append(float(numpy.mean(performances)))
        standard_deviation.append(float(numpy.std(performances)))

    subspace_sizes = numpy.asarray(subspace_sizes)
    mean = numpy.asarray(mean)
    standard_deviation = numpy.asarray(standard_deviation)

    processed_data = dict(mean=mean,
                          standard_deviation=standard_deviation)

    return subspace_sizes, processed_data
