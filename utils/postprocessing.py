#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on September 23, 2019 at 15:22.

@author: a.scotto

Description:
"""

import numpy


def read_report(REPORT_PATH):

    metadata = dict()

    stochastic_results = dict(sizes=[], means=[], stdevs=[])
    deterministic_results = dict(sizes=[])

    # Open up the report text file
    with open(REPORT_PATH, 'r') as report:
        # Get the list of text lines

        content = report.readlines()

        for data in content:
            # Skip report header
            if data.startswith('>'):
                continue

            # Read and store operator metadata
            elif data.startswith('~operator_metadata'):
                _metadata = data.split('#')[1:]
                metadata['size'] = int(_metadata[0])
                metadata['nnz'] = int(_metadata[1])
                metadata['cond'] = float(_metadata[2])
                metadata['source'] = _metadata[3]

            # Read and storebenchmark setup metadata
            elif data.startswith('~benchmark_metadata'):
                _metadata = data.split('#')[1:]
                metadata['first_lvl_preconditioner'] = _metadata[0]
                metadata['reference'] = int(_metadata[1])
                metadata['sampling_parameter'] = float(_metadata[2])

            # Read algorithm runs data
            else:
                deterministic_data, stochastic_data = data.split('#')

                k_det, deterministic_perfs = deterministic_data.split('_')
                k_sto, stochastic_perfs = stochastic_data.split('_')

                # Convert data into list of float
                deterministic_perfs = list(map(lambda x: float(x), deterministic_perfs.split(',')))
                stochastic_perfs = list(map(lambda x: float(x), stochastic_perfs.split(',')))

                deterministic_results['sizes'].append(int(k_det))
                stochastic_results['sizes'].append(int(k_sto))

                for i in range(len(deterministic_perfs)):
                    try:
                        deterministic_results[str(i)].append(deterministic_perfs[i])
                    except KeyError:
                        deterministic_results[str(i)] = [deterministic_perfs[i]]

                stochastic_results['means'].append(numpy.mean(stochastic_perfs))
                stochastic_results['stdevs'].append(numpy.std(stochastic_perfs))

    stochastic_results['sizes'] = numpy.asarray(stochastic_results['sizes'])
    stochastic_results['means'] = numpy.asarray(stochastic_results['means'])
    stochastic_results['stdevs'] = numpy.asarray(stochastic_results['stdevs'])

    deterministic_results['sizes'] = numpy.asarray(deterministic_results['sizes'])
    for i in range(len(deterministic_results.keys()) - 1):
        deterministic_results[str(i)] = numpy.asarray(deterministic_results[str(i)])

    return metadata, deterministic_results, stochastic_results
