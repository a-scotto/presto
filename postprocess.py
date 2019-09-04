#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on August 20, 2019 at 09:09.

@author: a.scotto

Description:
"""

import os
import numpy
import argparse

from matplotlib import pyplot

COLORS = ['tab:blue',
          'tab:orange',
          'tab:green',
          'tab:red',
          'tab:purple',
          'tab:brown',
          'tab:pink',
          'tab:gray',
          'tab:olive',
          'tab:cyan']

MARKER = ['', 'o', 's', '^']

# Parse command line argument
parser = argparse.ArgumentParser()
parser.add_argument('-r',
                    nargs='*',
                    required=True,
                    dest='reports',
                    help='Paths of reports text files to post-process.')
parser.add_argument('-c',
                    required=True,
                    dest='comparison',
                    choices=['pb', 'spl'],
                    help='Comparison criteria, either problems or sampling methods.')
args = parser.parse_args()
comparison = args.comparison

reports_sets = dict()

# Pre-processing the files path provided regarding the comparison choice
for REPORT_PATH in args.reports:
    _, report_name = os.path.split(REPORT_PATH)

    problem_name = report_name.split('_')[0]
    sampling = report_name.split('_')[-1]

    # Sort the reports according to the comparison criterion provided
    if comparison == 'spl':
        if problem_name not in reports_sets.keys():
            reports_sets[problem_name] = [REPORT_PATH]
        else:
            reports_sets[problem_name].append(REPORT_PATH)
    else:
        if sampling not in reports_sets.keys():
            reports_sets[sampling] = [REPORT_PATH]
        else:
            reports_sets[sampling].append(REPORT_PATH)

# Go through all the reports
for key, reports_paths in reports_sets.items():
    pyplot.figure()
    pyplot.grid()

    # Set corresponding title regarding the comparison criterion
    if comparison == 'spl':
        plot_title = 'Sampling strategies results on ' + key + ' problem'
    else:
        plot_title = 'Operators preconditioned by ' + key + ' sampled subspaces.'

    # Browse through the reports file names in the report set
    for j, REPORT_PATH in enumerate(reports_paths):
        _, report_name = os.path.split(REPORT_PATH)

        if comparison == 'spl':
            label = report_name.split('_')[-1]
        else:
            label = report_name.split('_')[0]

        det_perfs = list()
        sto_sizes = list()
        det_sizes = list()
        means = list()
        stdevs = list()
        minima = list()
        maxima = list()

        # Read and store the data contained in reports files
        with open(REPORT_PATH, 'r') as report:
            # Get the list of text lines
            content = report.readlines()

            for data in content:
                # Skip report header
                if data[0] == '>':
                    pass

                # Read operator metadata
                elif '~operator_metadata' in data:
                    metadata = data.split('#')[1:]
                    size = int(metadata[0])
                    nnz = int(metadata[1])
                    cond = float(metadata[2])
                    source = metadata[3]

                # Read benchmark setup metadata
                elif '~benchmark_metadata' in data:
                    metadata = data.split('#')[1:]
                    reference_run = int(metadata[0])
                    d = float(metadata[1])

                # Read data
                else:
                    det_data, sto_data = data.split('#')
                    k_0, det_perf = det_data.split('_')
                    k, sto_perf = sto_data.split('_')

                    sto_perf = list(map(lambda x: float(x), sto_perf.split(',')))

                    sto_sizes.append(int(k))
                    det_sizes.append(int(k_0))

                    det_perfs.append(float(det_perf))
                    means.append(numpy.mean(sto_perf))
                    stdevs.append(numpy.std(sto_perf))
                    minima.append(min(sto_perf))
                    maxima.append(max(sto_perf))

        lower = [means[i] - stdevs[i] for i in range(len(means))]
        upper = [means[i] + stdevs[i] for i in range(len(means))]

        sto_sizes = numpy.asarray(sto_sizes)
        det_sizes = numpy.asarray(det_sizes)

        stochastic_cost = 4 * nnz + 6 * (sto_sizes * (1 - d) + size) + sto_sizes**2
        deterministic_cost = 8 * size * det_sizes

        # Add plot to the current figure
        pyplot.plot(stochastic_cost,
                    means,
                    linestyle='-',
                    mec='k',
                    ms=5,
                    marker=MARKER[(j // len(COLORS)) % len(MARKER)],
                    color=COLORS[j % len(COLORS)],
                    label=label)

        pyplot.fill_between(stochastic_cost, lower, upper, color=COLORS[j % len(COLORS)], alpha=0.2)
        pyplot.plot(stochastic_cost, minima, color=COLORS[j % len(COLORS)], marker='+', lw=0)
        pyplot.plot(deterministic_cost, det_perfs, color=COLORS[j % len(COLORS)], ls='--')

    # Add the legend, title, and axis titles
    pyplot.legend()
    pyplot.title(plot_title)
    pyplot.xlabel('Ratios of subspace size and problem size k / n')
    pyplot.ylabel('PCG iterations compared to ' + str(reference_run))

    # pyplot.xlim(subspace_sizes[0], subspace_sizes[-1])
    pyplot.ylim(bottom=0.)

pyplot.show()
