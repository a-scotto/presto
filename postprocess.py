#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on August 20, 2019 at 09:09.

@author: a.scotto

Description:
"""

import argparse

from matplotlib import pyplot
from core.projection_subspace import DeterministicSubspaceFactory, RandomSubspaceFactory
from utils.postprocessing import arrange_report, convert_to_dense, process_data, read_report


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
parser.add_argument('reports',
                    nargs='*',
                    help='Paths of reports text files to post-process.')

args = parser.parse_args()

# Sort the reports by operator tested
reports = arrange_report(args.reports)

# Go through all the reports
for operator, REPORT_PATHS in reports.items():
    pyplot.figure()
    pyplot.grid()

    # Initialization for script coherence
    metadata, subspace_sizes = None, None

    # Browse through the reports file names in the report set
    for j, REPORT_PATH in enumerate(REPORT_PATHS):
        # Retrieve data from the report
        metadata, data = read_report(REPORT_PATH)
        # Process the data
        subspace_sizes, processed_data = process_data(data)

        if metadata['subspace_type'] in RandomSubspaceFactory.subspace_type.keys():
            _type = RandomSubspaceFactory.subspace_type[metadata['subspace_type']]

        elif metadata['subspace_type'] in DeterministicSubspaceFactory.subspace_type.keys():
            _type = DeterministicSubspaceFactory.subspace_type[metadata['subspace_type']]

        else:
            raise ValueError('Type of subspace not recognized, must be either dense or sparse.')

        if _type == 'sparse':
            subspace_sizes = convert_to_dense(subspace_sizes, metadata['nnz'], metadata['size'])

        label = metadata['subspace_type'].replace('_', ' ').capitalize()
        label = label + ': ' + metadata['subspace_parameter']

        pyplot.errorbar(subspace_sizes,
                        processed_data['mean'] * metadata['reference'],
                        yerr=processed_data['standard_deviation'] * metadata['reference'],
                        c=COLORS[j % len(COLORS)],
                        fmt='o-',
                        mec='k',
                        ms=4,
                        elinewidth=0.5,
                        ecolor='k',
                        capsize=2.5,
                        label=label)

    # Plot reference line
    pyplot.hlines(y=metadata['reference'],
                  xmin=0.,
                  xmax=subspace_sizes[-1],
                  linestyle='dashed',
                  label='PCG with M alone')

    # Set corresponding title regarding the comparison criterion
    plot_title = 'LMP results on $A=$ {}, $n=$ {}, $M=$ {}'\
        .format(operator, metadata['size'], metadata['first_precond'].capitalize())

    # Add the legend, title, and axis titles
    pyplot.legend()
    pyplot.title(plot_title)
    pyplot.xlabel('Equivalent dense subspace size.')
    pyplot.ylabel('Preconditioned CG number of iterations.')
    pyplot.xlim(left=0.)
    pyplot.ylim(bottom=0.)

pyplot.show()
