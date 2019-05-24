#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on May 21, 2019 at 11:00.

@author: a.scotto

Description:
"""

import pickle


def load_problem(file_name):

    with open('problems/' + file_name, 'rb') as file:
        p = pickle.Unpickler(file)
        problem = p.load()

    return problem


def print_problem(problem):

    print('Problem {}: shape {} | Conditioning = {:1.2e}'
          .format(problem['name'], problem['shape'], problem['conditioning']))
