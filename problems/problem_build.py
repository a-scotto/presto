#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on May 21, 2019 at 10:38.

@author: a.scotto

Description:
"""

import pickle
import scipy.io

file_name = ''

obj = scipy.io.loadmat(file_name)['Problem'][0][0]

problem = dict()

problem['name'] = file_name
problem['shape'] = (8219, 8219)
problem['non_zeros'] = 242669
problem['source'] = 'Quantum Chemistry problem'
problem['conditioning'] = 1.454681e3
problem['rank'] = 8219
problem['symmetric'] = True
problem['def_pos'] = True

for i, obj_i in enumerate(obj):
    print('Element', i, ':', obj_i)
    print()

problem['operator'] = obj[2]

print(problem)

with open(file_name, 'wb') as file:

    p = pickle.Pickler(file)
    p.dump(problem)
