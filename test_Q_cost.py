# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 10:14:40 2019

@author: Nino
"""

import roe
import pvm
import numpy as np
import timeit


# TEST PARAMETERS
N = 100_000  # sample size
runs = 5  # number of runs


def get_flux_matrix(u1, u2, c1sq, c2sq, r, g=9.8):
    # matrix B (two-layer coupling)
    B = np.zeros([N, 4, 4])
    B[:, 1, 2] = -c1sq
    B[:, 3, 0] = -c2sq * r
    # Jasobian matrix
    J = np.zeros([N, 4, 4])
    J[:, 0, 1] = 1
    J[:, 1, 0] = - u1*u1 + c1sq
    J[:, 1, 1] = 2*u1
    J[:, 2, 3] = 1
    J[:, 3, 2] = - u2*u2 + c2sq
    J[:, 3, 3] = 2*u2
    # flux matrix
    return J - B


def generate_params(N, g=9.8):
    r = 0.98
    u1 = np.random.rand(N)*0.3
    u2 = -np.random.rand(N)*0.3
    h1 = np.random.rand(N)*1 + 1
    h2 = np.random.rand(N)*1 + 1
    c1sq = g * h1
    c2sq = g * h2
    return u1, u2, c1sq, c2sq, r, g


def rmse(prediction, target):
    return np.sqrt(np.mean((prediction-target)**2))


def test_comp_speed(params, A, runs, scheme, eig_type):
    if scheme == 'roe':
        func = roe
    elif scheme[:3] == 'pvm' or scheme == 'ifcp':
        func = pvm
    else:
        raise ValueError('unknown scheme {}'.format(scheme))
    u1, u2, c1sq, c2sq, r, g = params
    times = []
    for i in range(runs):
        start = timeit.default_timer()
        func.comp_Q(u1, u2, c1sq, c2sq, r, g, A,
                    scheme=scheme, eig_type=eig_type)
        stop = timeit.default_timer()
        time = stop - start
        times.append(time)
    print('{} scheme, {} eigenvalues: {:.2f} ms (best of {} runs)'
          .format(scheme, eig_type,
                  np.min(np.array(times))*1000, runs))


params = generate_params(N)
A = get_flux_matrix(*params)
print('\nCPU time analysis:')
test_comp_speed(params, A, runs, 'roe', 'numerical')
test_comp_speed(params, A, runs, 'roe', 'analytical')
test_comp_speed(params, A, runs, 'roe', 'approximated')
test_comp_speed(params, A, runs, 'ifcp', 'analytical')
test_comp_speed(params, A, runs, 'ifcp', 'approximated')
test_comp_speed(params, A, runs, 'pvm2', 'analytical')
test_comp_speed(params, A, runs, 'pvm2', 'approximated')
test_comp_speed(params, A, runs, 'pvm4', 'analytical')
test_comp_speed(params, A, runs, 'pvm4', 'approximated')
test_comp_speed(params, A, runs, 'pvm_roe', 'analytical')
test_comp_speed(params, A, runs, 'pvm_roe', 'approximated')



