# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 22:17:29 2019

@author: Nino
"""

import numeric_eig
import analytic_eig
import timeit
import numpy as np


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


# Generate random parameters
def generate_params(N, g=9.8):
    r = 0.98
    u1 = np.random.rand(N)*0.3
    u2 = -np.random.rand(N)*0.3
    h1 = np.random.rand(N)*1 + .01
    h2 = np.random.rand(N)*1 + .01
    c1sq = g * h1
    c2sq = g * h2
    return u1, u2, c1sq, c2sq, r, g


def test_comp_speed(eigvecs=False, hyp_corr=True):
    times = []
    for i in range(runs):
        start = timeit.default_timer()
        L_num, K_num = numeric_eig.numeric_eig(A, eigvecs=eigvecs,
                                               hyp_corr=hyp_corr)
        stop = timeit.default_timer()
        time = stop - start
        times.append(time)
    print('Numerical eigestructure: {:.2f} ms (best of {} runs)'
          .format(np.min(np.array(times))*1000, runs))
    times = []
    for i in range(runs*10):
        start = timeit.default_timer()
        L_an, K_an, F_an = analytic_eig.analytic_eig(u1, u2, c1sq, c2sq, r, g,
                                                     eigvecs=eigvecs,
                                                     hyp_corr=hyp_corr)
        stop = timeit.default_timer()
        time = stop - start
        times.append(time)
    print('Analytical eigestructure: {:.2f} ms (best of {} runs)'
          .format(np.min(np.array(times))*1000, runs))
    times = []
    for i in range(runs*10):
        start = timeit.default_timer()
        L_app, K_app, F_app = analytic_eig.approx_eig(u1, u2, c1sq, c2sq, r, g,
                                                      eigvecs=eigvecs,
                                                      hyp_corr=hyp_corr)
        stop = timeit.default_timer()
        time = stop - start
        times.append(time)
    print('Approximated eigestructure: {:.2f} ms (best of {} runs)'
          .format(np.min(np.array(times))*1000, runs))


u1, u2, c1sq, c2sq, r, g = generate_params(N)
A = get_flux_matrix(u1, u2, c1sq, c2sq, r, g)
print('\nCPU time analysis (only eigenvalues):')
test_comp_speed(eigvecs=False)
print('\nCPU time analysis (entire eigenstructure):')
test_comp_speed(eigvecs=True)