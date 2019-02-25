# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 22:17:29 2019

@author: Nino
"""

import numeric_eig
import analytic_eig
import numpy as np


# TEST PARAMETERS
N = 10_000  # sample size
rs = [0.98, 0.8, 0.6, 0.4]  # different r values


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
    u1 = np.random.rand(N)*0.3
    u2 = -np.random.rand(N)*0.3
    h1 = np.random.rand(N)*1 + 1
    h2 = np.random.rand(N)*1 + 1
    c1sq = g * h1
    c2sq = g * h2
    return u1, u2, c1sq, c2sq, g


def max_abs_error(target, prediction):
    return np.max(np.abs(prediction - target))


def find_errors(u1, u2, c1sq, c2sq, r, g, A, hypcorr=True):
    L_num, K_num = numeric_eig.numeric_eig(A, hyp_corr=hypcorr)
    L_an, K_an, F_an = analytic_eig.analytic_eig(u1, u2, c1sq, c2sq, r, g,
                                                 hyp_corr=hypcorr)
    L_app, K_app, F_app = analytic_eig.approx_eig(u1, u2, c1sq, c2sq, r, g,
                                                  hyp_corr=hypcorr)
    L_num = np.sort(L_num)
    max_err_cf = max_abs_error(L_num, L_an)
    max_err_app = max_abs_error(L_num, L_app)
    print('Closed-form eigenvalues max. AE (r={}): {:.3e}'
          .format(r, max_err_cf))
    print('Approximated eigenvalues max. AE (r={}): {:.3e}'
          .format(r, max_err_app))
    return L_num, L_an, L_app


print('\nComputing accuracy of eigenvalues')
for r in rs:
    u1, u2, c1sq, c2sq, g = generate_params(N)
    A = get_flux_matrix(u1, u2, c1sq, c2sq, r, g)
    L_num, L_an, L_app = find_errors(u1, u2, c1sq, c2sq, r, g, A)
