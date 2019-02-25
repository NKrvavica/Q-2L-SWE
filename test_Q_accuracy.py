# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 10:14:40 2019

@author: Nino
"""

import roe
import pvm
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


def max_ae(target, prediction):
    ''' Maximum absolute error '''
    target = target.sum(axis=1)
    prediction = prediction.sum(axis=1)
    return np.max(np.abs(prediction - target))


def comp_visc_matrix(params, A, scheme, eig_type):
    if scheme == 'roe':
        func = roe
    elif scheme[:3] == 'pvm' or scheme == 'ifcp':
        func = pvm
    else:
        raise ValueError('unknown scheme {}'.format(scheme))
    u1, u2, c1sq, c2sq, r, g = params
    Q, *_ = func.comp_Q(u1, u2, c1sq, c2sq, r, g, A,
                       scheme=scheme, eig_type=eig_type)
    if func == roe:
        return Q
    else:
        return Q @ A


def find_errors(u1, u2, c1sq, c2sq, r, g, A):
    params = u1, u2, c1sq, c2sq, r, g
    Q_nroe = comp_visc_matrix(params, A, 'roe', 'numerical')
    Q_aroe = comp_visc_matrix(params, A, 'roe', 'analytical')
    Q_eroe = comp_visc_matrix(params, A, 'roe', 'approximated')
    Q_aifcp = comp_visc_matrix(params, A, 'ifcp', 'analytical')
    Q_eifcp = comp_visc_matrix(params, A, 'ifcp', 'approximated')
    Q_apvm2 = comp_visc_matrix(params, A, 'pvm2', 'analytical')
    Q_epvm2 = comp_visc_matrix(params, A, 'pvm2', 'approximated')
    Q_apvm4 = comp_visc_matrix(params, A, 'pvm4', 'analytical')
    Q_epvm4 = comp_visc_matrix(params, A, 'pvm4', 'approximated')
    Q_apvmroe = comp_visc_matrix(params, A, 'pvm_roe', 'analytical')
    Q_epvmroe = comp_visc_matrix(params, A, 'pvm_roe', 'approximated')
    print('Roe solver with analytical eigenvalues, max AE (r={}): {:.3e}'
          .format(r, max_ae(Q_nroe, Q_aroe)))
    print('Roe solver with approximated eigenvalues, max AE (r={}): {:.3e}'
          .format(r, max_ae(Q_nroe, Q_eroe)))
    print('IFCP solver with analytical eigenvalues, max AE (r={}): {:.3e}'
          .format(r, max_ae(Q_nroe, Q_aifcp)))
    print('IFCP solver with approximated eigenvalues, max AE (r={}): {:.3e}'
          .format(r, max_ae(Q_nroe, Q_eifcp)))
    print('PVM-2U solver with analytical eigenvalues, max AE (r={}): {:.3e}'
          .format(r, max_ae(Q_nroe, Q_apvm2)))
    print('PVM-2U solver with approximated eigenvalues, max AE (r={}): {:.3e}'
          .format(r, max_ae(Q_nroe, Q_epvm2)))
    print('PVM-4 solver with analytical eigenvalues, max AE (r={}): {:.3e}'
          .format(r, max_ae(Q_nroe, Q_apvm4)))
    print('PVM-4 solver with approximated eigenvalues, max AE (r={}): {:.3e}'
          .format(r, max_ae(Q_nroe, Q_epvm4)))
    print('PVM-Roe solver with analytical eigenvalues, max AE (r={}): {:.3e}'
          .format(r, max_ae(Q_nroe, Q_apvmroe)))
    print('PVM-Roe solver with approximated eigenvalues, max AE (r={}): {:.3e}'
          .format(r, max_ae(Q_nroe, Q_epvmroe)))


print('\nComputing accuracy of viscosity matrices:')
for r in rs:
    u1, u2, c1sq, c2sq, g = generate_params(N)
    A = get_flux_matrix(u1, u2, c1sq, c2sq, r, g)
    find_errors(u1, u2, c1sq, c2sq, r, g, A)




