# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 10:41:43 2019

@author: Nino Krvavica
"""

import numpy as np
import analytic_eig
import numeric_eig


def recompose_matrix(K, L):
    '''
    Recomposes a matrix from known eigenvalues `L` and eigevectors `K`.

    Instead of solving `Q = K@L@inv(K)` derived from equality `QK = KL`,
    the algorithm rewrites the equation as `K.T Q.T=(KL).T`,
    and uses `linalg.solve` to solve `Ax=B`, where `A=K.T`, `x=A.T`,
    and `B = (KL).T`. Finally `Q` is derived as `Q=x.T`.

    Parameters
    ----------
    K: stacked array
        matrices where columns are right eigenvectors
    L: stacked array
        matrices where diagonal values are eigenvalues

    Returns
    -------
    Q: stacked array
    '''
    A = K.transpose(0, 2, 1)
    B = (L[:, np.newaxis, :] * K).transpose(0, 2, 1)
    return np.linalg.solve(A, B).transpose(0, 2, 1)


def harten_regularization(Lam, eps=1e-1):
    ''' Performs Harten regularization, which is needed if one of the
    eigenvelues is zero.

    Parameters
    ----------
    Lam: ndarray
        stacked eigenvalue arrays
    eps: float, optional
        Harten's parameter

    Returns
    ------_
    A_abs: ndarray
        stacked arrays of apsolute values of eigenvalues,
        eigenvalues are corrected by Hartens regularizations if one of them
        is equal to zero.
    '''
    A_abs = np.abs(Lam)
    A_abs += (0.5 * ((1 + np.sign(eps - A_abs))
              * ((Lam**2 + eps*eps) / (2*eps) - A_abs)))
    return A_abs


def comp_Q(u1, u2, c1sq, c2sq, r, g, A, scheme='Roe', eig_type='numerical',
           hyp_corr=True):
    ''' Returns the numerical viscosity matrix of a two-layer shallow water
    system. Uses a Q-scheme of Roe.

    Parameters
    ----------
    u1: ndarray
        velocities of the upper layer
    u2: ndarray
        velocities of the lower layer
    c1sq: ndarray
        celerity of the upper layer (c1^2 = g * h1)
    c2sq: ndarray
        celerity of the lower layer (cw^2 = g * h2)
    r: flot or ndarray
        relative density `r = rho1/rho2`, where `rho1` and `rho2` are the
        respective densities of the upper and lower layer.
    g: float or ndarray, optional
        acceleration of gravity
    A: ndarray
        stacked flux Jacobian matrices of the two-layer shallow water system
    eig_type: string, optional
        type of eigenvalues ('numerical', 'analytical', 'approximated')
    hyp_corr: bool, optional
        if set to `True` hyperbolicity correction is performed

    Returns
    -------
    Q: ndarray
        stacked 4x4 numerical viscosity matrices
    P_plus: ndarray
        stacked 4x4 projection matrices of positive sign elements
    P_minus: ndarray
        stacked 4x4 projection matrices of negative sign elements
    Lam: ndarray
        stacked eigenvalue arrays
    F: ndarray
        array of correction friction (0 for hyperbolic system)
        '''
    # Get eigenvalues and eigenvectors
    if eig_type == 'numerical':
        Lam, K = numeric_eig.numeric_eig(A)
        F = np.zeros((Lam.shape[0], 1))
    elif eig_type == 'analytical':
        Lam, K, F = analytic_eig.analytic_eig(u1, u2, c1sq, c2sq, r, g,
                                              hyp_corr=hyp_corr)
    elif eig_type == 'approximated':
        Lam, K, F = analytic_eig.approx_eig(u1, u2, c1sq, c2sq, r, g,
                                            hyp_corr=hyp_corr)
    else:
        raise ValueError('''wrong type of calculation, expected either '''
                         ''''numerical', 'analytical', or 'approximated' ''')

    # Perform Harten regularization
    A_abs = harten_regularization(Lam)

    # Recompose viscosity and projection matrices
    Q = recompose_matrix(K, A_abs)
    C = recompose_matrix(K, np.sign(Lam))
    Id = np.eye(4, 4)
    Pp = 0.5 * (Id + C)  # positive part of the projection matrix
    Pm = 0.5 * (Id - C)  # negative part of the projection matrix

    return Q, Pp, Pm, Lam, F
