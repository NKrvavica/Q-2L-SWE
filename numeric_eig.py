# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 10:44:37 2019

@author: Nino Krvavica
"""

import numpy as np
from scipy import linalg


def numeric_eig(A, eigvecs=True, hyp_corr=True):
    ''' Computes real numerical eigenvalues `Lam` and eigenvectors `K`
    of matrix `A`::

        A = [[0, 1, 0, 0],
             [c1**2-u1**2, 2*u1, c1**2, 0],
             [0, 0, 0, 1],
             [r*c2**2, 0, c2**2 - u2**2, 2*u2]]

    If complex eigenvalues are computed, it corrects them by using real Jordan
    decomposition.

    Parameters
    ----------
    A: ndarray
        stacked array of flux matrix
    eigvecs: bool, optional
        if set to `True` the function computes and returns eigenvalues and
        eigenvectors, otherwise it returns only eigenvalues
    hyp_corr: bool, optional
        is set to `True` the function performs the hyperbolicity correction
		where needed.

    Returns
    -------
    Lam: ndarray
        stacked eigenvalue arrays
    K: ndarray or empty
        stacked 4x4 matrices whose columns are right eigenvectors (empty if
        eigvecs is set to `False`)
    '''

    # Get eigenvalues and eigenvectors
    if eigvecs:
        Lam, K = np.linalg.eig(A)  # compute eigenvalues and eigenvectors
    else:
        Lam = np.linalg.eigvals(A)  # compute only eigenvalues
        K = []

    ''' If complex eigenvalues are found and eigenvectors are available,
    apply real Jordan decomposition, if only eigenvalues, take real part'''
    if hyp_corr and eigvecs:
        if np.iscomplex(Lam).any():
            idx = np.unique(np.argwhere(np.iscomplex(Lam))[:, 0])
            L_real, K_real = linalg.cdf2rdf(Lam[idx, :], K[idx, :])
            Lam = Lam.real
            for j in range(4):
                Lam[idx, j] = L_real[:, j, j]
            K = K.real
            K[idx, :, :] = K_real[:, :, :]
    elif hyp_corr:
        if np.iscomplex(Lam).any():
            Lam = Lam.real

    return Lam, K
