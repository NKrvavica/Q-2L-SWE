# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 17:29:53 2019

@author: Nino Krvavica
"""

import numpy as np
import analytic_eig


def ifcp(A, Lam):
    ''' IFCP scheme '''
    def parameters(Lam):
        # Compute parameters for IFCP scheme
        Lam1, Lam2, Lam3, Lam4 = Lam.T
        S_ext = np.sign(Lam2 + Lam3)
        S_ext[S_ext == 0] = 1
        Chi_int = S_ext * np.max([np.abs(Lam2), np.abs(Lam3)], axis=0)
        Lam14 = Lam1 - Lam4
        Lam1_Chi = Lam1 - Chi_int
        Lam4_Chi = Lam4 - Chi_int
        delta1 = np.abs(Lam1) / (Lam14 * Lam1_Chi)
        delta4 = -np.abs(Lam4) / (Lam14 * Lam4_Chi)
        delta_int = np.abs(Chi_int) / (Lam1_Chi * Lam4_Chi)
        alpha0 = (delta1 * Lam4 * Chi_int + delta4 * Lam1 * Chi_int
                  + delta_int * Lam1 * Lam4)
        alpha1 = (-Lam1 * (delta4 + delta_int) - Lam4 * (delta1 + delta_int)
                  - Chi_int * (delta1 + delta4))
        alpha2 = delta1 + delta4 + delta_int
        return alpha0, alpha1, alpha2

    def vis_matrix(alpha0, alpha1, alpha2, A, Id):
        # compute viscosity matrix
        # Asq = A @ A
        # Q = (alpha0[:, None, None] * Id[np.newaxis, :]
        #      + alpha1[:, None, None] * A
        #      + alpha2[:, None, None] * Asq)
        A_inv = np.linalg.solve(A, Id[np.newaxis, :])
        C = (alpha0[:, None, None] * A_inv
             + alpha1[:, None, None] * Id[np.newaxis, :]
             + alpha2[:, None, None] * A)
        Pp = 0.5 * (Id + C)  # positive part of the projection matrix
        Pm = 0.5 * (Id - C)  # negative part of the projection matrix
        return C, Pp, Pm

    alpha0, alpha1, alpha2 = parameters(Lam)
    Id = np.eye(4, 4)
    C, Pp, Pm = vis_matrix(alpha0, alpha1, alpha2, A, Id)

    return C, Pp, Pm


def pvm2(A, Lam):
    ''' PVM-2U scheme '''
    def parameters(Lam):
        SM = np.max(Lam, axis=1)
        Sm = np.min(Lam, axis=1)
        second_max = np.argwhere(np.abs(SM) < np.abs(Sm))
        if second_max.size > 0:
            SM[second_max], Sm[second_max] = Sm[second_max], SM[second_max]
        sgnSM = np.sign(SM)
        sgnSm = np.sign(Sm)
        Sm_SMsq = (Sm - SM)**2
        alpha2 = Sm * (sgnSm - sgnSM) / Sm_SMsq
        alpha1 = (SM * (np.abs(SM) - np.abs(Sm))
                  + Sm * (Sm * sgnSM - SM * sgnSm)) / Sm_SMsq
        alpha0 = SM**2 * alpha2
        return alpha0, alpha1, alpha2

    def vis_matrix(alpha0, alpha1, alpha2, A, Id):
        # compute viscosity matrix
        # Asq = A @ A
        # Q = (alpha0[:, None, None] * Id[np.newaxis, :]
        #      + alpha1[:, None, None] * A
        #      + alpha2[:, None, None] * Asq)
        A_inv = np.linalg.solve(A, Id[np.newaxis, :])
        C = (alpha0[:, None, None] * A_inv
             + alpha1[:, None, None] * Id[np.newaxis, :]
             + alpha2[:, None, None] * A)
        Pp = 0.5 * (Id + C)  # positive part of the projection matrix
        Pm = 0.5 * (Id - C)  # negative part of the projection matrix
        return C, Pp, Pm

    alpha0, alpha1, alpha2 = parameters(Lam)
    Id = np.eye(4, 4)
    C, Pp, Pm = vis_matrix(alpha0, alpha1, alpha2, A, Id)

    return C, Pp, Pm


def pvm4(A, Lam):
    ''' PVM-4 scheme '''
    def parameters(Lam):
        # Compute parameters for IFCP scheme
        SM = np.max(np.abs(Lam), axis=1)
        SI = np.partition(np.abs(Lam), 2)[:, -2]
        SI_SMsq = (SI + SM)**2
        alpha0 = 0.5 * SM * SI * (SI + 2*SM) / SI_SMsq
        alpha1 = 0.5 / SM + SM / SI_SMsq
        alpha2 = -0.5 / (SM * SI_SMsq)
        return alpha0, alpha1, alpha2

    def vis_matrix(alpha0, alpha1, alpha2, A, Id):
        # compute viscosity matrix
        Asq = A @ A
        Acub = Asq @ A
        # Aquar = Asq @ Asq
        # Q = (alpha0[:, None, None] * Id[np.newaxis, :]
        #      + alpha1[:, None, None] * Asq
        #      + alpha2[:, None, None] * Aquar)
        A_inv = np.linalg.solve(A, Id[np.newaxis, :])
        C = (alpha0[:, None, None] * A_inv
             + alpha1[:, None, None] * A
             + alpha2[:, None, None] * Acub)
        Pp = 0.5 * (Id + C)  # positive part of the projection matrix
        Pm = 0.5 * (Id - C)  # negative part of the projection matrix
        return C, Pp, Pm

    alpha0, alpha1, alpha2 = parameters(Lam)
    Id = np.eye(4, 4)
    C, Pp, Pm = vis_matrix(alpha0, alpha1, alpha2, A, Id)

    return C, Pp, Pm


def pvm_roe(A, Lam):
    ''' PVM Roe scheme '''
    def parameters(Lam):
        Lam1, Lam2, Lam3, Lam4 = Lam.T
        N = Lam.shape[0]
        a_lam = np.zeros((N, 4, 4))
        ones = np.ones(N)
        a_lam[:, 0, :] = np.array([ones, Lam1, Lam1**2, Lam1**3]).T
        a_lam[:, 1, :] = np.array([ones, Lam2, Lam2**2, Lam2**3]).T
        a_lam[:, 2, :] = np.array([ones, Lam3, Lam3**2, Lam3**3]).T
        a_lam[:, 3, :] = np.array([ones, Lam4, Lam4**2, Lam4**3]).T
        b_lam = np.zeros((N, 4))
        b_lam[:, 0] = np.abs(Lam1)
        b_lam[:, 1] = np.abs(Lam2)
        b_lam[:, 2] = np.abs(Lam3)
        b_lam[:, 3] = np.abs(Lam4)
        alphas = np.linalg.solve(a_lam, b_lam)
        a0 = alphas[:, 0]
        a1 = alphas[:, 1]
        a2 = alphas[:, 2]
        a3 = alphas[:, 3]
        return a0, a1, a2, a3

    def vis_matrix(a0, a1, a2, a3, A, Id):
        # compute viscosity matrix
        Asq = A @ A
        A_inv = np.linalg.solve(A, Id[np.newaxis, :])
        C = (a0[:, None, None] * A_inv
             + a1[:, None, None] * Id
             + a2[:, None, None] * A
             + a3[:, None, None] * Asq)
        Pp = 0.5 * (Id + C)  # positive part of the projection matrix
        Pm = 0.5 * (Id - C)  # negative part of the projection matrix
        return C, Pp, Pm

    a0, a1, a2, a3 = parameters(Lam)
    Id = np.eye(4, 4)
    C, Pp, Pm = vis_matrix(a0, a1, a2, a3, A, Id)

    return C, Pp, Pm


def comp_Q(u1, u2, c1sq, c2sq, r, g, A, scheme='ifcp',
           eig_type='approximated', hyp_corr=True):
    ''' Returns the numerical viscosity matrix of a two-layer shallow water
    system. Uses several PVM schemes, in particular IFCP, PVM-2U, PVM-4,
	and PVM-Roe.

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
    g: float or ndarray
        acceleration of gravity
    A: ndarray
        stacked flux Jacobian matrices of the two-layer shallow water system
    eig_type: string, optional
        type of eigenvalues ('analytical', 'approximated')
    hyp_corr: bool, optional
        if set to `True` hyperbolicity correction is performed

    Returns
    -------
    Q: ndarray
        stacked 4x4 viscosity matrices
    P_plus: ndarray
        stacked 4x4 projection matrices of positive sign elements
    P_minus: ndarray
        stacked 4x4 projection matrices of negative sign elements
    Lam: ndarray
        stacked eigenvalue arrays
    F: ndarray
        array of correction friction (0 for hyperbolic system)
        '''

    # Approximate eigenvalues
    if eig_type == 'analytical':
        Lam, K, F = analytic_eig.analytic_eig(u1, u2, c1sq, c2sq, r, g,
                                              eigvecs=False, hyp_corr=hyp_corr)
    elif eig_type == 'approximated':
        Lam, K, F = analytic_eig.approx_eig(u1, u2, c1sq, c2sq, r, g,
                                            eigvecs=False, hyp_corr=hyp_corr)
    else:
        raise ValueError('''wrong type of calculation, expected either '''
                         ''''analytical', or 'approximated' ''')

    if scheme == 'ifcp':
        C, Pp, Pm = ifcp(A, Lam)
    elif scheme == 'pvm2':
        C, Pp, Pm = pvm2(A, Lam)
    elif scheme == 'pvm4':
        C, Pp, Pm = pvm4(A, Lam)
    elif scheme == 'pvm_roe':
        C, Pp, Pm = pvm_roe(A, Lam)
    else:
        raise ValueError('''wrong scheme, expected either '''
                         ''''ifcp', 'pvm2', 'pvm4' or 'pvm_roe' ''')

    return C, Pp, Pm, Lam, F
