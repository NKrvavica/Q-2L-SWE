# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 09:53:19 2019

@author: Nino Krvavica
"""

import numpy as np
import warnings


def analytic_eigenvectors(u1, c1sq, g, Lam, N):
    '''Computes analytical eigevector matrix from eigenvalues'''
    Id = np.ones((N, 4))
    k = 1 - (Lam - u1[:, np.newaxis])**2 / c1sq[:, np.newaxis]
    K = np.zeros((N, 4, 4))
    K[:, 0, :] = Id
    K[:, 1, :] = Lam
    K[:, 2, :] = -k
    K[:, 3, :] = -k * Lam
    return K


def analytic_eig(u1, u2, c1sq, c2sq, r, g,
                 eigvecs=True, hyp_corr=True):
    ''' Computes the analytical closed-form solution to the eigenvalues `Lam`
    and eigenvectors `K` of matrix `A`::

        A = [[0, 1, 0, 0],
             [c1**2-u1**2, 2*u1, c1**2, 0],
             [0, 0, 0, 1],
             [r*c2**2, 0, c2**2 - u2**2, 2*u2]]

    If complex eigenvalues are predicted, it corrects the velocities by
    an optimal friction factor `F` so that the hyperbolic condition is
    restored.

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
    eigvecs: bool, optional
        if set to `True` the function computes and returns eigenvalues and
        eigenvectors, otherwise it returns only eigenvalues
    hyp_corr: bool, optional
        is set to `True` the function performs the hyperbolicity correction
		where needed.

    Returns
    -------
    K: ndarray or empty
        stacked 4x4 matrices whose columns are right eigenvectors (empty if
        eigevces is set to `False`)
    Lam: ndarray
        stacked eigenvalue arrays
    F: ndarray
        array of correction friction (0 for hyperbolic system)
    '''

    def corr_u(u1, u2, c1sq, c2sq, r, g, F):
        ''' Hyperbolicity correction.
        Modifies velocities `u1` and `u2` by the correction (friction)
        factor F'''
        sign_du = np.sign(u2 - u1)
        u1 += F * sign_du / c1sq * g
        u2 -= r * F * sign_du / c2sq * g
        return u1, u2

    def char_poly_coeff(u1, u2, c1sq, c2sq, r, g):
        ''' Computes coefficients of a characteristic polynomial `p(x)`:
        p(x) = x^4 + a*x^3 + b*x^2 + c*x + d'''
        u_c_1 = u1*u1 - c1sq
        u_c_2 = u2*u2 - c2sq
        a = -2*(u1 + u2)
        b = u_c_1 + 4*u1*u2 + u_c_2
        c = - 2*u2*u_c_1 - 2*u1*u_c_2
        d = u_c_1*u_c_2 - r*c1sq*c2sq
        return a, b, c, d

    def get_discr(params, F=0):
        '''Computes the coefficients of the characteristic polynomial and
        finds the discriminant of the characteristic polynomial'''
        u1, u2, c1sq, c2sq, r, g = params
        if F != 0:
            u1, u2 = corr_u(*params, F)
        # Coefficients of the characteristic polynomial
        a, b, c, d = char_poly_coeff(u1, u2, c1sq, c2sq, r, g)
        d0 = b*b + 12*d - 3*a*c
        d1 = 27*a*a*d - 9*a*b*c + 2*b*b*b - 72*b*d + 27*c*c
        sqrt_d0 = np.sqrt(d0)
        acos = 0.5 * d1 / (d0 * sqrt_d0)  # term that goes under acos
        check = np.abs(acos) - 1  # condition for hyperboclity loss
        return a, b, c, d, d0, d1, sqrt_d0, acos, check

    def illinois(f, a, b, params, max_iterations=20, tol=1e-5):
        ''''Iterative Illions algorithm (adapted)'''
        *_, f_a = f(params, a)
        *_, f_b = f(params, b)
        for i in range(max_iterations):
            c = b - f_b * (a - b) / (f_a - f_b)
            *_, f_c = f(params, c)
            if f_b*f_c < 0:
                a = c
                f_a = f_c
            else:
                b = c
                f_b = f_c
                f_a = f_a / 2
            if np.abs(a-b) < tol:
                return np.max([a, b])
        else:
            warnings.warn('Not converging...')
        return np.max([a, b])

    def hyper_correction(check, params):
        ''' iterative hyperbolicity correction '''
        u1, u2, c1sq, c2sq, r, g = params
        F = np.zeros((N, 1))
        if (check >= 0).any():
            Fd = (u1 - u2)**2 / ((1-r) * (c1sq + c2sq))
            for idx in np.where(check >= 0)[0]:
                a = 0
                b = (np.sqrt((1-r) * (c1sq[idx] + c2sq[idx]))
                     * (np.sqrt(Fd[idx]) - 1)
                     / (1/c1sq[idx] + r/c2sq[idx]) / g)
                params_idx = (u1[idx], u2[idx], c1sq[idx], c2sq[idx], r, g)
                F[idx] = illinois(get_discr, a, b, params_idx)
                u1[idx], u2[idx] = corr_u(*params_idx, F[idx])
        return (u1, u2, c1sq, c2sq, r, g), F

    def analytic_eigenvalues(u1, u2, c1sq, c2sq, r, g, N):
        '''Computes analytical eigenvalues matrix by finding real roots of
        characteristic polynomial'''
        params = (u1, u2, c1sq, c2sq, r, g)
        (a, b, c, d, d0, d1, sqrt_d0, acos, check) = get_discr(params)
        if hyp_corr:
            params, F = hyper_correction(check, params)
            if (F > 0).any():
                a, b, c, d, d0, d1, sqrt_d0, acos, check= get_discr(params)
        else:
            F = np.zeros((N, 1))
        third = 1./3.
        a2 = a*a
        A = -0.75*a2 + 2*b
        B = 0.25*a2*a - a*b + 2*c
        fi = np.arccos(acos)
        Z_2 = third * (2 * sqrt_d0 * np.cos(third * fi) - A)
        Z = np.sqrt(Z_2)
        B_Z = B / Z
        sqrt_p = np.sqrt(-A - Z_2 + B_Z)
        sqrt_m = np.sqrt(-A - Z_2 - B_Z)
        a025 = -0.25*a
        Lam = np.zeros((N, 4))
        Lam[:, 0] = a025 - 0.5*(Z + sqrt_p)
        Lam[:, 1] = a025 - 0.5*(Z - sqrt_p)
        Lam[:, 2] = a025 + 0.5*(Z - sqrt_m)
        Lam[:, 3] = a025 + 0.5*(Z + sqrt_m)
        return Lam, F

    # get coefficient and discriminant of the characteristic polynomial
    N = u1.size  # Sample size

    # Compute eigenvalues
    Lam, F = analytic_eigenvalues(u1, u2, c1sq, c2sq, r, g, N)

    # Compute eigenvectors
    if eigvecs:
        K = analytic_eigenvectors(u1, c1sq, g, Lam, N)
    else:
        K = []

    return Lam, K, F


def approx_eig(u1, u2, c1sq, c2sq, r, g,
               eigvecs=True, hyp_corr=True):
    ''' Computes the approximated expressions for the eigenvalues `Lam`
    and eigenvectors `K` of matrix `A`::

        A = [[0, 1, 0, 0],
             [c1**2-u1**2, 2*u1, c1**2, 0],
             [0, 0, 0, 1],
             [r*c2**2, 0, c2**2 - u2**2, 2*u2]]

    If complex eigenvalues are predicted, it corrects the velocities by
    an optimal friction factor `F` so that the hyperbolic condition is
    restored.

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
    eigvecs: bool, optional
        if set to `True` the function computes and returns eigenvalues and
        eigenvectors, otherwise it returns only eigenvalues
    hyp_corr: bool, optional
        is set to `True` the function performs the hyperbolicity correction
		where needed.

    Returns
    -------
    K: ndarray or empty
        stacked 4x4 matrices whose columns are right eigenvectors (empty if
        eigevces is set to `False`)
    Lam: ndarray
        stacked eigenvalue arrays
    F: ndarray
        array of correction friction (0 for hyperbolic system)
    '''

    def corr_u(u1, u2, c1sq, c2sq, r, g, F):
        ''' Hyperbolicity correction.
        Modifies velocities `u1` and `u2` by the correction (friction)
        factor F'''
        sign_du = np.sign(u2 - u1)
        u1 += F * sign_du / c1sq * g
        u2 -= r * F * sign_du / c2sq * g
        return u1, u2

    def hyper_correction(Fd, params, eps=1e-6):
        F = np.zeros((N, 1))
        if (Fd >= 1).any():
            idx = np.argwhere(Fd >= 1)
            du = (u1 - u2)[idx]
            gH = ((1-r) * (c1sq + c2sq))[idx]
            delta = np.abs(du) - np.sqrt(gH) + eps
            F[idx, 0] = delta * ((c1sq[idx] * c2sq[idx])
                                 / (g * (r*c1sq[idx] + c2sq[idx])))
            params_idx = (u1[idx], u2[idx], c1sq[idx], c2sq[idx], r, g)
            u1[idx], u2[idx] = corr_u(*params_idx, F[idx, 0])
            Fd[idx] = (u1[idx] - u2[idx])**2 / gH
        return u1, u2, Fd, F

    def approximate_eigenvalues(u1, u2, c1sq, c2sq, r, g, N):
        ''' Computes the approximation of the eigenvalues for the two-layer
        system, based on the assumption that r=1, and u1=u2'''
        Fd = (u1 - u2)**2 / ((1-r) * (c1sq + c2sq))
        if hyp_corr:
            params = (u1, u2, c1sq, c2sq, r, g)
            u1, u2, Fd, F = hyper_correction(Fd, params)
        else:
            F = np.zeros((N, 1))
        c12_2 = 1 / (c1sq + c2sq)
        a = (u1 * c1sq + u2 * c2sq) * c12_2
        b = np.sqrt(c1sq + c2sq)
        c = (u1 * c2sq + u2 * c1sq) * c12_2
        d = np.sqrt((1-r) * c1sq * c2sq * c12_2 * (1 - Fd))
        Lam = np.zeros((N, 4))
        Lam[:, 0] = a - b
        Lam[:, 1] = c - d
        Lam[:, 2] = c + d
        Lam[:, 3] = a + b
        return Lam, F

    # get coefficient and discriminant of the characteristic polynomial
    N = u1.size  # Sample size

    # Compute eigenvalues
    Lam, F = approximate_eigenvalues(u1, u2, c1sq, c2sq, r, g, N)

    # Compute eigenvectors
    if eigvecs:
        K = analytic_eigenvectors(u1, c1sq, g, Lam, N)
    else:
        K = []

    return Lam, K, F

