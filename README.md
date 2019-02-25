# Q-2L-SWE
Viscosity matrix for 2-layer shallow water equations

These repository includes several numerical algorithms for computation of a viscosity matrix `Q` of a two-layer shallow water system, where eigenvalues are computed by three different approaches.
The Roe and four PVM (Polynomial Viscosity Matrix) schemes are considered, in particular IFCP, PVM-2U, PVM-4 and PVM-Roe. Implementation differs in the way that the eigenvalues are computed, either by a numerical eigensolver, approximated expressions or closed-form solutions.

These algorithms are a supplement to a article:

`Krvavica, N. On eigenvalues of two-layer shallow water systems: Re-evaluating efficiency of first-order Roe and PVM schemes`

summited to an international journal.


# Requirements
 
Python 3+, Numpy, Scipy

# Usage

There are two main functions and two subfunctions. The first is `roe.py` which returns the viscosity matrix in a Roe scheme, the second is `pvm.py` which returns the viscosity matrix of a corresponding PVM scheme. Both functions call subfunction `numeric_eig.py` if a numerical eigensolver is chosen, and `analytic_eig.py` if closed-form or approximated solutions are chosen.

There are also four tests included, which can also help with understanding the usage of these functions.
* The first test `test_eig_accuracy.py` analyzes the accuracy of numerical, approximated, and closed-form solutions to eigenvalues of a `4x4` Jacobian matrix.
* The second test `test_eig_cost.py` estimates the computational cost (CPU time) needed for these three algorithms to compute a given number of eigenvalues.
* The test `test_Q_accuracy.py` examines the accuracy of different numerical schemes (Roe, IFCP, PVM-2U, PVM4, PVM-Roe) in computing the viscosity matrix, depending on the choice for the eigenvalues solver.
* Finally, test `test_Q_cost.py` estimates the computational cost (CPU time) needed for these different schemes to compute the viscosity matrix.


# License
 
[MIT license](LICENSE.txt)