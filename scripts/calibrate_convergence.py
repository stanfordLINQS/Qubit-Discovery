"""In this script we will calibrate the convergence condition for circuits."""

import numpy as np
import matplotlib.pyplot as plt

import SQcircuit as sq

from qubit_discovery.optimization.truncation import (
    check_convergence, test_convergence
)


################################################################################
# Convergence of Pokemon (JJL)
################################################################################

h = 6.6260701e-34
e = 1.60217663e-19
GHz = 1e9

E_C_theta = 0.1 * GHz * h
E_C_phi = 0.09 * GHz * h

C_J_value = e**2 / (4 * E_C_theta)
C_L_value = 0.5 * (e**2/(4 * E_C_phi) - C_J_value)

loop = sq.Loop(0.21)

C_J = sq.Capacitor(C_J_value, "F")
C_L = sq.Capacitor(C_L_value, "F")
JJ = sq.Junction(100, "GHz", loops=[loop])
L = sq.Inductor(0.5, "GHz", loops=[loop])

elements = {
    (0, 1): [C_J, JJ],
    (0, 2): [C_J, JJ],
    (1, 2): [C_L, L],
}

cr = sq.Circuit(elements)
idx = 0

# # [11, 11] not converged for levels with index 1, 2, 4
# cr.set_trunc_nums([11, 11])
# _, _ = cr.diag(n_eig=5)
# print(
#     f"trunc_nums: {cr.m}, "
#     f"check_convergence: {check_convergence(cr, i=idx)}, "
#     f"test_convergence: {test_convergence(cr)}"
# )
#
# # [16, 11] not well converged
# cr.set_trunc_nums([16, 11])
# _, _ = cr.diag(n_eig=5)
# print(
#     f"trunc_nums: {cr.m}, "
#     f"check_convergence: {check_convergence(cr, i=idx)}, "
#     f"test_convergence: {test_convergence(cr)}"
# )
#
# cr.set_trunc_nums([30, 20])
# _, _ = cr.diag(n_eig=5)
# print(
#     f"trunc_nums: {cr.m}, "
#     f"check_convergence: {check_convergence(cr, i=idx)}, "
#     f"test_convergence: {test_convergence(cr)}"
# )
#
# cr.set_trunc_nums([25, 15])
# _, _ = cr.diag(n_eig=5)
# print(
#     f"trunc_nums: {cr.m}, "
#     f"check_convergence: {check_convergence(cr, i=idx)}, "
#     f"test_convergence: {test_convergence(cr)}"
# )
#
#

idx = 0
cr.set_trunc_nums([90, 24])
print(f"trunc_nums: {cr.m}")
_, _ = cr.diag(n_eig=20)
print(
    f"trunc_nums: {cr.m}, "
    f"check_convergence: {check_convergence(cr, i=idx)}, "
    f"test_convergence: {test_convergence(cr)}"
)
print("freq:", cr.efreqs[1] - cr.efreqs[0])
print(60*"-")
# external flux for sweeping over
phi = np.linspace(0.0, 1.0, 100)

# spectrum of the circuit
n_eig = 20
spec = np.zeros((n_eig, len(phi)))

for i in range(len(phi)):
    # set the external flux for the loop
    loop.set_flux(phi[i])

    # diagonalize the circuit
    spec[:, i], _ = cr.diag(n_eig)

plt.figure()
for i in range(n_eig):
    plt.plot(phi, spec[i, :] - spec[0, :])

plt.xlabel(r"$\Phi_{ext}/\Phi_0$", fontsize=13)
plt.ylabel(r"$f_i-f_0$[GHz]", fontsize=13)
plt.show()
