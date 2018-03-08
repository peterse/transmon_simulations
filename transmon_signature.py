"""transmon_signature.py - a script to reproduce transmon-resonator frequency crossing for a set of known params"""

import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
import math
import sys


e = 1 # electron charge (C)
Phi0 = 1 # flux quantum

# GLOBAL QUBIT PARAMS
I_c = 1
C_tot = 1


# derived parameters
# ejmax = Phi0*I_c
# E_C = (2*e)**2/C_tot
EJMAX = 10
EC = 1
NG = .5

def EJ(flux, ejmax):
    # calculate E_J using specifically local variables
    return ejmax*abs(np.cos(np.pi*flux/Phi0))


# Some operators
def COS(k,A, n=8):
    # expand cosine of an operator*const cos(kA) = 1 + (ikA)^2/2 - (ikA)^4/4! + ...
    # n is the order of expansion i.e. highest power
    out = 0
    for i in range(n+1):
        if i%2:
            continue
        else:
            out += A**i *(1j*k)**i /math.factorial(i)
    return out

def n_phi_tr(flux, ejmax, ec, N=3):
    """
     number operator and phase operator for EJ>>Ec in transmon HO approximation
     :param flux: external flux bias, affecting EJ
     :param N: number of basis states in transmon subspace to consider
     :return: [n_phi, n_tr]: both operators
    """

    # Koch, 2007 eqn C4
    b = qt.destroy(N=N)

    n_tr = -(1j)*(b - b.dag())*(EJ(flux, ejmax)/(8*ec))**.25/np.sqrt(2)
    phi_tr = ( b + b.dag() )*(8*ec/EJ(flux, ejmax))**.25/np.sqrt(2)

    return (phi_tr, n_tr)


def H_approx(flux, ng, ejmax, ec, N=6):
    """
    Calculate the qubit hamiltonian using external flux, system energies using
    Koch eqn C1
    :param  flux: external flux bias, affecting EJ
            N: number of Hilbert space dimensions on uncoupled qubit
    :return: qubit subspace hamiltonian
    """

    b = qt.destroy(N=N)
    # Koch (2007) eqn C1 for EJ >> EC
    # valid only for flux very close to 0 +/- n*2pi
    H = np.sqrt(8*ec*EJ(flux, ejmax))*(b.dag()*b + .5) - EJ(flux, ejmax) - ec/12*(b + b.dag())**4

    return H


def H_CPB(flux, ng, ejmax, ec, N=6):
    """
    Approximate CPB hamiltonian for EJ>>EC using HO operators and unshifted frame;
    Koch eqn 2.1 with EJ>>EC applied to cosine expansion
    :param flux:
    :param ng: gate charge number
    :param N:
    :return:
    """
    (phi_tr, n_tr) = n_phi_tr(flux, ejmax, ec, N=N)
    return 4*ec*(n_tr + ng)**2 - EJ(flux, ejmax)*COS(1,phi_tr,n=4)


def E_m_approx(flux, ejmax, ec, m):
    # apply Koch eqn 2.11 for the approximate eigenenergy as a fxn of flux
    return -EJ(flux, ejmax) + np.sqrt(8*ec*EJ(flux, ejmax))*(m+.5) - ec/12*(6*m**2 + 6*m + 3)


OPTION = 2

if OPTION == 1:
    # PLOT EIGENENERGIES VS. EXTERNAL FLUX
    fluxes = np.arange(-np.pi, np.pi, .01)
    # energies = [H_approx(f, N=4).eigenstates()[0] for f in fluxes]
    # plt.plot(fluxes, energies)
    # for m in range(4):
    #     plt.plot(fluxes, [E_m_approx(flux, EJMAX, EC, m) for flux in fluxes], "--")
    # plt.show()


elif OPTION == 2:
    # PLOT QUBIT FREQUENCY VS EXTERNAL FLUX
    # VERIFY THAT H_CPB VS. FLUX PRODUCES THE SAME RESULTS AS EQN C1: NG-SHIFTED H
    fluxes = np.arange(-np.pi, np.pi, .01)

    # E01 using H_approx with a "gauge transformation" to remove ng
    E01_array = []
    for f in fluxes:
        energies = H_approx(f, NG, EJMAX, EC, N=4).eigenstates()[0]
        E01 = energies[1] - energies[0]
        E01_array.append(E01)

    # E01 approximation that includes anharmonic terms - Koch 2.11
    E01_approx1 = [E_m_approx(flux, EJMAX , EC, 1) - E_m_approx(flux, EJMAX, EC, 0) for flux in fluxes]

    # E01 approximation with no anharmonicity - E01 = sqrt(8EjEc)
    E01_approx2 = [np.sqrt(8*EC*EJ(flux, EJMAX)) for flux in fluxes]

    # E01 using H_CPB with Ej>>Ec instead of ng-shifted hamiltonian
    fig = plt.figure()

    for p, ng_local in enumerate([0,.5,1]):
        E01_array2 = []
        for f in fluxes:
            energies = H_CPB(f, ng_local, EJMAX, EC, N=4).eigenstates()[0]
            E01 = energies[1] - energies[0]
            E01_array2.append(E01)

        ax = fig.add_subplot(3,1,p+1)
        ax.plot(fluxes, E01_array, fluxes, E01_approx1, "--" , fluxes, E01_approx2, fluxes, E01_array2)
        ax.legend(["solved H_approx", "approx_1", "approx2", "solved H_CPB"])
        ax.set_xlabel("flux/phi0")
        ax.set_ylabel("E01")

    plt.show()

    # plot against NG
    ng_vals = np.arange(0,4,.5)
    #ng_vals = [0, .51, 1]

    # # plot E12 - excited qubit state
    # E02_approx = [E_m_approx(flux, EJMAX , EC, 2) - E_m_approx(flux, EJMAX, EC, 1) for flux in fluxes]
    # plt.plot(fluxes, E02_approx, fluxes, E01_approx1)
    # plt.legend(["E02", "E01"])
    # plt.xlabel("flux/phi0")
    # plt.ylabel("E01")
    # plt.show()

elif OPTION == 3:
    # PLOT EIGENENERGIES VS GATE CHARGE
    # FIXME: this doesn't reproduce Koch fig 2


    gate_charges = np.arange(-2, 2, .05)
    fig = plt.figure()
    ratios = [50]
    # iteratoe over different EJ/EC; overwrite global ejmax, ec
    for p, ejmax in enumerate(ratios):
        ec=1
        energies = []
        for k, ng in enumerate(gate_charges):
            energy = H_CPB(0, ng, ejmax, ec, N=4).eigenstates()[0]
            E01 = energy[1] - energy[0]
            energies.append(energy/E01)
        ax = fig.add_subplot(len(ratios),1,p+1)
        ax.plot(gate_charges, energies)

    plt.show()
elif OPTION == 4:
    # INTRODUCE COUPLING TERMS WITH RESONATOR
    pass