"""perturb_transmon.py - library of transmon perturbative solutions based on Didier...Rigetti (2018)"""

import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
import math
import sys
from timer import Timer
from qtools import pad_ket
from qtools import truncate_ket
import io_tools as io

# # # # # # # # # # # # # # # # # # # #
# Initialization
DEBUG = False
time = Timer()
# # # # # # # # # # # # # # # # # # # #


# # # # # # # # # # # # # # # # # # # #

# GLOBAL PARAMS debug
W0 = 1
WR = 1.2
XI = .1
G = .2
# # # # # # # # # # # # # # # # # # # #

def get_phieff(phiext, ej1, ej2):
    # didier eqn 17
    return np.arctan2(np.sin(phiext), np.cos(phiext)+ej2/ej1)

def get_ejeff(phiext, ej1, ej2):
    # didier eqn 16
    return np.sqrt(ej1**2 + ej2**2 + 2*ej1*ej2*np.cos(phiext))





def get_En(n, pmax, w0, N, xi):
    """
    do a straightforward sum over pmax orders of xi; eqn (9) in Didier(2018)
    :param n: 
    :param pmax: 
    :param w0: 
    :param N: 
    :param xi: 
    :return: 
    """

    out = 0
    for p in range(pmax+1):
        if DEBUG:
            print("\n calculating E_%i(%i)" % (n, p))
        out += xi**p*En_p(n,p,w0,N)
    return out

def En_p(n, p, w0, N):
    """
    Get the transmon energy to p-th order in xi; eqn 10 in Didier (2018)

    :param n: Hamiltonian energy level
    :param p: Order of energy calculation
    :param w0:
    :param N:
    :return:
    """

    filename = io.get_dumpname_En(n, p, w0, N)
    load_check = io.load_obj(filename, io.tempdir)
    if load_check is not None:
        # must decode the json str formatting I imposed in io_tools.py
        return complex(load_check)

    # 0th order energy is Harmonic Oscillator solution
    if p == 0:
        return n*w0

    # other orders are found by perturbation theory
    nket = qt.fock(N, n)
    out = 0
    # reproduce eqn 10 exactly:
    for q in range(p):

        psi_raw = psi_n_p(n, q, w0, N)
        # if psi_n Fock space size is too small, pad it
        if q <= p-q:
            psi_n = pad_ket(psi_raw, N + 4*(p - q))
            H = H_u(p-q, w0, N+4*(p-q))
            nbra = qt.fock(N+4*(p-q), n).dag()

        # if psi_n is larger than the Hamiltonian, H and |n> must be computed to that size
        elif q > p-q:
            psi_n = psi_raw
            H = H_u(p-q, w0, N+4*q)
            nbra = qt.fock(N+4*q, n).dag()

        if DEBUG:
            print("n=%i, q=%i, p=%i"%(n, q,p))
            #print(H)
            print("Nbra=",nbra)
            #
            # print(psi_n_p(n, q, w0, N), N+4*(p-q))
            print("PSI=",psi_n.dag())
        out += nbra.overlap(H*psi_n)

    # dump to tempdir for future use
    io.dump_obj(io.complex2str(out), filename, io.tempdir)
    return out

def get_psi_n(n, umax, w0, N, xi):
    # Calculate the full perturbed eigenstate to umax-order, with full length N
    # WARNING: DOES NOT TRUNCATE! returns vector length n+4*umax
    out = 0
    for u in range(umax):
        out += xi**u*pad_ket(psi_n_p(n, u, w0, N), N+4*umax)

    return out.unit()

def psi_n_p(n, p, w0, N):
    """
    Generate the eigenstate for transmon hamiltonian to p-th order in xi. This is done
    recursively, and will use calls to En_p, which calls p_sin_p in turn. eqn 11 in Didier (2018)
    :param n: n refers to the energy level of the system
    :param p: order of expansion for eigenstate, i.e. p-th order in xi
    :param w0:
    :param N: Size of the ORIGINAL Hspace; this will be modified according to the order p of calculation
    :return: p-th order nth eigenstate PSI, with SIZE = N + 4*p
    """

    filename = io.get_dumpname_Psi(n, p, w0, N)
    load_check = io.load_obj(filename, io.tempdir)
    if load_check is not None:
        if DEBUG:
            print("loading psi from file %i" % filename)
        return load_check

    # adjusted size of Fock state Hspace to account for larger H_u
    NN = N + 4*p
    nket = qt.fock(NN, n)

    # the zeroth order component of the psi_n is just |n>, in our expanded Hspace
    if p == 0:
        return qt.fock(N, n)

    out = 0
    # FIXME: add "+1" to the top limit of the range?
    for m in range(N+4*p):

        # always ignore m=n because n-th state is fully described by 0th order term
        if m == n:
            continue
        # first term is the pth order hamiltonian on 0th order eigenstates
        first = qt.fock(N+4*p, m).dag()*H_u(p,w0,N+4*p)*qt.fock(N+4*p, n)*qt.fock(N+4*p, m)
        out += first/( w0*(n-m))

    # remaining terms: varying in size depending on the order of H_u, and so must be padded to N+4*p
    # the orders of sum over q and sum over m are switched because FIXME!!!
    for q in range(1, p):
        for m in range(N+4*q):
            if m == n:
                continue
            psi_n_q = psi_n_p(n,q,w0,N)
            if DEBUG:
                print("psi_%i(%i) calling get_E_%i(%i)" % (n,p,n,p) )
            qth = qt.fock(N+4*q, m).dag()*(H_u(p-q, w0, N+4*q)- En_p(n, p-q, w0, N+4*q))*psi_n_q*qt.fock(N+4*p, m)
            out += qth/( w0*(n-m))

    io.dump_obj(out, filename, io.tempdir)

    return out.unit()


def H_u(u, w0, N):
    """
    Get the transmon hamiltonian perturbative expansion to u-th order in xi
    :param u:
    :param w0 = sqrt(8*EJ*EC)
    :param N: size of operator hilbert space
    :return: H_u of Hspace SIZE=N
    """


    filename = io.get_dumpname_H(u, w0, N)
    load_check = io.load_obj(filename, io.tempdir)
    if load_check is not None:
        if DEBUG:
            print("loading psi from file %i" % filename)
        return load_check

    # the hamiltonian will include raising operators like a^(2u + 2), increasing the Hspace size N by 4
    # per order in u
    a = qt.destroy(N=N)
    # a normal-ordered perturbative expansion of a regular transmon H
    out = 0

    # 0th order energy of the harmonic oscillator
    if u == 0:
        return w0*a.dag()*a
    # u >=0 order energy of an expanded harmonic oscillator; eqn 8 in Didier (2018)
    else:
        for v in range(u+1):
            # N5-121: I confirmed that their bounds in eqn (8) are correct.
            k1 = (-1)**u/(2**(u-v+1)*math.factorial(u-v))
            for w in range(-(v+1),v+2):
                # Now this is an INCLUSIVE bounded sum; k1, k2 represent raising/lowering operators and their prefactors
                k2 = a.dag()**(v+1+w)/math.factorial(v+1+w)
                k3 = a**(v+1-w)/math.factorial(v+1-w)
                # print("u=%i v=%i w=%i"%(u,v,w))
                # print(k2)
                # print(k3)
                out += w0*k1*k2*k3

    io.dump_obj(out, filename, io.tempdir)
    return out




def get_H_co(umax, w0, g, wr, N_qb, N_res, xi):
    """
    Get the coupled resonator-transmon hamiltonian perturbative expansion to u-th order in xi. This is the JCM model
    Hamiltonian for the u-th order transmon states in the transmon basis, as in equation 3.2 in Koch (2007)
    :param umax: Highest order of perturbation
    :param wr: sqrt(8*EJ*EC)
    :param N_qb: size of transmon operator hilbert space
    :param N_res: size of resonator operator H space
    :param g: Coupling constant for the transmon-resonator system
    :param xi: sqrt(2EC/EJ)
    :return
    """
    #FIXME how do i determine g? Which "g" is it anyway?

    # resonator H.O. hamiltonian
    b = qt.destroy(N=N_res)
    H_res = qt.tensor(qt.qeye(N_qb), wr*b.dag()*b)

    # transmon term
    H_tr = 0

    # coupling term
    H_co = 0

    #FIXME : WHEN DO I PROPERLY TRUNCATE??? currently: Get the coupling matrix values using full space, then truncate
    # FIXME once you're constructing the raising/lowering operators
    for i in range(N_qb):
        ket_i = get_psi_n(i, umax, w0, N_qb, xi)

        Ei = get_En(i, umax, w0, N_qb, xi)
        H_tr += Ei * qt.tensor(truncate_ket(ket_i, N_qb), truncate_ket(ket_i, N_qb).dag(), qt.qeye(N_res))

        for j in range(N_qb):
            ket_j = get_psi_n(j, umax, w0, N_qb, xi)

            # find the coupling terms by calculating <j | n_tr | i>
            # psi_n_p has size N_qb + 4*u !!!
            a = qt.destroy(N=N_qb+4*umax)
            n_tr = 1j / (2 * xi) * (a.dag() - a)
            g_ij = g*ket_i.dag()*n_tr*ket_j

            # qubit coupling term looks like raising/lowering operator
            if i != j:
                print(ket_i.dims)
                print(qt.fock(N_qb,0).dims)
                print(truncate_ket(ket_i, N_qb).dims)
                print(qt.tensor(truncate_ket(ket_i, N_qb), truncate_ket(ket_j, N_qb).dag()).dims)
                H_co += g_ij*qt.tensor(truncate_ket(ket_i, N_qb), truncate_ket(ket_j, N_qb).dag(), b+b.dag())
                print(H_co.dims)
            break

    print(H_tr)
    print(H_res)
    print(H_co)

    return H_tr + H_res + H_co






if __name__ == "__main__":
    H_co_u(3, W0, G, WR, 3, 4, XI)
