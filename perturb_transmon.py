"""perturb_transmon.py - library of transmon perturbative solutions based on Didier...Rigetti (2018)"""

import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
import math
import sys
from timer import Timer
from qtools import pad_ket

# # # # # # # # # # # # # # # # # # # #
# Initialization
DEBUG = False
time = Timer()
# # # # # # # # # # # # # # # # # # # #


# # # # # # # # # # # # # # # # # # # #

# GLOBAL PARAMS debug
W0 = 1
XI = .1
# # # # # # # # # # # # # # # # # # # #


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
            H = H = H_u(p-q, w0, N+4*q)
            nbra = qt.fock(N+4*q, n).dag()

        if DEBUG:
            print("n=%i, q=%i, p=%i"%(n, q,p))
            #print(H)
            print("Nbra=",nbra)
            #
            # print(psi_n_p(n, q, w0, N), N+4*(p-q))
            print("PSI=",psi_n.dag())
        out += nbra.overlap(H*psi_n)
    return out


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

    return out


def H_u(u, w0, N):
    """
    Get the transmon hamiltonian perturbative expansion to u-th order in xi
    :param u:
    :param N: size of operator hilbert space
    :return: H_u of Hspace SIZE=N
    """

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

    return out


n = 1
N = 3
for u in range(3):
    val = get_En(n,u,W0,N, XI)
    print("ORDER %i En=" % u, val)
