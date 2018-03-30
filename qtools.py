"""
qtools.py - some unassociated random tools for working with QuTiP
"""

import numpy as np
import qutip as qt
import math

def ketify(ket, M, LaTeX=False):
    """
    Given a state vector of N dimensions and composed of M subspaces, transform
    into an expression of bra, kets
    :param M: number of subspaces to divide state into
    :kwarg LaTeX: If True, return in LaTeX formatting
    :return:
    """

    # check to make sure this is <= 2 dimensions
    if len(ket.shape) != 2:
        raise ValueError("Ket must be vector type")

    N  = max(ket.shape)
    # check to make sure this is not a matrix
    for d in ket.shape:
        if d != N and d != 1:
            raise ValueError("cannot pass matrix to ketify")

    # one subspace means all of the operations are really simple
    if M == 1:
        full_ket = ""
        for j in range(N):
            val = ket.overlap(qt.basis(N, j))
            ket_str = "({0.real:.5}+{0.imag:.5}j)".format(val) + "|" + "%i"%j + ">  "
            full_ket += ket_str
        return full_ket

    else:
        # Determine the base for counting subspace states
        base = N**(1/M)
        # Check if the user is lying about their subspace composition
        if not base.is_integer():
            print("Ket of dimension %i cannot be subdivided into %i-dimensional subspaces"%(N,M))
            return
        base=int(base)

        # simple N-nary labeling scheme based on number of subspaces
        full_ket = ""
        # grab ket values using a dot product (cleanest way of doing it)
        for j in range(N):
            val = ket.overlap(qt.basis(N,j))
            # skip states we don't have
            if abs(val) == 0:
                continue

            # find the base-M representation and pad it
            base_str = np.base_repr(j, base)
            while len(base_str) < M:
                base_str = "0"+base_str
            base_str = ",".join([c for c in base_str])
            ket_str = "({0.real:.5}+{0.imag:.5}j)".format(val) + "|" + base_str + ">  "
            full_ket += ket_str

    return full_ket

def pad_ket(phi, N):
    # passed a FOCK state ket phi, pad it until it has length N OR GREATER

    # shape should be (X, 1)
    if phi.shape[1] != 1:
        raise TypeError("pad_ket cannot accept bra state")

    # if phi is larger than N dims, do nothing.
    M = phi.shape[0]
    if M >= N:
        return phi

    # otherwise, fill the output list with all of phi's values, then N zeros
    out = []
    for v in phi:
        out.append(v[0][0])
    for k in range(N-M):
        out.append(0)
    return qt.Qobj(np.array(out)).unit()



def truncate_ket(phi,N):
    # passed a fock state ket phi, truncate to length N OR LESS

    # shape should be (X, 1)
    if phi.shape[1] != 1:
        raise TypeError("pad_ket cannot accept bra state")

    # extract info on the size. if phi is already shorter than N, no need to truncate
    M = phi.shape[0]
    if M <= N:
        return phi

    # otherwise, fill a list with the first N values of phi and turn that list into a normalized ket
    out = []
    for k in range(N):
        out.append(phi[k][0][0])
    return qt.Qobj(np.array(out)).unit()

def qeye2(N):
    # create an identity out of adding outer products
    out = 0
    for i in range(N):
        v = qt.fock(N,i)
        out+= v*v.dag()
    return out

# TESTING
# ket = qt.Qobj([ [.2], [0], [.4], [.1], [0], [.6], [.4], [.1], [0],])
# print(ketify(ket, 1))
# print(ketify(ket, 2))
# print(ketify(ket,3))