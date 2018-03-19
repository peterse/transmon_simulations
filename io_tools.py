"""io_tools.py - interacting with shell and file IO
Evan (2018)
"""

import sys
import os
import json
import numpy as np
import qutip as qt
import shutil


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#PROJECT GLOBAL VARS
tempdir = "C:\\Users\\e6peters\\Desktop\\qubit_simulations\\temp"


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

######################################################################
#Sub-Tree splitting:

def path_exists(path):
    #check that the directory 'path' exists
    try:
        with cd(path) as d:
            pass
    except:
        #FIXME: what's the error I'm looking for?
        print(sys.exc_info()[0])
        return False
        sys.exit()


    return True

######################################################################lp
class cd:
    """Context management for directory changes"""
    def __init__(self, new_path):
        self.new_path = os.path.expanduser(new_path)

    def __enter__(self):
        #Change to the argument directory
        self.saved_path = os.getcwd()
        os.chdir(self.new_path)

    def __exit__(self, typ, val, traceback):
        #Go back to the original directory
        os.chdir(self.saved_path)


######################################################################
#PYTHON EXTERNAL OBJECT MANAGEMENT
# update (2018) - converting to JSON file management because apparently its way faster

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# object encoding and serialization
# TODO: apparently qutip.fileio module has better solutions to this; for now I do not need higher functionality
def complex2str(val):
    # convert a complex number into string format
    return "({0.real:.5}+{0.imag:.5}j)".format(val)

def encode_ket(phi):
    # FIXME: Qobj's dont support json serialization merp. Hack it myself by 'listify'
    # return a list-form of the qobj
    if phi.shape[1] != 1:
        raise TypeError("pad_ket cannot accept bra state")
    out = []
    for v in phi:
        out.append(complex2str(v[0][0]) )
    return out

def decode_ket(phi_lst):
    # convert a ket list back into a qobj
    out = [complex(v) for v in phi_lst]
    return qt.Qobj(np.array(out))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# external file management

def get_dumpname_En(n, p, w0, N):
    # generate a filename for dumping En loads
    return "EN_%i_%i_%5.4f_%i.json" % (n, p, w0, N)

def get_dumpname_Psi(n, p, w0, N):
    # generate a filename for dumping En loads
    return "Psi_%i_%i_%5.4f_%i.qt" % (n, p, w0, N)

def get_dumpname_H(u, w0, N):
    # generate a filename for u-th order hamiltonian
    return "H_%i_%5.4f_%i.qt" % (u, w0, N)

def dump_obj(obj, filename, path):
    #serialize a python object to a file
    #A file takes a single object.
    with cd(path):
        # loading Qobjs
        if type(obj) is qt.qobj.Qobj:
            qt.file_data_store(filename, obj)
        # Loading anything else (probably string)
        else:
    return filename

def load_obj(filename, path):
    # grab the sole pickled python object from path directory
    # WARNING - returns None if the file does not exist! Use with caution
    # default return for no object found: "None"
    with cd(path):
        # Loading qobjs
        if ".qt" in filename:
            try:
                obj = qt.file_data_read(filename, sep=",")

            except FileNotFoundError:
                return None

            return qt.Qobj(obj)

        # in the case of anything NOT qobj, I would like not to use their unpacking method
        else:
            try:
                obj = json.load(open(filename))
                return obj

            except IOError:
                return None



#
# def clean_pickle(source, target):
#     with cd(source):
#         pickles = [f for f in os.listdir(os.getcwd()) if ".pickle" in f]
#         for fname in pickles:
#             shutil.move(fname, target)