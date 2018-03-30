import perturb_transmon as transmon
import numpy as np
from timer import Timer
import matplotlib.pyplot as plt




def test_transmon_frequency(ej, ec):
    # the paper gives 5th order frequency and anharmonicity

    w0 = np.sqrt(8*ej*ec)
    xi = np.sqrt(2*ec/ej)
    N = 3 # 3-level system
    order_max = 5

    # 5th order frequency and anharmonicity
    freq = w0 - ec*(1 + xi/2**2 + 21*xi**2/2**7 + 19*xi**3/2**7 + 5319*xi**4/2**15)
    anharm = ec*(1 + 9*xi/2**4 + 81*xi**2/2**7 + 3645*xi**3/2**12 + 46899*xi**4/2**15)


    n0lst = []
    n1lst = []
    n2lst = []
    E_p_lst = [ None for i in range(order_max+1)]
    anharm_p_lst = [ None for i in range(order_max+1)]
    for order in range(order_max+1):

        print("\n\nCALCULATE E_0(%i)" % order)
        time.start("E0")
        E0 = transmon.get_En(0, order, w0, N, xi)
        t1 = time.end()
        n0lst.append( (order, t1))

        print("\n\nCALCULATE E_1(%i)" % order)
        time.start("E1")
        E1 = transmon.get_En(1, order, w0, N, xi)
        t1 = time.end()
        n1lst.append( (order, t1))

        print("\n\nCALCULATE E_2(%i)" % order)
        time.start("E2")
        E2 = transmon.get_En(2, order, w0, N, xi)
        t1 = time.end()
        n2lst.append( (order, t1))

        # p-th order frequency and anharmonicity
        E10 = E1 - E0
        E21 = E2 - E1
        eta = E10 - E21
        E_p_lst[order] = E10
        anharm_p_lst[order] = eta

        print("Analytical freq =", freq, "  %ith order freq=" % order, E10, "Error=", (freq-E10)/freq)
        print("analytical anharm =", anharm, "  %ith oder anharm =" % order, eta, "Error=", (anharm-eta)/anharm)

    # Plot time consumption for each n-level, vs order of calculation
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot([k[0] for k in n0lst], [k[1] for k in n0lst])
    ax1.plot([k[0] for k in n1lst], [k[1] for k in n1lst])
    ax1.plot([k[0] for k in n2lst], [k[1] for k in n2lst])
    ax1.set_xlabel("p")
    ax1.set_ylabel("time")
    ax1.legend(["n=0", "n=1", "n=2"])
    plt.show()

    # plot convergence of energies as p increases
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.plot(E_p_lst)
    ax1.axhline(freq,  ls="--", color="red")
    ax1.set_ylabel("E01")

    ax2 = fig.add_subplot(212)
    ax2.plot(anharm_p_lst)
    ax2.axhline(anharm, ls="--", color="red")
    ax2.set_ylabel("anharmonicity")

    plt.show()

    return

def test_transmon_tuning(ej1, ej2, ec):

    N = 3 # 3-level system
    order = 4

    phi_arr = np.arange(0,2*np.pi, np.pi/10)
    E01_arr = [0 for k in phi_arr]
    # ej, w0, xi must be calculated within the loop since they all depend on phi ext now
    for i, phiext in enumerate(phi_arr):
        ejeff = transmon.get_ejeff(phiext, ej1, ej2)
        phieff = transmon.get_phieff(phiext, ej1, ej2)
        w0 = np.sqrt(8 * ejeff * ec)
        xi = np.sqrt(2 * ec / ejeff)


        print("\n\nCALCULATE E_0(%i) at phi=%4.3f" % (order, phiext) )
        time.start("E0")
        E0 = transmon.get_En(0, order, w0, N, xi)
        t1 = time.end()

        print("\n\nCALCULATE E_1(%i) at phi=%4.3f" % (order, phiext) )
        time.start("E1")
        E1 = transmon.get_En(1, order, w0, N, xi)
        t2 = time.end()

        E01_arr[i] = E1-E0



    # plot qubit frequency vs. external flux
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(phi_arr, E01_arr)
    ax1.set_ylabel("E01")
    ax1.set_xlabel("phi_ext")
    plt.show()

def test_transmon_coupling(ej1, ej2, ec, g, wr):


    w0 = np.sqrt(8*ej1*ec)
    xi = np.sqrt(2*ec/ej1)

    Nqb = 2 # 3-level system
    Nres = 4 # resonator levels
    order = 2

    H_co = transmon.get_H_co(order, w0, g, wr, Nqb, Nres, xi)

time = Timer()
#test_transmon_frequency(10,1)
#test_transmon_tuning(10,10, 1)
test_transmon_coupling(10,10, 1, .2, 9)