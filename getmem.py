
import numpy as np
from numpy import pi, sin, cos

def GetMem(a1, b1, a2, b2, ndir):
    
    """
    Compute the directional spectrum using the Maximum Entropy 
    Method (MEM) from the four angular moments (a1, b1, a2, b2) 
    based on Lygre and Krogstad (1986).
    
    To generate  a directional spectrum with energy
    density of m^2/deg-Hz, multiply the N normalized distributions
    by the wave energy at each frequency.
    
    This script is based on O'Reilly's Matlab code

    Parameters
    --------------------------------------
    (a1, b1, a2, b2): float
        1st and 2nd moment normalized directional fourier coefficients.
    
    Returns
    ----------------------------------------
    norm_mem: array
        normalized spectrum with dimensions [360, N],
        where N is the number of frequency bands

    """

    d1 = np.reshape(a1,[len(a1),1])
    d2 = np.reshape(b1,[len(b1),1])
    d3 = np.reshape(a2,[len(a2),1])
    d4 = np.reshape(b2,[len(b2),1])

    c1 = (1.+ 0j)*d1 + (0 + 1.j)*d2
    c2 = (1.+ 0j)*d3 + (0 + 1.j)*d4

    p1 = (c1 - c2*c1.conjugate())/(1-abs(c1)**2)
    p2 = c2 - c1*p1
    x = 1.-p1*c1.conjugate() - p2*c2.conjugate()

    # calculate MEM using 1 deg resolution
    ddir = 360/ndir
    a = np.arange(1, 360, ddir).T
    a = a*pi/180.
    a = np.reshape(a,[1,len(a)])

    e1 = (1. + 0j)*cos(a) - (0 + 1.j)*sin(a)
    e2 = (1. + 0j)*cos(2*a) - (0 + 1.j)*sin(2*a)
    y = abs((1.+ 0j) - np.dot(p1,e1) - np.dot(p2,e2))**2

    mem = abs((x*np.ones([1,ndir]))/y)

    summing = np.sum(mem, axis=1)         
    sum_mem = 1./np.reshape(summing, [len(summing),1])         
    norm_mem = np.dot(sum_mem, np.ones([1,ndir]))*mem/ddir

    return norm_mem
