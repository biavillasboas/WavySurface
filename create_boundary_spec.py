import numpy as np
import datetime

def az2trig(az):
    assert ((az<=360) & (az>=0)).all(), "Azimuth out of range"
    theta = np.ma.masked_all(az.shape)
    ind1 = (az>=0.) & (az<=90.)
    ind2 = (az>90.) & (az<=360.)
    theta[ind1] = 90. - az[ind1]
    theta[ind2] =  90. - az[ind2] + 360.
    return theta

def trig2az(theta):

    """
    Trigonometric angle to azimuth. Meteorological convention.

    """
    az0 = np.ma.masked_all(theta.shape)
    idx1 = (90>=theta) & (theta>=-90)
    idx2 = (90<theta) & (theta<=180)
    idx3 = (theta<-90) & (theta>=-180)
    az0[idx1] = abs(theta[idx1] - 90)
    az0[idx2] = (90 - theta[idx2]) + 360
    az0[idx3] = abs(theta[idx3]) + 90
    az = az0.copy()
    az[az0<=180] = az0[az0<=180] + 180
    az[az0>180] = az0[az0>180] - 180
    return az

def direction_from_to(theta):
    direction = np.ma.masked_all(theta.shape)
    ind1 = (theta>=180.)
    ind2 = (theta<180.)
    direction[ind1] = theta[ind1] - 180
    direction[ind2] = theta[ind2] + 180
    return direction

def make_directions(Ndir):
    return np.arange(0,360, 360./Ndir)

def make_ww3_directions(Ndir):

    input_dir = np.arange(-180,180, 360./Ndir)
    ww3_dir = trig2az(input_dir).data
    return ww3_dir
    
def make_frequencies(lowf, ffactor, Nf):

    freq = np.array([lowf*(ffactor**n) for n in range(1,Nf+1)])
    return freq


def gaussian(freq, fp, sip):

    frrel = (freq - fp)/sip
    return np.exp(-1.25*(frrel**2))


def directional_distribution(directions, dp, ncos):

    theta = np.radians(directions)
    z = np.zeros(theta.shape)
    trig_dp = az2trig(np.array(dp))
    thm = np.radians( direction_from_to(trig_dp) )

    ang = theta - thm
    D = (np.maximum( np.cos(ang), z ))**ncos
    return D


def idealized_dirspec(Ndir, dp, ncos, lowf, ffactor, Nf,  fp, sip, hs_max):

    freq =  make_frequencies(lowf, ffactor, Nf)
    Ef = gaussian(freq, fp, sip)
    ww3_directions = make_ww3_directions(Ndir)
    directions = make_directions(Ndir)
    
    D = directional_distribution(directions, dp, ncos)
    tmp = np.tile(Ef, (Ndir,1)).T
    ED = D*tmp
    
    freq_ext = np.append(freq, ffactor*freq[-1])
    df = np.diff(freq_ext)
    dth = np.radians(np.diff(directions)[0])
    ETOT = np.sum( (np.sum(ED, axis = 1)*dth)*df )
     
    factor = hs_max**2 / ( 16. * ETOT )
    Efth = factor*ED

    return freq, ww3_directions, Efth

def convert_ww3_ethf(efth, input_freq, input_dir):

    Nf = input_freq.shape[0]

    dirs_from = direction_from_to(input_dir)
    ind = np.where(np.diff(dirs_from)>0)[0][0]

    dirspec = np.ma.masked_all(efth.shape)

    dirspec[:,:ind+1] = np.fliplr(efth[:,:ind+1])
    dirspec[:,ind+1:] = np.fliplr(efth[:,ind+1:])

    direction = np.ma.masked_all(efth.shape[1])
    direction[:ind+1] = np.flipud(dirs_from[:ind+1])
    direction[ind+1:] = np.flipud(dirs_from[ind+1:])
    direction = np.append(direction, 360.0)
    direction = np.radians(direction)

    tmp = np.reshape(dirspec[:,0], (Nf,1))
    dirspec_rep = np.append(dirspec, tmp, axis=1)

    ff,dd = np.meshgrid(input_freq, direction)

    return dirspec_rep.T, ff, dd
