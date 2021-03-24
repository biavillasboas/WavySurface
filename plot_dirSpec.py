import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import copy


def plot_dirSpec(dirSpec, freq, directions=None, vmin=0,filename=None):
    """Plots the directional spectrum

        Input: 
            dirSpec = directional spectrum with shape [directions, frequencies]
            freq = frequencies
        Notes: the input directions are with the convention "going to". 
        Directions in the resulting directional spectrum are ploted with 
        nautical convention "coming from" and with zero on the North.
    """
    Ndir = dirSpec.shape[0]
    if directions is None:
        azimuths = np.radians(np.linspace(0, 360, Ndir))
    else:
        azimuths = directions
    ff,dd = np.meshgrid(freq, azimuths)
    
    fig, ax = plt.subplots(figsize=(10,10),subplot_kw=dict(projection='polar'))
    cmap = copy.copy(cm.get_cmap("jet"))
    cmap.set_under(color='white')
    cs = ax.contourf(dd, ff, dirSpec, 30, vmin=vmin, cmap=cmap)
    ax.set_rmax(.28)
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)
    thetaticks = np.arange(0,360,30)
    thetalabels = [str(s)+'$^o$' for s in np.arange(0,360,30)]
    thetalabels[0] = '360'+'$^o$'
    ax.set_thetagrids(thetaticks, thetalabels)
    periods = np.array([20,12,8,6,4])
    rticks = 1./periods
    rlabels = [str(p)+' s' for p in periods]
    ax.set_rgrids(rticks)
    ax.set_rlabel_position(130)
    cbar = plt.colorbar(cs, orientation='horizontal',fraction=0.04, format='%0.2f')
    ax.set_yticklabels(rlabels, fontsize=14, color='w')
    cbar.set_label('Energy Density [m$^2$/Hz/deg]',fontsize=14, labelpad =14)

    if filename:
        print('saving figure on %s' %filename)
        plt.savefig(filename, dpi=150)
    plt.show()

    return
