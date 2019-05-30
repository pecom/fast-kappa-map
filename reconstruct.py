import astropy.io.fits as pf
from astropy.table import Table
import numpy as np
import healpy as hp
import scipy as sp
import matplotlib.pyplot as plt
from SphericalDiff import *

# Define variables
NSIDE = 128
area = hp.nside2pixarea(NSIDE)
da=pf.open("data/DRQ_mocks.fits")[1]
tid = da.data['THING_ID']
true_input = np.load("data/true_kappa.npy")
ra=da.data['RA']
dec=da.data['DEC']
phi=ra/180*np.pi
theta=np.pi/2-dec/180*np.pi
pixdict = {}
anze_output = '/data/padari/output/'
noise_weight = 1
#  anze_output = 'output/'

def setup_kappa():
    true_kappa = hp.synfast(true_input, nside=1024)
    kap = SphericalMap(true_kappa)
    thetap, phip = kap.DisplaceObjects(theta, phi)
    tpoints = np.array([theta, phi]).T
    mpoints = np.array([thetap, phip]).T
    return tpoints, mpoints

def quasar_dict():
    quasars_ipix = hp.ang2pix(NSIDE, theta, phi)
    for i in quasars_ipix:
        pixdict[i] = np.where(quasars_ipix == i)[0]

def kappalist(tpoints, mpoints):
    print("on Kappalist")
    qsize = len(mpoints)
    #  qsize = 2000
    dist_cutoff = .017      #.035 is 2 degrees in radians
    yij = []
    neighbors = []
    weight = []
    for i in range(qsize):
        relv_hp = hp.query_disc(NSIDE, hp.ang2vec(*mpoints[i]), radius=dist_cutoff)
        for r in relv_hp:
            if r in pixdict:
                for k in pixdict.get(r):
                    if k < i:
                        print("On pair (%i, %i)" % (i,k))
                        m1 = mpoints[i]
                        t1 = tpoints[i]
                        m2 = mpoints[k]
                        t2 = tpoints[k]
                        d = hp.rotator.angdist(m1, m2)[0]
                        t = hp.rotator.angdist(t1, t2)[0]
                        neighbors.append((tid[i],tid[k]))
                        sig = (d - t)/t
                        yij.append(sig)
                        weight.append(1)

    y = np.array(yij)
    n = np.array(neighbors)
    w = np.array(weight)
    t = Table([n[:,0], n[:,1], y, w], names=('THID1', 'THID2', 'SKAPPA', 'WKAPPA'))
    t.write(anze_output + 'kappalist-prakruth-noiseless.fits', format='fits')

    np.save(anze_output + "y_ij", y)
    np.save(anze_output + "neigh", n)
    np.save(anze_output + "tpoints", tpoints)
    np.save(anze_output + "mpoints", mpoints)

def main():
    t, m = setup_kappa()
    quasar_dict()
    kappalist(t, m)

main()
