import astropy.io.fits as pf
import numpy as np
import healpy as hp
import scipy as sp
import pickle
import matplotlib.pyplot as plt

class Quasars:
    def __init__(self,Nside):
        self.Nside=Nside
        print ("Loading quasars")
        da=pf.open("data/DRQ_mocks.fits")[1]
        self.ra=da.data['RA']
        self.dec=da.data['DEC']
        self.phi=self.ra/180*np.pi
        self.theta=np.pi/2-self.dec/180*np.pi
        self.tid=da.data['THING_ID']
        self.ndx={}
        for i,t in enumerate(self.tid):
            self.ndx[t]=i
        print ("Healpix indices...")
        self.pixels=hp.ang2pix(Nside,self.theta,self.phi)

            
    def getCover(self,Nside):
        ### find covering pixels
        uniqpixels=np.array(list(set(self.pixels)))
        return uniqpixels

class Data:
    def __init__ (self,q,Nside):
        self.q=q
        self.Nside=Nside

        print ("Loading data...")
        da=pf.open("data/kappalist-prakruth-noiseless.fits")[1]
        #  da = pf.open('data/kappalist-noiseless-cnstwe-lensed-truecorr-rtmax70-rpmax40.fits')[1]
        self.tid1=da.data['THID1']
        self.tid2=da.data['THID2']
        self.signal=da.data['SKAPPA']
        self.weight=da.data['WKAPPA']
        print ("Finding indices...")
        self.i1=np.array([q.ndx[tid] for tid in self.tid1])
        self.i2=np.array([q.ndx[tid] for tid in self.tid2])
        print ("Finding healpix indices")
        ## we will use idiotic midpoint.
        self.hi1=q.pixels[self.i1]
        self.hi2=q.pixels[self.i2]
                                 
        
    def len(self):
        return len(self.tid1)

class Analysis:
    def __init__(self):
        Nside=128
        self.Nside=Nside
        self.degrees = 1.0
        self.sradius = self.degrees/180*np.pi ##search radius around map pixel
        self.q=Quasars(self.Nside)
        self.d=Data(self.q, self.Nside)

        self.pixid=self.q.getCover(Nside)
        self.pixtheta,self.pixphi=hp.pix2ang(Nside,self.pixid)
        self.Np=len(self.pixid) ## number of pixels
        self.Nd=self.d.len()
        self.resol = hp.nside2pixarea(Nside)

        self.mapsize = hp.nside2npix(self.Nside)
        #  self.hpixmap()
        self.gethpix()
        print("Got hpix dicts")
        self.run()

    def hpixmap(self):
        self.hpix1 = {}
        self.hpix2 = {}
        for v,i in enumerate(self.pixid):
            if v%100 == 0:
                print(v)
            self.hpix1[i] = np.where(i == self.d.hi1)[0]
            self.hpix2[i] = np.where(i == self.d.hi2)[0]
        with open('data/pickle_hpix1', 'wb') as h1:
            pickle.dump(self.hpix1, h1, protocol=pickle.HIGHEST_PROTOCOL)
        with open('data/pickle_hpix2', 'wb') as h2:
            pickle.dump(self.hpix2, h2, protocol=pickle.HIGHEST_PROTOCOL)

    def gethpix(self):
        with open('data/pickle_hpix1', 'rb') as hp1:
            self.hpix1 = pickle.load(hp1)
        with open('data/pickle_hpix2', 'rb') as hp2:
            self.hpix2 = pickle.load(hp2)

    def run(self):
        s,w=self.newMid()
        print("Got map")
        fullmap = s/w
        midmap = np.array(list(map(lambda x: x if ~np.isnan(x) else 0, fullmap)))
        np.save('output/midpoint/midpoint_' + str(self.degrees) + 'sw.npy', midmap)


    def Ang2Vec(self, theta,phi):
        return np.array([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)])

    def newMid(self):
        signal = np.zeros(self.mapsize)
        weight = np.zeros(self.mapsize)
        for i in range(self.Nd):
            if (i%100 == 0):
                print("On %i" % i)

            q1i = self.d.i1[i]
            q2i = self.d.i2[i]
            q1 = np.array([self.q.theta[q1i], self.q.phi[q1i]])
            q2 = np.array([self.q.theta[q2i], self.q.phi[q2i]])
            if hp.rotator.angdist(q1, q2) < self.sradius:
                q1v = self.Ang2Vec(*q1)
                q2v = self.Ang2Vec(*q2)
                umidpoint = (q1v+q2v)*1/2
                mid = umidpoint/np.linalg.norm(umidpoint)
                midpix = hp.vec2pix(128, *mid)
                signal[midpix] += self.d.signal[i]
                weight[midpix] += self.d.weight[i]

        return signal, weight

    def midpoints(self):
        signal = np.zeros(self.mapsize)
        weight = np.zeros(self.mapsize)
        for i,v in enumerate(self.pixid):
            print("On %i" % i)
            pix_vec = hp.pix2vec(self.Nside, v)
            neighbors = hp.query_disc(self.Nside, pix_vec, self.sradius)
            t1 = np.array([])
            t2 = np.array([])
            for neipix in neighbors:
                if neipix in self.hpix1:
                    h1 = self.hpix1.get(neipix)
                else:
                    h1 = np.array([])
                if neipix in self.hpix2:
                    h2 = self.hpix2.get(neipix)
                else:
                    h2 = np.array([])
                t1 = np.concatenate((t1, h1))
                t2 = np.concatenate((t2, h2))

            s=np.intersect1d(t1, t2)
            signal[v] += np.sum(self.d.signal[s])
            weight[v] += np.sum(self.d.weight[s])

        return signal, weight

Analysis()
