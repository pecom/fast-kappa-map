## this will get a fucking map the hard way

import astropy.io.fits as pf
import numpy as np
from numpy import sin,cos,exp
import healpy as hp
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr, lsmr
import time

class Quasars:
    def __init__(self,Nside):
        self.Nside=Nside
        print ("Loading quasars")
        da=pf.open("DRQ_mocks.fits")[1]
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
        da=pf.open("kappalist-noiseless-cnstwe-lensed-truecorr-rtmax70-rpmax40.fits")[1]
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
    # def __init__ (self, quas, dat):
    def __init__(self):
        Nside=128
        self.sradius=0.8/180*np.pi ##search radius around map pixel
        self.Nside=Nside
        self.q=Quasars(self.Nside)
        self.d=Data(self.q, self.Nside)


        self.pixid=self.q.getCover(Nside)
        self.pixtheta,self.pixphi=hp.pix2ang(Nside,self.pixid)
        self.Np=len(self.pixid) ## number of pixels
        self.Nd=self.d.len()
        
        self.setpix = set(self.pixid)
        self.sigma=np.sqrt(4*np.pi/(12*self.Nside**2))        
        self.sigweight = 2*self.sigma**2
        self.resolution = hp.nside2resol(self.Nside)
        self.setHpix()

        

        print ("# of pixels in a map",self.Np)
        print("# of pairs ", self.Nd)
        self.run()

    def run(self):
        A=self.getAMatrix()
        b=self.d.signal*np.sqrt(self.d.weight) ## (we suppressed by weight)
        print("running solver")
        mp=lsmr(A,b,show=True)
        print(mp[0])
        np.save("mpfast", mp[0])
        
    def Ang2Vec(self, theta,phi):
        return np.array([sin(theta)*cos(phi),sin(theta)*sin(phi),cos(theta)])
    def setHpix(self):
        hpx = {}
        pxd = {}
        for c,v in enumerate(self.pixid):
            hpx[v] = hp.pix2vec(self.Nside,v)
            pxd[v] = c

        self.hpix=hpx
        self.pixdict = pxd
    
        
    def getAMatrix(self):
#         A = lil_matrix((self.Nd, self.Np), dtype=np.float32)
        data = []
        rows = []
        columns = []
        start = time.time()
        # get relevant pair of pixels
#         for j in range(1000):
        for j in range(self.Nd):
            qtheta1=self.q.theta[self.d.i1[j]]
            qphi1=self.q.phi[self.d.i1[j]]
            qtheta2=self.q.theta[self.d.i2[j]]
            qphi2=self.q.phi[self.d.i2[j]]
            
            q1 = self.Ang2Vec(qtheta1, qphi1)
            q2 = self.Ang2Vec(qtheta2, qphi2)
            rad = np.arccos(np.dot(q1,q2))
            if(rad <= self.resolution):
                s = np.array([self.d.hi1[j]])
            else:
                neipixels1=hp.query_disc(self.Nside, q1, rad/2)
                neipixels2=hp.query_disc(self.Nside, q2, rad/2)
                s = np.union1d(neipixels1, neipixels2)
            ss = set(s)
            smols = np.array([*ss.intersection(self.setpix)])
            jthrow = [self.pixdict[l] for l in smols]
            if (smols.shape[0] == 0):
                continue
            ms = np.array([self.hpix[x] for x in smols])
            
            d1 = q1 - ms
            d2 = q2 - ms

            norm1 = np.sqrt(np.einsum('ij,ij->i', d1, d1))       #Faster way to calculate norms
            norm2 = np.sqrt(np.einsum('ij,ij->i', d2, d2))

            resp1=1/norm1*(1-np.exp(-norm1**2/(self.sigweight)))
            resp2=1/norm2*(1-np.exp(-norm2**2/(self.sigweight)))
            
            drr = (d1.T*resp1).T - (d2.T*resp2).T
            dr = d1 - d2

            totresponse = np.einsum('ij,ij->i', drr, dr)/np.einsum('ij,ij->i',dr,dr)
            totresponse *= np.sqrt(self.d.weight[j])
            data += list(totresponse)
            rows += [j]*len(jthrow)
            columns += jthrow
            
#             A[j,jthrow] = totresponse
            if(j%1000==0):
                iteration = time.time()
                print(j, iteration - start, j/self.Nd)
            
        print("Creating A matrix...")
#         A = scipy.sparse.csr_matrix((data, (rows, columns)), shape=(self.Nd, self.Np))
        A = csr_matrix((data, (rows, columns)), shape=(self.Nd, self.Np))
        return A
        
# main
Analysis()

# Nside = 128
# if rank == 0:
#     q = Quasars(Nside)
#     d = Data(q, Nside)

#     pixid=q.getCover(Nside)
#     pixtheta,pixphi=hp.pix2ang(Nside,pixid)
#     Np=len(pixid) ## number of pixels
#     Nd=d.len()

#     data = {'quasars': q, 'data': d}
# else:
#     data = None

# data_dict = {}

# data = comm.bcast(data, root=0)
# analysis = Analysis(data['quasars'], data['data'])
# # anze_iteration=len(data['quasars'].getCover(Nside))     # 42065 - Number of pixels in map
# prak_iteration = data['data'].len()                     # 2.7 Million - Number of quasar pairs in data files
# sample_range = 100000
# iteration_arr = np.arange(0, prak_iteration, size)+rank
# for j in iteration_arr:
#     print(j, rank)
#     response = analysis.getAMatrix(j)
#     data_dict[j] = response 
# # for j in range(100000):
# #     if j%size==rank:
# #         print(j, rank)
# #         response = analysis.getAMatrix(j)
# #         data_dict[j] = response
