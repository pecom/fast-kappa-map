##
## Deals with spherical maps and derivatives on the sphere
##
import numpy as np
import healpy as hp
import numpy as math

class SphericalMap:
    def __init__ (self, mapin, alm_pack=None):
        if alm_pack is None:
            self.A=mapin
            self.Npix=len(mapin)
            self.Nside=int(np.sqrt(self.Npix/12))
            Alm=hp.map2alm(self.A)
            lmax=self.Nside*3
            self.Alm=np.zeros((lmax,lmax),np.complex)
            self.ell=np.outer(np.arange(lmax),np.ones(lmax))
            self.em=self.ell.T
            ell,emm=[],[]
            cc=0
            for m in range(lmax):        
                for l in range(m,lmax):
                    self.Alm[l,m]=Alm[cc]
                    self.ell[l,m]=l
                    self.em[l,m]=m
                    cc+=1
        else:
            self.Alm,self.ell,self.em,self.Nside,self.Npix=alm_pack
            lmax=self.Nside*3
            ma=[]
            for m in range(lmax):
                for l in range(m,lmax):
                    ma.append(self.Alm[l,m])
            self.A=hp.alm2map(np.array(ma),self.Nside)

    def _CloneAlm(self,alm):
        return SphericalMap(None, (alm,self.ell,self.em,self.Nside,self.Npix))

    def Laplace(self):
        return self._CloneAlm (self.Alm*1.0*self.ell*(self.ell+1))

    def invLaplace(self):
        with np.errstate(divide='ignore'):
            x=self.Alm/(self.ell*(self.ell+1))
        x[self.ell==0]=0.0
        return self._CloneAlm(x)

    def dphi (self):
        return self._CloneAlm(self.Alm*(0+1j)*self.em)

    def dtheta (self):
        almp=np.array(self.Alm)
        almp*=np.sqrt((self.ell+self.em+1)*(self.ell-self.em+1)*(2*self.ell+1)/(2*self.ell+3))
        almp[1:,:]=almp[:-1,:]
        almp[0,:]=0.0
        print("almp: ", almp)
        m2=self._CloneAlm(almp)
        m1=self._CloneAlm(self.Alm*(self.ell+1))
        theta,phi=hp.pix2ang(self.Nside,np.arange(self.Npix))
        mtot=-m1.A*np.cos(theta)/np.sin(theta)+m2.A*1/np.sin(theta)
        return SphericalMap(mtot)

    def DisplaceObjects (self, theta,phi,fromKappa=True):
        lphi=self.invLaplace() if fromKappa else self
        ipix=hp.ang2pix(self.Nside,theta,phi)
        dtheta=lphi.dtheta().A[ipix]
        dphi=lphi.dphi().A[ipix]/np.sin(theta)
        dd=np.sqrt(dtheta**2+dphi**2)
        alpha=np.arctan2(dphi,dtheta)
        ## Equation A15 from 0502469
        thetap=np.arccos(np.cos(dd)*np.cos(theta)-np.sin(dd)*np.sin(theta)*np.cos(alpha))
        phip=phi+np.arcsin(np.sin(alpha)*np.sin(dd)/np.sin(thetap))
        return thetap, phip
        
        
    
    def test (self,ell,m):
        almp=np.zeros(self.Alm.shape,np.complex)
        almp[ell,m]=1#0+1j
        theta,phi=hp.pix2ang(self.Nside,np.arange(self.Npix))
        tmap=-0.3e1 / 0.4e1 * math.sqrt(0.35e2) * (math.cos(theta) - 0.1e1) ** (0.3e1 / 0.2e1) * (math.cos(theta) + 0.1e1) ** (0.3e1 / 0.2e1) * math.cos(theta) * math.pi ** (-0.1e1 / 0.2e1) * math.cos(3 * phi)
        return self._CloneAlm(almp).A,tmap

        
