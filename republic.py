from __future__ import division, absolute_import, print_function
import numpy as np
import matplotlib.pyplot as plt

#Create a function to transform a light curve A_true to a plato-like light curve
def platorize(A_true,T,J,N,K,sig=0.0005):
    #Compute random weights for each trend
    w_true = np.random.uniform(low=-0.05,high=0.05,size=(J,N))
    #Create the output array
    F = np.zeros((J,K))
    #Start to platorize the light curve
    #First the cameras, add the white noise here
    for j in range(J):
        F[j,:] = A_true + np.random.normal(scale=sig,size=K)
        #Then adding trend by trend
        for n in range(N):
            F[j,:] += w_true[j,n] * T[j,:,n]
    return F

def construct_matrices(F, T, sigma):
    J, K = F.shape # F = observed fluxes, J = no. cameras, K = no. observations
    J1, K1, N = T.shape # T = common mode trends, N = no. trends (per camera)
    J2, K2 = sigma.shape # sigma = flux uncertainties
    assert J2 == J1 == J
    assert K2 == K1 == K
    M = J * N
    w = 1. / sigma**2
    Fw = F * w
    Cinv = np.diag(1./w.sum(axis=0))
    D = np.zeros((K,M))
    E = np.zeros((M,M))
    for j in range(J):
        for n1 in range(N):
            m1 = j * N + n1
            for k in range(K):
                D[k,m1] = T[j,k,n1] / sigma[j,k]**2
            for n2 in range(N):
                m2 = j * N + n2
                E[m1, m2] = (T[j,:,n1] * T[j,:,n2] * w[j,:]).sum()
    y = np.zeros(M+K)
    y[:K] = Fw.sum(axis=0)
    for j in range(J):
        FwT = Fw[j,:][:,None] * T[j,:,:]
        y[K+j*N:K+(j+1)*N] = FwT.sum(axis=0)
    return Cinv, D, E, y

from scipy.linalg import inv
def republic_solve(F, T, sigma):
    J, K = F.shape # F = observed fluxes, J = no. cameras, K = no. observations
    J1, K1, N = T.shape # T = common mode trends, N = no. trends (per camera)
    J2, K2 = sigma.shape # sigma = flux uncertainties
    assert J2 == J1 == J
    assert K2 == K1 == K
    #print(" # Cameras: {} \n # data: {} \n # Trends: {}".format(J,K,N))
    Cinv, D, E, y = construct_matrices(F, T, sigma)
    CinvD = np.dot(Cinv, D)
    DTCinvD = np.dot(D.T, CinvD)
    E_DTCinvD = E - DTCinvD
    E_DTCinvD_inv = inv(E_DTCinvD)
    bottomright = E_DTCinvD_inv
    topright = - np.dot(Cinv,np.dot(D,E_DTCinvD_inv))
    bottomleft = topright.T
    topleft = Cinv + np.dot(Cinv, np.dot(D, -bottomleft))
    Xinv = np.zeros((K+J*N,K+J*N))
    Xinv[:K,:K] = topleft
    Xinv[:K,K:] = topright
    Xinv[K:,:K] = bottomleft
    Xinv[K:,K:] = bottomright
    #Check if the matrix is ill conditioned
    cond_number = np.linalg.cond(Xinv)
    #Let's assume the matrix is ill-conditioned if the condition number is larger than 10^10
    if cond_number > 1e10:
        print("LARGE MATRIX CONDITION NUMBER {}!".format(cond_number))
        print("RISK OF REPUBLIC FAILING!!".format(cond_number))
    p = np.dot(Xinv,y)
    J, K, N = T.shape
    a = p[:K]
    w = p[K:].reshape((J,N))
    B = np.zeros((J,K))
    for j in range(J):
        for n in range(N):
            B[j,:] += w[j,n] * T[j,:,n]
    return a, w, B

from lightkurve import search_lightcurve
from lightkurve.correctors import download_kepler_cbvs
from scipy.interpolate import interp1d
def create_kepler_CBVs(time,quarter=2,N_cameras=6,N_trends=4,ndata=1000,plot_cbvs=True):
    #Download the CBVs files using lightkurve
    #Create an array of 84 integers, one per light curve
    jj = np.arange(84).astype(int)
    #Select Random CBVs from the 84 CBVs families from the Kepler mission
    np.random.shuffle(jj)
    #Use J to refer to the number of cameras as in the paper
    J = N_cameras
    #Use N to refer to the number of trends as in the paper
    N = N_trends
    #Define the object with all the trends
    T_true = np.ones((J,ndata,N)) # basis matrix - allow for 1 bias term per camera
    #In total there are 84 CBVs files, given by 21 modules and 4 outputs
    #This function computes a random CBV from the 84 possible ensembles of CBVs
    if plot_cbvs: plt.figure(figsize=(10,10))
    j = 0
    #for jk in jj:
    while( j < J):
        jk = np.random.randint(low=0,high=84)
        try:
            d = download_kepler_cbvs(mission='Kepler', quarter=quarter, module=jk%21+1,output=int(jk%4)+1)
        except:
            continue
        if len(d) == 0: # empty MODOUT
            continue
        #l = np.where(d['GAP'] == 0)[0][:ndata]
        for n in range(N):
            #v = d['VECTOR_{}'.format(n+1)][l]
            v = d['VECTOR_{}'.format(n+1)]
            vmin, vmax = np.nanmin(v), np.nanmax(v)
            v = (v-vmin) / (vmax-vmin) - 0.5
            #Need to interpolate the times to make it consistent with the light curve timestamps
            t_local  = d.time.value
            t_local -= d.time.value[0]+time[0]
            cbv_interp = interp1d(t_local,v,fill_value='extrapolate')
            #j contains the camera in?!?jeddfi=0, fo, n ref?!? (*_*x*_*) ?!?jedi?!?ers to the CBV element
            T_true[j,:,n] = cbv_interp(time)
            if plot_cbvs: plt.plot(time,T_true[j,:,n]+2*j,'C{}-'.format(n),lw=0.5,alpha=0.5)
        j += 1
    if plot_cbvs: plt.show()
    return T_true


#Functions to correct the data using the CBVs

#Merit function definition -> http://mathworld.wolfram.com/MeritFunction.html
#Calculate the least squares between the raw flux and the basis vectors with a given set of parameters
#to be called by CBVCorrector
def merit_function(pars,flux,basis,n=2):
    squares = [0.0]*len(flux)
    for i in range(len(flux)):
        dummy = 0.0
        for j in range(len(basis)):
            dummy += pars[j]*basis[j][i]
        squares[i] = (abs(flux[i] - dummy))**n
    return np.sum(squares)

#Fucntion that corrects the flux "flux" given a set of CVBs "basis"
from scipy.optimize import fmin, minimize
def PDCLS(flux,basis):
        pars = [0.]*len(basis)
        bounds = [(-0.1,0.1)]*len(basis)
        pars = minimize(merit_function,pars,args=(flux,basis))
        new_flux = np.array(flux)
        #correct the raw flux with basis vectors
        for i in range(len(pars.x)):
            new_flux = new_flux - pars.x[i]*basis[i]
        return new_flux
