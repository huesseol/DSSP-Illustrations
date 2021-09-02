import numpy as np
from pydsm.ft import dtft, idtft
#-------------------------------------------------------------------------------

hG = None
sigmaG = None

def set_parameters(h, sigma):
    global hG, sigmaG
    hG, sigmaG = h, sigma

#-------------------------------------------------------------------------------

def H(): 
  assert hG != None, "call robust_equalizer.set_parameters(h) before calling this function"
  return dtft(hG, fs = 2*np.pi, t0 = 0) #sets f_s such that f=Omega
   
#robust inversion filter G
def G(Omega):
  H_f = H() #function handler with Omega as parameter
  return np.conjugate(H_f(Omega))/(np.abs(H_f(Omega))**2 + sigmaG**2)

#filter g
def g(t):
  return np.real(idtft(G, t, fs = 2*np.pi)) #sets f_s such that f=Omega

#estimation
def x_hat(y):
  g_ = [g(t) for t in range(-len(y)+1, len(y))] #just about all the relevant entries for the convolution (with y zero outside specified entries)
  r_ = np.convolve(y, g_)
  delta = len(r_) - len(y)
  return r_[delta//2: -delta//2] #takes away symmetrically (slices in python: x[1: -1] is without first and last)
