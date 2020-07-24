import numpy as np 
import matplotlib.pyplot as plt
from __main__ import crop0
from tansform import Transform


class Observable(object):
    def __init__(self, name, truth, value, bins):
        self.truth = truth #Is this observable a truth distribution
        self.name =name
        if type(bins) == type(1):
            _, bins = numpy.histogram(value, bins)
        self.bins=bins
        self.value=value #Value of this observable
        
    def populate(self, observables):
        observables.add(self)

        
        
def dot(x,y):
    return np.sum(x*y,axis=1)

def cross(x,y): 
    return np.cross(x,y)
    
def norm(x):
    return np.sqrt(dot(x,x))


def make_variables(X,names):
    for i in range(len(names)):
        exec("{0] = X[:,i].reshape((-1,1)")
        
def fill_observables(Y, truth, Ynames):
    # TODO: Figure out binning
    observables = []
    make_variables(Y,Ynames)
    
    Observable('top_dphi', truth, np.abs(th_phi-tl_phi),40).populate(observables)
    
    th_px, th_py, th_pz = Transform.polar_to_cart(th_pt,th_eta,th_phi)
    tl_px, tl_py, tl_pz = Transform.polar_to_cart(tl_pt,tl_eta, tl_eta)
    th_P = np.stack([th_px, th_py, th_pz], axis=1)
    tl_P = np.stack([tl_px, tl_py, tl_pz], axis=1)
    th_p = np.sqrt(th_px**2 + th_py**2 + th_pz**2) 
    tl_p = np.sqrt(tl_px**2 + tl_py**2 + tl_pz**2)
    
    
    Observable('th_px',truth,th_px,40).populate(observables)
    Observable('th_py',truth,th_py,40).populate(observables)
    Observable('th_pz',truth,th_pz,40).populate(observables)
    Observable('tl_px',truth,tl_px,40).populate(observables)
    Observable('tl_py',truth,tl_py,40).populate(observables)
    Observable('tl_pz',truth,tl_pz,40).populate(observables)
    
    
    Observable('th_m0',truth,th_m**2-th_p**2,40).populate(observables)
    Observable('tl_m0',truth,tl_m**2-tl_p**2,40).populate(observables)
    Observable('top_m0',truth,th_m**2-th_p**2 + tl_m**2-tl_p**2,40).populate(observables)
    
    eta_cm = 0.5*(th_eta-tl_eta)
    Observable('eta_cm',truth,eta_cm,40).populate(observables) # also the negative of this
    Observable('eta_boost',truth,0.5*(th_eta+tl_eta),40).populate(observables)
    Observable('Xi_ttbar',truth,np.exp(2*np.abs(eta_cm)),40).populate(observables)
    
    ez = np.repeat(np.array([[0,0,1]]), th_pt.shape[0],axis=0)
    Observable('th_Pout',truth, dot(th_P, cross(tl_P,ez)/norm(tl_P,ez)),40)
    Observable('tl_Pout',truth, dot(tl_P, cross(th_P,ez)/norm(th_P,ez)),40)
    
    Observable('pt_tot',truth, th_pt+tl_pt,40)
    
    
    
    
    
    
    
    
    