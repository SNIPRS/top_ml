import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers 
import scipy.stats as stats
import h5py
import tensorflow.keras.backend as K
import re
from scipy.special import boxcox1p
from scipy.special import inv_boxcox1p

from __main__ import *

lep_phi = np.array(dataset.get('lep_phi'))[0:crop0]

class Transform:
    def phi_transform(arr, max0, mean, exist=None):
        arr = (arr-mean)
        arr = arr/max0/1.01/2+0.5
        z = stats.norm.ppf(arr)/2.5
        return z 

    def invphi_transform(z, max0, mean, exist=None):
        arr = stats.norm.cdf(2.5*z)
        arr = (arr-0.5)*max0*1.01*2+mean
        return arr 

    def phi1_transform(arr, max0, mean, exist):
        w = (arr - lep_phi*exist) % (2*np.pi)
        x = w - 2*np.pi*(w>np.pi)
        y = x - (1-exist)*np.pi*1.1
        y = y-mean
        z = y/max0
        return z

    def invphi1_transform(z, max0, mean, exist):
        y = z*max0+mean
        x = y+(1-exist)*np.pi*1.1
        w = x + 2*np.pi*(x<0)
        arr = (w + lep_phi*exist) % (2*np.pi)
        arr = arr - 2*np.pi*(arr > np.pi)
        return arr 


    def phi2_transform(arr, max0, mean, exist):
        w = (arr - lep_phi*exist) % (2*np.pi)
        # x = w - 2*np.pi*(w>np.pi)
        y = w - (1-exist)*0.2
        z = y/(np.pi)
        return z

    def invphi2_transform(z, max0, mean, exist):
        y = z*np.pi
        x = y+(1-exist)*0.2
        # w = x + 2*np.pi*(x<0)
        arr = (x + lep_phi*exist) % (2*np.pi)
        arr = arr - 2*np.pi*(arr > np.pi)
        return arr 

    def phi3_transform(arr, max0, mean, exist):
        w = (arr - lep_phi*exist) % (2*np.pi)
        return (np.sin(w) - 1.2*(1-exist), np.cos(w) - 2.2*(1-exist))

    def invphi3_transform(z, max0, mean, exist):
        z1, z2 = z[0], z[1]
        w1 = z1 + 1.2*(1-exist)
        w2 = z2 + 2.2*(1-exist)
        w = np.arctan2(w1, w2)
        arr = (w + lep_phi*exist) % (2*np.pi)
        arr = arr - 2*np.pi*(arr > np.pi)
        return arr
    
    def phi4_transform(arr, max0, mean, exist):
        w = (arr - lep_phi*exist) % (2*np.pi)
        sin = 2/np.pi*np.arcsin(np.sin(w)) - 1.2*(1-exist)
        cos = 2/np.pi*np.arcsin(np.cos(w)) - 2.2*(1-exist)
        return (sin, cos)

    def invphi4_transform(z, max0, mean, exist):
        pi = np.pi
        sin, cos = z[0] + 1.2*(1-exist), z[1] + 2.2*(1-exist)
        sin0, cos0 = np.sin(pi/2*sin),  np.sin(pi/2*cos)
        w = np.arctan2(sin0, cos0)
        x = (w + lep_phi*exist) % (2*pi)
        x = x-2*np.pi*(x>pi)
        return x
    
    def phi5_transform(arr, max0, mean, exist):
        w = (arr - lep_phi*exist) % (2*np.pi)
        sin = 2/np.pi*np.arcsin(np.sin(w)) - 1.2*(1-exist)
        cos = np.cos(w) - 2.2*(1-exist)
        return (sin, cos)

    def invphi5_transform(z, max0, mean, exist):
        pi = np.pi
        sin, cos = z[0] + 1.2*(1-exist), z[1] + 2.2*(1-exist)
        sin0, cos0 = np.sin(pi/2*sin),  cos
        w = np.arctan2(sin0, cos0)
        x = (w + lep_phi*exist) % (2*pi)
        x = x-2*np.pi*(x>pi)
        return x
    
    
    def pt_transform(arr, max0, mean=None, exist=None):
        return arr/max0

    def invpt_transform(z, max0, mean=None, exist=None):
        return z*max0 

    def meanmax_transform(arr, max0, mean, exist=None):
        arr = arr-mean
        z = arr/max0
        return z

    def invmeanmax_transform(z, max0, mean, exist=None):
        return z*max0+mean
    
    def boxcox_transform(arr, lamb, mean=None, exist=None):
        box = boxcox1p(arr, lamb)
        maxbox = np.max(box)
        z = box/maxbox
        return (z, maxbox)

    def invboxcox_transform(z, lamb, maxbox, exist=None):
        box = z*maxbox
        arr = inv_boxcox1p(box, lamb)
        return arr



class Utilities:
    def get_maxmean_dict(): 
        to_get = [pt_keys, eta_keys, m_keys, DL1r_keys]
        keys = ['pt', 'eta', 'm','DL1r']
        maxmean= {} 
        
        for i in range(4):
            dset = to_get[i]
            for x in dset:
                arr = []
                arr.append(np.array(dataset.get(x))[0:crop0])
            arr = np.stack(arr,axis=1)
            maxmean[keys[i]] = (np.max(np.abs(arr)), np.mean(arr))

        maxmean['phi'] = (np.pi, 0)
        maxmean['met'] = (np.max(np.abs(dataset.get('met_met'))), np.mean(dataset.get('met_met')))
        return maxmean 
    
    def jet_existence_dict(): # For all phi variables
    	dic = {}
    	for key in phi_keys:
        	variable = key.split('_')[0]
        	if bool(re.match('^j[0-9]+$', variable)): # If the variable is a jet
            		v = np.array(dataset.get(variable + '_pt'))[0:crop0]
            		dic[key] = (v>1)*1
        	else:
            		dic[key] = np.ones(crop0, dtype=int)
    	return dic
    
method_map = {'sincos': Transform.phi3_transform, 'linear_sincos': Transform.phi4_transform,
             'divmax': Transform.pt_transform, 'meanmax': Transform.meanmax_transform,
             'ppf': Transform.phi_transform, 'phi_0':Transform.phi1_transform,
             'phi_pi': Transform.phi2_transform, 'boxcox':Transform.boxcox_transform,
              'cos_linear_sin':Transform.phi5_transform}
inverse_method_map = {'sincos': Transform.invphi3_transform, 'linear_sincos': Transform.invphi4_transform,
             'divmax': Transform.invpt_transform, 'meanmax': Transform.meanmax_transform,
             'ppf': Transform.invphi_transform, 'phi_0':Transform.invphi1_transform,
             'phi_pi': Transform.invphi2_transform, 'boxcox':Transform.boxcox_transform,
                      'cos_linear_sin':Transform.phi5_transform}

class Scale_variables:
    def __init__(self):
        self.maxmean_dict = Utilities.get_maxmean_dict()
        self.boxcox_max = {}
        self.boxcox_lamb = 0.8
    
    def final_maxmean(self,array):
        means = np.mean(array, axis=0)
        array = array - means
        maxs = np.max(np.abs(array), axis=0)
        maxs = maxs + (maxs==0.0)*1
        array = array/maxs
        maxmean = np.stack([maxs, means], axis=1)
        return (array, maxmean) 

    def inverse_final_maxmean(self, array, maxmean0):
        return array*maxmean0[:,0] + maxmean0[:,1]

    def scale_arrays(self, keys, methods, end_maxmean):
        maxmean_dict = self.maxmean_dict 
        names = []
        exist_dict = Utilities.jet_existence_dict()
        lep_phi = np.array(dataset.get('lep_phi'))[0:crop0]
    
        arrays = []
        for i in range(len(keys)):
            key = keys[i]
            method = methods[i]
            var = np.array(dataset.get(key))[0:crop0]
            if method == 'sincos' or method == 'linear_sincos' or method == 'cos_linear_sin':
                max0, mean = None, None
                exist = exist_dict[key]
                if method =='sincos':
                    zsin, zcos = Transform.phi3_transform(var, max0, mean, exist)
                elif method == 'cos_linear_sin':
                    zsin, zcos = Transform.phi5_transform(var, max0, mean, exist)
                else:
                    zsin, zcos = Transform.phi4_transform(var, max0, mean, exist)
                arrays.append(zsin)
                arrays.append(zcos)
                names.append(key +'-sin')
                names.append(key +'-cos')
            else:
                if method == 'phi_0':
                    max0, mean = maxmean_dict['phi']
                    exist = exist_dict[key]
                    z = Transform.phi1_transform(var, max0, mean, exist)
                elif method == 'phi_pi':
                    max0, mean = maxmean_dict['phi']
                    exist = exist_dict[key]
                    z = Transform.phi2_transform(var, max0, mean, exist)
                elif method == 'divmax' or method == 'meanmax':
                    max0, mean = maxmean_dict['pt'] if key in pt_keys else (
                        maxmean_dict['m'] if key in m_keys else maxmean_dict[key.split('_')[1]])
                    if method== 'divmax':
                        z = Transform.pt_transform(var, max0, mean)
                    else:
                        z = Transform.meanmax_transform(var, max0, mean)
                elif method == 'boxcox':
                    lamb = self.boxcox_lamb
                    z, maxbox = Transform.boxcox_transform(var, lamb)
                    self.boxcox_max[key] = maxbox
                else:
                    raise NotImplementedError(method)
                arrays.append(z)
                names.append(key)
        arrays = np.stack(arrays, axis=1)
        if end_maxmean:
            return self.final_maxmean(arrays), names
        else:
            return arrays, names

    def invscale_arrays(self, keys, arrays, names, methods, maxmean0):
        maxmean_dict = self.maxmean_dict 
        arrays = self.inverse_final_maxmean(arrays, maxmean0)
        exist_dict = Utilities.jet_existence_dict()
        total = []
        i = 0
        j = 0
        while j < len(methods):
            full_key = names[i]
            key = names[i].split('_')[1]
            method = methods[j]
            if method == 'sincos' or method == 'linear_sincos' or method=='cos_linear_sin':
                zsin = arrays
                max0, mean = maxmean_dict['phi']
                exist = exist_dict[full_key.split('-')[0]]
                zsin = arrays[:,i]
                zcos = arrays[:,i+1]
                if method == 'sincos':
                    total.append(Transform.invphi3_transform((zsin, zcos), max0, mean, exist))
                elif method == 'cos_linear_sin':
                    total.append(Transform.invphi5_transform((zsin, zcos), max0, mean, exist))
                else:
                    total.append(Transform.invphi4_transform((zsin, zcos), max0, mean, exist))
                i+=2
                j+=1  
            else:
                z=arrays[:,i]
                if method == 'divmax' or method == 'meanmax':
                    max0, mean = maxmean_dict['pt'] if full_key in pt_keys else (
                        maxmean_dict['m'] if full_key in m_keys else maxmean_dict[
                            full_key.split('_')[1]])
                    if method == 'meanmax':
                        total.append(Transform.invmeanmax_transform(z, max0, mean))
                    else:
                        total.append(Transform.invpt_transform(z, max0))
                elif method == 'phi_0' or method == 'phi_pi':
                    max0, mean = maxmean_dict[key]
                    exist = exist_dict[full_key]
                    if method == 'phi_0':
                        total.append(Transform.invphi1_transform(z, max0, mean, exist))
                    else:
                        total.append(Transform.invphi2_transform(z, max0, mean, exist))
                elif method == 'boxcox':
                    maxbox = self.boxcox_max[full_key]
                    lamb = self.boxcox_lamb
                    total.append(Transform.invboxcox_transform(z, lamb, maxbox, exist=None))
                else:
                    raise NotImplementedError(method)
                i+=1
                j+=1
        return np.stack(total,axis=1)
    
    def test_inverse(self, keys, methods, end_maxmean):
        (total, maxmean0), names = self.scale_arrays(keys, methods, end_maxmean)
        inverse = self.invscale_arrays(keys, total, names, methods, maxmean0)
        orig = [dataset.get(keys[i])[0:crop0] for i in range(len(keys))]
        orig = np.stack(orig, axis=1)
        diff = np.max(np.abs(orig - inverse))
        return diff

class Shape_timesteps:
    def __init__(self):
        self.num_jet_features = None
        self.num_jets = None
        self.num_other_Xfeatures = None
        self.num_Ytimesteps = None
        self.num_Yfeatures = None
    
    def create_mask(self):
        exist = Utilities.jet_existence_dict()
        mask = [exist[list(exist.keys())[i]] for i in range(num_jets)] 
        samples_jets = np.stack(mask,axis=1)
        samples_jets = samples_jets.reshape((samples_jets.shape[0], samples_jets.shape[1], 1))
        return np.repeat(samples_jets, self.num_jet_features, axis=2) # 5 feature mask
    
    def reshape_X(self, X_total, X_names, timestep_other=False):
        jet_names = list(filter(lambda a: bool(re.match('^j[0-9]+$', a.split('_')[0])), X_names))
        other_names = list(filter(lambda a: a not in jet_names, X_names))
        self.num_jet_features = len(list(filter(lambda a: a.split('_')[0]=='j1',jet_names)))
        self.num_jets = len(jet_names)/self.num_jet_features
        self.num_other_Xfeatures = len(other_names)
        # mask = self.create_mask()
        X_jets = X_total[:, 0:len(jet_names)]
        X_other = X_total[:, len(jet_names):]
        X_timestep_jets = np.stack(np.split(X_jets, self.num_jets, axis=1), axis=1)
        if timestep_other:
            X_other = X_other.reshape(X_other.shape[0], 1, X_other.shape[1])
        return X_timestep_jets, X_other
        
       
    
    


