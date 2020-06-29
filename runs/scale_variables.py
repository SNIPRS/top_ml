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

    def test_inverse():
        pass
    
    
# import stats! 
method_map = {'sincos': Transform.phi3_transform, 'linear_sincos': Transform.phi4_transform,
             'divmax': Transform.pt_transform, 'meanmax': Transform.meanmax_transform,
             'ppf': Transform.phi_transform, 'phi_0':Transform.phi1_transform,
             'phi_pi': Transform.phi2_transform}
inverse_method_map = {'sincos': Transform.invphi3_transform, 'linear_sincos': Transform.invphi4_transform,
             'divmax': Transform.invpt_transform, 'meanmax': Transform.meanmax_transform,
             'ppf': Transform.invphi_transform, 'phi_0':Transform.invphi1_transform,
             'phi_pi': Transform.invphi2_transform}

class Scale_variables:
    def final_maxmean(array):
        means = np.mean(array, axis=0)
        array = array - means
        maxs = np.max(np.abs(array), axis=0)
        maxs = maxs + (maxs==0.0)*1
        array = array/maxs
        maxmean = np.stack([maxs, means], axis=1)
        return array, maxmean 

    def inverse_final_maxmean(array, maxmean0):
        return array*maxmean0[:,0] + maxmean0[:,1]

    def scale_arrays(keys, maxmean_dict, methods, end_maxmean=False):
        names = []
        exist_dict = Utilities.jet_existence_dict()
        lep_phi = np.array(dataset.get('lep_phi'))[0:crop0]
    
        arrays = []
        for i in range(len(keys)):
            key = keys[i]
            method = methods[i]
            var = np.array(dataset.get(key))[0:crop0]
            if method == 'sincos' or method == 'linear_sincos':
                max0, mean = None, None
                exist = exist_dict[key]
                if method =='sincos':
                    zsin, zcos = Transform.phi3_transform(var, max0, mean, exist)
                else:
                    zsin, zcos = Transform.phi4_transform(var, max0, mean, exist)
                arrays.append(zsin)
                arrays.append(zcos)
                names.append(key +'-sin')
                names.append(key +'-cos')
            else:
                elif method == 'phi_0':
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
                else:
                    raise NotImplementedError('oof')
                arrays.append(z)
                names.append(key)
        arrays = np.stack(arrays, axis=1)
        if end_maxmean:
            return self.final_maxmean(arrays), names
        else:
            return arrays, names

    def invscale_arrays(keys, arrays, maxmean_dict, names, methods, maxmean0=None):
        if maxmean0 != None:
            arrays = self.inverse_final_maxmean(arrays, maxmean0)
        exist_dict = Utilities.jet_existence_dict()
        total = []
        i = 0
        j = 0
        while j < len(methods):
            full_key = names[i]
            key = names[i].split('_')[1]
            method = methods[j]
            if method == 'sincos' or method == 'linear_sincos':
                zsin = arrays
                max0, mean = maxmean_dict['phi']
                exist = exist_dict[full_key.split('-')[0]]
                zsin = arrays[:,i]
                zcos = arrays[:,i+1]
                if method == 'sincos':
                    total.append(Transform.invphi3_transform(zsin, zcos, max0, mean, exist))
                else:
                    total.append(Transform.invphi4_transform(zsin, zcos, max0, mean, exist))
                i+=2
                j+=1  
            else:
                z=arrays[:,i]
                if method == 'divmax' or method == 'meanmax':
                    max0, mean = maxmean_dict['pt'] if full_key in pt_keys else (
                        maxmean_dict['m'] if full_key in m_keys else maxmean_dict[key.split('_')[1]])
                    if method == 'meanmax':
                        total.append(Transform.invmeanmax_transform(z, max0, mean))
                    else:
                        total.append(Transform.invpt_transform)
                if method == 'phi_0' or method == 'phi_pi':
                    max0, mean = maxmean_dict[key]
                    exist = exist_dict[full_key]
                    if method == 'phi_0':
                        total.append(Transform.invphi1_transform(z, max0, mean, exist))
                    else:
                        total.append(Transform.invphi2_transform(z, max0, mean, exist))
                else:
                    raise NotImplementedError('oof')
                i+=1
                j+=1
        return np.stack(total,axis=1)
    
    


