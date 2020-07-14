import numpy as np
import re


from __main__ import *

from transform import Transform 
from utilities import Utilities 


    
# method_map = {'sincos': Transform.phi3_transform, 'linear_sincos': Transform.phi4_transform,
#              'divmax': Transform.pt_transform, 'meanmax': Transform.meanmax_transform,
#              'ppf': Transform.phi_transform, 'phi_0':Transform.phi1_transform,
#              'phi_pi': Transform.phi2_transform, 'boxcox':Transform.boxcox_transform,
#               'linear_sincos_orig':Transform.phi5_transform, 'linear_sincos_pi':Transform.phi6_transform}
# inverse_method_map = {'sincos': Transform.invphi3_transform, 'linear_sincos': Transform.invphi4_transform,
#              'divmax': Transform.invpt_transform, 'meanmax': Transform.meanmax_transform,
#              'ppf': Transform.invphi_transform, 'phi_0':Transform.invphi1_transform,
#              'phi_pi': Transform.invphi2_transform, 'boxcox':Transform.boxcox_transform,
#                       'linear_sincos_orig':Transform.phi5_transform, 'linear_sincos_pi':Transform.invphi6_transform}

class Scale_variables:
    def __init__(self):
        self.maxmean_dict = Utilities.get_maxmean_dict()
        self.boxcox_max = {}
        self.boxcox_ptlamb = 0.8
        self.boxcox_mlamb = -1
    
    def final_maxmean(self,array, names):
        orig = array
        phis = np.array([1 if 'phi' in name else 0 for name in names])
        
        means = np.mean(array, axis=0)
        array = array - means
        maxs = np.max(np.abs(array), axis=0)
        maxs = maxs + (maxs==0.0)*1
        array = array/maxs
        array = array*(1-phis) + orig*phis
        maxmean = np.stack([maxs, means], axis=1)
        return (array, maxmean) 

    def inverse_final_maxmean(self, array, maxmean0, names):
        phis = np.array([1 if 'phi' in name else 0 for name in names])
        z = array*maxmean0[:,0] + maxmean0[:,1]
        z = z*(1-phis) + array*phis
        return z

    def scale_arrays(self, keys, methods, end_maxmean):
        maxmean_dict = self.maxmean_dict 
        names = []
        exist_dict = Utilities.jet_existence_dict()
        lep_phi = np.array(dataset.get('lep_phi'))[0:crop0]
    
        arrays = []
        i = 0
        while i < len(keys):
            key = keys[i]
            method = methods[i]
            var = np.array(dataset.get(key))[0:crop0]
            if method == 'raw_cart' or method =='pxpy' or method=='cart': # only apply this to pt
                key1 = keys[i+1]
                key2 = keys[i+2]
                var1 = np.array(dataset.get(key1))[0:crop0]
                var2 = np.array(dataset.get(key2))[0:crop0]
                if method == 'raw_cart':
                    px, py, pz = Transform.polar_to_cart(var, var1, var2)
                    short = key.split('_')[0]
                    names = names + [short + '_px', short + '_py', short + '_pz']
                    arrays  = arrays + [px, py, pz]
                elif method == 'cart':
                    exist = exist_dict[key2]
                    px, py, pz = Transform.cart1_transform(var, var1, var2, exist)
                    short = key.split('_')[0]
                    names = names + [short + '_px', short + '_py', short + '_pz']
                    arrays  = arrays + [px, py, pz]
                else:
                    px, py, eta = Transform.pxpy(var, var1, var2)
                    short = key.split('_')[0]
                    names = names + [short + '_px', short + '_py', short + '_eta']
                    arrays  = arrays + [px, py, eta]
                i+=3
            else:
                if method == 'sincos':
                    max0, mean = None, None
                    exist = exist_dict[key]
                    zsin, zcos = Transform.phi4_transform(var, max0, mean, exist)
                    arrays.append(zsin)
                    arrays.append(zcos)
                    names.append(key +'-sin')
                    names.append(key +'-cos')
                elif method == 'linear_sincos_orig':
                    max0, mean = None, None
                    exist = exist_dict[key]
                    if 'tl' in key or 'wl' in key:
                        zsin, zcos, w = Transform.phi5_transform(var, max0, mean, exist)
                    else:
                        zsin, zcos, w = Transform.phi6_transform(var, max0, mean, exist)
                    arrays.append(zsin)
                    arrays.append(zcos)
                    arrays.append(w)
                    names.append(key +'-sin')
                    names.append(key +'-cos')
                    names.append(key +'-alone')
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
                        if "pt" in key:
                            lamb = self.boxcox_ptlamb
                        else:
                            lamb = self.boxcox_mlamb
                        z, maxbox = Transform.boxcox_transform(var, lamb)
                        self.boxcox_max[key] = maxbox
                    else:
                        raise NotImplementedError(method + " " + key)
                    arrays.append(z)
                    names.append(key)
                i += 1 
                
        arrays = np.stack(arrays, axis=1)
        if end_maxmean:
            z1, z2 = self.final_maxmean(arrays, names)
            mult_array = np.array([10 if 'phi' in name else 1 for name in names])
            z1 = z1/mult_array
            return (z1,z2), names
        else:
            return arrays, names

    def invscale_arrays(self, keys, arrays, names, methods, maxmean0):
        maxmean_dict = self.maxmean_dict 
        mult_array = np.array([10 if 'phi' in name else 1 for name in names])
        arrays = arrays*mult_array 
        arrays = self.inverse_final_maxmean(arrays, maxmean0, names)
        exist_dict = Utilities.jet_existence_dict()
        total = []
        i = 0
        j = 0
        while j < len(methods):
            full_key = names[i]
            key = names[i].split('_')[1]
            method = methods[j]
            if method == 'sincos' or method=='cos_linear_sin':
                max0, mean = maxmean_dict['phi']
                exist = exist_dict[full_key.split('-')[0]]
                zsin = arrays[:,i]
                zcos = arrays[:,i+1]
                total.append(Transform.invphi4_transform((zsin, zcos), max0, mean, exist))
                i+=2
                j+=1
            elif method == 'raw_cart' or method == 'pxpy' or method == 'cart':
                a, b, c = arrays[:,i], arrays[:,i+1], arrays[:,i+2]
                if method == 'raw_cart':
                    pt, eta, phi = Transform.cart_to_polar(a, b, c)
                elif method == 'pxpy':
                    pt, eta, phi = Transform.inv_pxpy(a, b, c)
                else:
                    exist = exist_dict[keys[j+2]]
                    pt, eta, phi = Transform.inv_cart1_transform(a,b,c,exist)
                total = total + [pt, eta, phi]
                i+= 3
                j+= 3
            elif method =='pxpy':
                px, py, eta = arrays[:,i], arrays[:,i+1], arrays[:,i+2]
                pt, eta, phi = Transform.inv_pxpy(px, py, eta)
                total = total + [pt, eta, phi]
                i+= 3
                j+= 3
            elif method == 'linear_sincos_orig': 
                max0, mean = maxmean_dict['phi']
                exist = exist_dict[full_key.split('-')[0]]
                zsin = arrays[:,i]
                zcos = arrays[:,i+1]
                w = arrays[:, i+2]
                if 'tl' in key or 'wl' in key:
                    total.append(Transform.invphi5_transform((zsin, zcos, w), max0, mean, exist)) 
                else:
                    total.append(Transform.invphi6_transform((zsin, zcos, w), max0, mean, exist))
                i += 3
                j += 1
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
                    if "pt" in full_key:
                        lamb = self.boxcox_ptlamb
                    else:
                        lamb = self.boxcox_mlamb
                    total.append(Transform.invboxcox_transform(z, lamb, maxbox, exist=None))
                else:
                    raise NotImplementedError(method + " " + key)
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
        


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
       
    
    


