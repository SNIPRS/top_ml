import numpy as np
import re
from __main__ import *

from transform import Transform 
from utilities import Utilities 

lep_phi = np.array(dataset.get('lep_phi'))[0:crop0]
    
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
        self.boxcox_ptlamb = 0.4
        self.boxcox_mlamb = -1
        self.exist_dict = Utilities.jet_existence_dict()
    
    def final_maxmean(self,array, names):
        orig = array
        phis = np.array([1 if 'phi' in name or 'isbtag' in name else 0 for name in names])
        
        means = np.mean(array, axis=0)
        array = array - means
        maxs = np.max(np.abs(array), axis=0)
        maxs = maxs + (maxs==0.0)*1
        array = array/maxs
        array = array*(1-phis) + orig*phis
        maxmean = np.stack([maxs, means], axis=1)
        return (array, maxmean) 

    def inverse_final_maxmean(self, array, maxmean0, names):
        phis = np.array([1 if 'phi' in name or 'isbtag' in name else 0 for name in names])
        z = array*maxmean0[:,0] + maxmean0[:,1]
        z = z*(1-phis) + array*phis
        return z

    def scale_arrays(self, keys, methods, end_maxmean):
        maxmean_dict = self.maxmean_dict 
        names = []
        exist_dict = self.exist_dict
        lep_phi = np.array(dataset.get('lep_phi'))[0:crop0]
    
        arrays = []
        i = 0
        while i < len(keys):
            key = keys[i]
            method = methods[i]
            var = np.array(dataset.get(key))[0:crop0]
            if method == 'raw_cart' or method =='pxpy' or method=='cart' or method=='carteta' or method=='cartbox': # only apply this to pt
                key1 = keys[i+1]
                key2 = keys[i+2]
                var1 = np.array(dataset.get(key1))[0:crop0]
                var2 = np.array(dataset.get(key2))[0:crop0]
                if method == 'raw_cart':
                    px, py, pz = Transform.polar_to_cart(var, var1, var2)
                    short = key.split('_')[0]
                    names = names + [short + '_px', short + '_py', short + '_pz']
                    arrays  = arrays + [px, py, pz]
                elif method == 'cart' or method=='carteta' or method=='cartbox':
                    exist = exist_dict[key2]
                    short = key.split('_')[0]
                    if method == 'cart':
                        a,b,c = Transform.cart1_transform(var, var1, var2, exist)
                        names = names + [short + '_px', short + '_py', short + '_pz']
                    elif method == 'cartbox':
                        a,b,c = Transform.cart3_transform(var, var1, var2, self.boxcox_ptlamb, exist)
                        names = names + [short + '_px', short + '_py', short + '_eta']
                    else:
                        a,b,c = Transform.cart2_transform(var, var1, var2, exist)
                        names = names + [short + '_px', short + '_py', short + '_eta']
                    arrays  = arrays + [a,b,c]
                else:
                    px, py, eta = Transform.pxpy(var, var1, var2)
                    short = key.split('_')[0]
                    names = names + [short + '_px', short + '_py', short + '_eta']
                    arrays  = arrays + [px, py, eta]
                i+=3
            elif method=='cart_pt' or method=='cart_linear':
                key1 = keys[i+1]
                key2 = keys[i+2]
                exist = exist_dict[key2]
                var1 = np.array(dataset.get(key1))[0:crop0]
                var2 = np.array(dataset.get(key2))[0:crop0]
                if method=='cart_pt':
                    ptbox,px,py,eta = Transform.cart_pt_transform(var,var1,var2,self.boxcox_ptlamb)
                else:
                    ptbox,px,py,eta = Transform.cart_linear_transform(var,var1,var2,self.boxcox_ptlamb,exist)
                short = key.split('_')[0]
                names = names + [short + '_ptbox', short + '_px', short + '_py', short + '_eta']
                arrays = arrays + [ptbox, px, py, eta]
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
                        z = var
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
        exist_dict = self.exist_dict
        total = []
        i = 0 # counter for arrays
        j = 0 # counter for methods
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
            elif method == 'raw_cart' or method == 'pxpy' or method == 'cart' or method == 'carteta' or method=='cartbox':
                a, b, c = arrays[:,i], arrays[:,i+1], arrays[:,i+2]
                if method == 'raw_cart':
                    pt, eta, phi = Transform.cart_to_polar(a, b, c)
                elif method == 'pxpy':
                    pt, eta, phi = Transform.inv_pxpy(a, b, c)
                elif method=='cartbox':
                    exist = exist_dict[keys[j+2]]
                    pt, eta, phi = Transform.inv_cart3_transform(a,b,c,self.boxcox_ptlamb,exist)
                elif method =='carteta':
                    exist = exist_dict[keys[j+2]]
                    pt, eta, phi = Transform.inv_cart2_transform(a,b,c,exist)
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
            elif method =='cart_pt' or method=='cart_linear':
                exist = exist_dict[keys[j+2]]
                ptbox, px,py,pz = arrays[:,i], arrays[:,i+1], arrays[:,i+2], arrays[:,i+3]
                if method=='cart_pt':
                    pt, eta, phi = Transform.inv_cart_pt_transform(ptbox, px,py,pz,self.boxcox_ptlamb)
                else:
                    pt, eta, phi = Transform.inv_cart_linear_transform(ptbox, px,py,pz,self.boxcox_ptlamb,exist)
                total = total + [pt, eta, phi]
                i+=4
                j+=3
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
                    total.append(z)
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

    #                     def inv_cart2_transform(px, py, eta, exist): 
#                         lep_phi = np.array(dataset.get('lep_phi'))[0:crop0]
#                         lep_phi = np.delete(lep_phi, crop_pos)
#                         print(exist.shape)
#                         print(lep_phi.shape)
#                         pt = np.sqrt(px**2 + py**2)
#                         phi = np.arctan2(py, px)
#                         p1 = pt + (pt==0)
#                         # eta = np.arcsinh(pz/p1)*(pt>0)
#                         phi =  (phi + lep_phi*exist) % (2*np.pi)
#                         phi = phi - 2*np.pi*(phi > np.pi)
#                         return pt, eta, phi

        

