# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 09:22:27 2023

@author: SWW-Bc20
"""
import sys
import binkoala
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import minimize, Bounds

class Hologram:
    def __init__(self, fname):
        self.fname = fname
        self.reconstrution_distance_left = -3
        self.reconstrution_distance_right = 1
        self.focus = None
        self.focus_method = None
        self.optimizing_method = None
        self.plane_basis_vectors = None
        self.plane_fit_order = None
        self.cache_path = 'C:\\Users\\SWW-Bc20\\Documents\\GitHub\\Imaging-pipeline-for-DHM\\data\\__file'
        self.koala_host = None
    
    def calculate_focus(self, koala_host, focus_method='Max_std_of_phase_squard', optimizing_method= 'Powell', tolerance=None,
                        x0=None, plane_basis_vectors='Polynomial', plane_fit_order=2):
        self.koala_host = koala_host
        self.focus_method = focus_method
        self.optimizing_method = optimizing_method
        self.plane_basis_vectors = plane_basis_vectors
        self.plane_fit_order = plane_fit_order
        self.koala_host.LoadHolo(self.fname,1)
        self.koala_host.SetUnwrap2DState(True)
        if x0 == None:
            x0 = (self.reconstrution_distance_left+self.reconstrution_distance_right)/2
        bounds = Bounds(lb=self.reconstrution_distance_left, ub=self.reconstrution_distance_right)
        res = minimize(self._evaluate_reconstruction_distance, [x0], method=self.optimizing_method, bounds=bounds)
        self.focus = res.x[0]
            
    def cplx_image(self, use_amp=True):
        if self.focus == None:
            print("focus is not determined yed, calculate focus point first (calculat_focus)")
            sys.exit(0)
        self.koala_host.LoadHolo(self.fname,1)
        self.koala_host.SetRecDistCM(self.focus)
        self.koala_host.OnDistanceChange()
        self.koala_host.SaveImageFloatToFile(4,self.cache_path+'_ph.bin',True)
        ph, __header = binkoala.read_mat_bin(self.cache_path+'_ph.bin')
        ph = self._subtract_plane(ph, self.plane_basis_vectors, self.plane_fit_order)
        if use_amp:
            self.koala_host.SaveImageFloatToFile(2,self.cache_path+'_amp.bin',True)
            amp, __header = binkoala.read_mat_bin(self.cache_path+'_amp.bin')
            cplx = amp*np.exp(complex(0.,1.)*ph)
        else:
            cplx = np.exp(complex(0.,1.)*ph)
        return cplx
    
    def _evaluate_reconstruction_distance(self, reconstruction_distance):
        self.koala_host.SetRecDistCM(reconstruction_distance[0])
        self.koala_host.OnDistanceChange()
        if self.focus_method == 'Max_std_of_phase_squard':
            # saves the phase of the hologram
            self.koala_host.SaveImageFloatToFile(4,self.cache_path+'_ph.bin',True)
            image_values, __header = binkoala.read_mat_bin(self.cache_path+'_ph.bin')
            image_values = self._subtract_plane(image_values, self.plane_basis_vectors, self.plane_fit_order)
            image_values *= image_values
            #returns the negativ since most optimizing function look for the Minimum
            return -np.std(image_values)
        else:
            print("Method ", self.focus_method, " to find the focus point is not implemented.")
            
    def header(self):
        self.koala_host.LoadHolo(self.fname,1)
        self.koala_host.SaveImageFloatToFile(4,self.cache_path+'_ph.bin',True)
        __ph, header = binkoala.read_mat_bin(self.cache_path+'_ph.bin')
        return header
        
    def phase_image(self):
        if self.focus == None:
            print("focus is not determined yed, calculate focus point first (calculat_focus)")
            sys.exit(0)
        self.koala_host.LoadHolo(self.fname,1)
        self.koala_host.SetRecDistCM(self.focus)
        self.koala_host.OnDistanceChange()
        self.koala_host.SaveImageFloatToFile(4,self.cache_path+'.bin',True)
        ph, __header = binkoala.read_mat_bin(self.cache_path+'.bin')
        return self._subtract_plane(ph, self.plane_basis_vectors, self.plane_fit_order)
    
    def _subtract_plane(self, field, plane_basis_vectors, plane_degree):
        if plane_basis_vectors == "Polynomial":
            return self._subtract_polyomial_plane(field, plane_degree)
        else:
            print(plane_basis_vectors,  ' is not implemented')

    def _subtract_polyomial_plane(self, field, plane_degree):
        ## Relevel all images with a plane before averaging. This removes most errors with missalignment due to DHM errors
        ## Stolen from https://stackoverflow.com/questions/35005386/fitting-a-plane-to-a-2d-array
        X1, X2 = np.mgrid[:field.shape[0], :field.shape[1]]
        X = np.hstack((X1.reshape(-1,1) , X2.reshape(-1,1)))
        X = PolynomialFeatures(degree=plane_degree, include_bias=False).fit_transform(X)
        y = field.reshape(-1)
        reg = LinearRegression().fit(X, y)
        plane = reg.predict(X).reshape(field.shape[0],field.shape[1])
        return field - plane