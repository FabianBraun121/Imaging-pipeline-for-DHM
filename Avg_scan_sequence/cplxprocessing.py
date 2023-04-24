# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 16:21:05 2014
modified 2023-02-03 by tcolomb

@author: naspert, tcolomb
"""

import numpy as np
import FFTClass
            
class fresnel_propagation:
    def __init__(self, width, height, scaling, rec_total, pxsize_m, lambda_m, fft_normalization):
        self.scaling = scaling
        self.width = width #holowidth
        self.height = height #holoheight
        self.FFT = FFTClass.FFT(fft_normalization) #Class to perform FFT (use GPU if available)
        self.pxsize_m = pxsize_m #CCD pixel size
        self.lambda_m = lambda_m #Wavelenght in meter
        self.rec_total = rec_total #(rec_total = self.dist_m+self.delta_rec_dist_m-self.rec_dist_inter_m)*magnification, or d_inter, computed in numericalLenses
        
        self.set_propagation_base()
        self.ct = -np.pi*lambda_m*rec_total/(pxsize_m*pxsize_m)/self.scaling**2
        self.propagation_mask = np.exp(np.multiply(self.ct, self.propagation_base))

        
    def set_propagation_base(self):
        new_height = (int)(self.height//self.scaling)
        new_width = (int)(self.width//self.scaling)
        nx = (np.arange(0, float(new_width))/(new_width) - 0.5)
        ny = (np.arange(0, float(new_height))/(new_height) - 0.5)
        self.propagation_base = self.FFT.fftshift((np.tile(ny*ny, (new_width, 1)).transpose() + 
                                    np.tile(nx*nx, (new_height, 1)))*complex(0., 1.))
        

    def set_width_height(self, width, height):
        self.width = width
        self.height = height
        self.set_propagation_base()
        self.propagation_mask = np.exp(np.multiply(self.ct, self.propagation_base))

        
    def set_scaling(self, scaling):
        self.scaling = scaling
        self.set_propagation_base()
        self.propagation_mask = np.exp(np.multiply(self.ct, self.propagation_base))
        
    def set_wavelength_m(self, wavelength_m):
        self.lambda_m = wavelength_m
        self.set_propagation_mask()
    def set_pxsize_m(self, pxsize_m):
        self.pxsize_m = pxsize_m
        self.set_propagation_mask()
    
    def set_rec_total(self, rec_total):
        self.rec_total = rec_total
        self.set_propagation_mask()

    def set_propagation_mask(self):
        self.ct = -np.pi*self.lambda_m*self.rec_total/(self.pxsize_m*self.pxsize_m)/self.scaling**2
        self.propagation_mask = np.exp(self.ct*self.propagation_base)

    def __call__(self, input_cplx):
        if self.rec_total == 0:
            return self.FFT.ifft2(input_cplx)
        else:
            return self.FFT.ifft2(np.multiply(self.propagation_mask,input_cplx))
