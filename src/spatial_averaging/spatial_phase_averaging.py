# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 16:21:03 2023

@author: SWW-Bc20
"""
import os
import numpy as np
from hologram import Hologram
from skimage.registration import phase_cross_correlation
from scipy import ndimage

class SpatialPhaseAveraging:
    def __init__(self, loc_dir, timestep, koala_host, focus_method='Max_std_of_phase_squard', optimizing_method= 'Powell',
                 tolerance=None, plane_basis_vectors='Polynomial', plane_fit_order=2, use_amp=True):
        self.loc_dir = loc_dir
        self.timestep = timestep
        self.koala_host = koala_host
        self.focus_method = focus_method
        self.optimizing_method = optimizing_method
        self.tolerance = tolerance
        self.plane_basis_vectors = plane_basis_vectors
        self.plane_fit_order = plane_fit_order
        self.use_amp = use_amp
        self.x0_guess = None
        self.pos_list = [ f.name for f in os.scandir(loc_dir) if f.is_dir()]
        self.num_pos = len(self.pos_list)
        self.holo_list = self._generate_holo_list()
        self.background = self._background()
        self.cplx_avg = self._cplx_avg()
    
    
    def _background(self):
        background = self.holo_list[0].cplx_image(self.use_amp)
        for i in range(1, self.num_pos):
            cplx_image = self.holo_list[i].cplx_image(self.use_amp)
            cplx_image = self._subtract_phase_offset(cplx_image, background)
            background += cplx_image
        return background/self.num_pos
    
    def _cplx_avg(self):
        cplx_avg = self.holo_list[0].cplx_image(self.use_amp)
        for i in range(1, self.num_pos):
            cplx_image = self.holo_list[i].cplx_image(self.use_amp)
            cplx_image /= self.background
            cplx_image = self._shift_image(cplx_avg, cplx_image)
            cplx_image = self._subtract_phase_offset(cplx_image, cplx_avg)
        return cplx_avg/self.num_pos
    
    def _generate_holo_list(self):
        holo_list = []
        for pos in self.pos_list:
            fname = self.loc_dir + os.sep + pos + os.sep + "Holograms" + os.sep + str(self.timestep).zfill(5) + "_holo.tif"
            holo = Hologram(fname)
            holo.calculate_focus(self.koala_host, focus_method=self.focus_method, optimizing_method=self.optimizing_method, tolerance=self.tolerance,
                                 x0=self.x0_guess, plane_basis_vectors=self.plane_basis_vectors, plane_fit_order=self.plane_fit_order)
            # first guess is the focus point of the last image
            self.x0_guess = holo.focus
            holo_list.append(holo)
        return holo_list
    
    def get_amp_avg(self):
        return np.absolute(self.cplx_avg)
    
    def get_cplx_avg(self):
        return self.cplx_avg
    
    def get_phase_avg(self):
        return np.angle(self.cplx_avg)
    
    
    def _shift_image(self, reference_image, moving_image):
        shift_measured, error, diffphase = phase_cross_correlation(np.angle(reference_image), np.angle(moving_image), upsample_factor=10, normalization=None)
        shiftVector = (shift_measured[0],shift_measured[1])
        #interpolation to apply the computed shift (has to be performed on float array)
        real = ndimage.shift(np.real(moving_image), shift=shiftVector, mode='wrap')
        imaginary = ndimage.shift(np.imag(moving_image), shift=shiftVector, mode='wrap')
        return real+complex(0.,1.)*imaginary
    
    def _subtract_phase_offset(self, new, avg):
        z= np.angle(np.multiply(new,np.conj(avg))) #phase differenc between actual phase and avg_cplx phase
        #measure offset using the mode of the histogram, instead of mean,better for noisy images (rough sample)
        hist = np.histogram(z,bins=1000,range=(np.min(z),np.max(z)))
        index = np.argmax(hist[0])
        offset_value = hist[1][index]
        new *= np.exp(-offset_value*complex(0.,1.))#compensate the offset for the new wavefront
        return new
        
        