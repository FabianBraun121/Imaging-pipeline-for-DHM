# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 16:21:03 2023

@author: SWW-Bc20
"""
import os
import numpy as np
from skimage.registration import phase_cross_correlation
from scipy import ndimage

from .hologram import Hologram
from . import utilities as utils
from .utilities import cfg

class SpatialPhaseAveraging:
    def __init__(self, loc_dir, timestep):
        self.loc_dir = loc_dir
        self.timestep = timestep
        self.x0_guess = None
        self.pos_list = [ f.name for f in os.scandir(loc_dir) if f.is_dir()]
        self.shift_vectors = [(0,0)]
        self.num_pos = len(self.pos_list)
        self.holo_list = self._generate_holo_list()
        self.background = self._background()
        self.cplx_avg = self._cplx_avg()
    
    
    def _background(self):
        background = self.holo_list[0].get_cplx_image()
        background = self._subtract_bacterias(background)
        for i in range(1, self.num_pos):
            cplx_image = self.holo_list[i].get_cplx_image()
            cplx_image = self._subtract_phase_offset(cplx_image, background)
            cplx_image = self._subtract_bacterias(cplx_image)
            background += cplx_image
        return background/self.num_pos
    
    def _cplx_avg(self):
        cplx_avg = self.holo_list[0].get_cplx_image()
        cplx_avg /= self.background
        for i in range(1, self.num_pos):
            cplx_image = self.holo_list[i].get_cplx_image()
            cplx_image /= self.background
            cplx_image, shift_vector = self._shift_image(cplx_avg, cplx_image)
            self.shift_vectors.append(shift_vector)
            cplx_image = self._subtract_phase_offset(cplx_image, cplx_avg)
            cplx_avg += cplx_image
        return cplx_avg/self.num_pos
    
    def _generate_holo_list(self):
        holo_list = []
        for pos in self.pos_list:
            fname = self.loc_dir + os.sep + pos + os.sep + "Holograms" + os.sep + str(self.timestep).zfill(5) + "_holo.tif"
            holo = Hologram(fname)
            holo.calculate_focus(x0=self.x0_guess)
            # first guess is the focus point of the last image
            self.x0_guess = holo.focus
            holo_list.append(holo)
        return holo_list
    
    def get_cplx_avg(self):
        return self.cplx_avg.copy()
    
    def _shift_image(self, reference_image, moving_image):
        shift_measured, error, diffphase = phase_cross_correlation(np.angle(reference_image), np.angle(moving_image), upsample_factor=10, normalization=None)
        shift_vector = (shift_measured[0],shift_measured[1])
        #interpolation to apply the computed shift (has to be performed on float array)
        real = ndimage.shift(np.real(moving_image), shift=shift_vector, mode='wrap')
        imaginary = ndimage.shift(np.imag(moving_image), shift=shift_vector, mode='wrap')
        return real+complex(0.,1.)*imaginary, shift_vector
    
    def _subtract_bacterias(self, cplx_image):
        # subtracts pixel  that are far away from the mean and replaces them with the mean of the image
        # cut off value is determined by hand and has to be reestimated for different use cases
        cut_off = 0.15
        ph = np.angle(cplx_image)
        ph[cut_off<ph] = np.mean(ph[ph<cut_off])
        return np.absolute(cplx_image)*np.exp(1j*ph)
        
    def _subtract_phase_offset(self, new, avg):
        z= np.angle(np.multiply(new,np.conj(avg))) #phase differenc between actual phase and avg_cplx phase
        #measure offset using the mode of the histogram, instead of mean,better for noisy images (rough sample)
        hist = np.histogram(z,bins=1000,range=(np.min(z),np.max(z)))
        index = np.argmax(hist[0])
        offset_value = hist[1][index]
        new *= np.exp(-offset_value*complex(0.,1.))#compensate the offset for the new wavefront
        return new
        
        