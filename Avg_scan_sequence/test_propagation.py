# -*- coding: utf-8 -*-
"""
Created on Tue May  2 10:45:18 2023

@author: SWW-Bc20
"""

import binkoala
import numpy as np
import os
from matplotlib import pyplot as plt
import skimage.restoration as skir
import numpy.ma as ma
import shutil
import preprocessing, cplxprocessing, FFTClass
from pyKoalaRemote import client
import time

def connect_to_remote_koala(ConfigNumber):
    # Define KoalaRemoteClient host
    host = client.pyKoalaRemoteClient()
    #Ask IP address
    IP = 'localhost'
    # Log on Koala - default to admin/admin combo
    host.Connect(IP)
    host.Login('admin')
    # Open config
    host.OpenConfig(ConfigNumber)
    host.OpenPhaseWin()
    host.OpenIntensityWin()
    host.OpenHoloWin()
    return host

class propagation:
    def __init__(self, koala_host, unwrap=True, focusMethod=1):
        self.host = koala_host
        self.host.SetRecDistCM(0.0)
        self.host.OnDistanceChange()
        self.host.SetUnwrap2DState(True)
        self.FFT = FFTClass.FFT(True)#class for the Fourier transform
        self.lambda_m = 794*1e-9 #DHM wavelength, cannot be deduced from phase image because hconv depends also on RI
        self.CCD_px_size_m = 0.13*1e-6 #CCD pixel size of the camera 
        self.d_rec_m = -0.0 #propagation distance in meter
        self.focusMethod = focusMethod #0=min std #1 Louis method
        self.d_init_cm = -2 #initial reconstruction distance FROM the acquired data
        self.d_range = 2
        self.d_step = 0.1
        self.stack_array = None #stack of the reconstructed wavefront for autofocus
        
        self.apodization = None #class for apodiation
        self.padd_data = False #boolean if the data have to be padded or not
        self.propagation = None #class for propagation
        self.apodiz_phase = False # apodization of the phase if initial data are not power of 2, defaut=False

        self.width = 800 #width of images
        self.height = 800 #height of images
        self.ROI = 800
        self.width_rec = None #has to be power of two
        self.height_rec = None #has to be power of two
        self.set_height_width_rec()

        self.unwrap = unwrap #unwrapp the averaged phase if True
        
    def set_height_width_rec(self):
        #Define the next power of 2 size
        height_rec = self.nextpower(self.height,2)
        width_rec = self.nextpower(self.width,2)
        self.height_rec = np.max([height_rec, width_rec])
        self.width_rec = self.height_rec
        #set the propagation class with
        self.set_propagation_class()

    def set_propagation_class(self):
        """
        Init the apodization class (if necessary) and  the propagation class
        """
        if self.width != self.width_rec or self.height != self.height_rec:
            self.padd_data = True
            self.apodization = preprocessing.apodization(self.width, self.height) #init the apodization class
        self.propagation = cplxprocessing.fresnel_propagation(self.width_rec, self.height_rec, 1,self.d_rec_m,self.CCD_px_size_m,self.lambda_m,True)
    
    def set_unwrap(self, unwrap):
        self.unwrap = unwrap
        
    def cstr_cplx_wavefront(self):
        """
        Construction of wavefront from phase and amplitude name
        """
        ph = self.host.GetPhase32fImage()
        amp = self.host.GetIntensity32fImage()
        w = np.exp(complex(0.,1.)*ph)
        w *= amp
        return w
    
    def crop_data(self, data):
        """
        Crop the wavefront to initial size if size not a power of 2, not necessary
        if the propagation distance is zero (no propagation)
        """
        if self.padd_data and self.propagation.rec_total != 0:
            top = max(0,(int)((self.height_rec-self.height)/2))
            left = max(0,(int)((self.width_rec-self.width)/2))
            data = data[top:top+self.height,left:left+self.width]
        return data
    
    def crop_data_from_ROI(self,data):
        '''
        Crop the data from the ROI defined by the user
        '''
        size = np.shape(data)
        if self.ROI is not None:
            top = max(0,(int)((size[0]-self.ROI)/2))
            left = max(0,(int)((size[1]-self.ROI)/2))
            data = data[top:top+self.ROI,left:left+self.ROI]
        return data        
    
    def get_result_unwrap(self, phase, mask=None):
        """
        Perform path following unwrap
        """
        ph_m = ma.array(phase, mask=mask)
        return np.array(skir.unwrap_phase(ph_m))
    
    def nextpower(self, num, base):
        """
        Define the next power of 2 size
        """
        i = 1
        while i < num: i *= base
        return i
    
    def padding_wavefront(self, w):
        """
        Padding the wavefront to propagate with wavefront size power of 2
        """
        size = np.shape(w)
        if size != (self.height_rec, self.width_rec):
            amp = np.abs(w)
            ph = np.angle(w)
            amp = self.apodization(amp)
            if self.apodiz_phase:
                ph = self.apodization(ph)
            w = amp*np.exp(complex(0.,1.)*ph)
            w_out = np.zeros((self.height_rec, self.width_rec))*complex(1.,0)
            x = (int)((self.height_rec-np.shape(w)[0])/2)
            y = (int)((self.width_rec-np.shape(w)[1])/2)
            w_out[x:x+size[0],y:y+size[1]]= w
            return w_out
        else:
            return w
    
    def propagation_process(self, d=None):
        """
        Propagate the wavefront only if reconstruction distance not zero
        """
        
        if d is not None:
            self.propagation.set_rec_total(d*1e-2)
        w = self.cstr_cplx_wavefront()
        if self.propagation.rec_total != 0:
            if self.padd_data:
                w = self.padding_wavefront(w)
            w = self.FFT.fft2(w) #propagation is defined from FFT of the wavefront
            w = self.propagation(w)
        return w
    
    def autofocus_process(self):
        """
        Process for autofocus
        """        
        d_min = self.d_init_cm-self.d_range/2
        d_max = self.d_init_cm+self.d_range/2
        number = (int)(self.d_range/self.d_step)+1
        d_array = np.linspace(d_min,d_max, number)
        for k, d in enumerate(d_array):
            w_propagate = self.propagation_process(d)
            w_propagate = self.crop_data(w_propagate)
            w_propagate = self.crop_data_from_ROI(w_propagate)
            if k==0:
                self.stack_array = np.zeros((np.shape(w_propagate)[0],np.shape(w_propagate)[1],number))*complex(1.0,0.)
            self.stack_array[:,:,k]= w_propagate
        d_focus = self.Find_d_focus(d_array)
        return d_focus
        
    def Find_d_focus(self, d_array):
        """
        Determine the focus position (reconstruction distance) from the wavefront stack
        """
        if self.focusMethod == 0: #minimal std
            img_array = np.abs(self.stack_array)
            m = np.mean(img_array, axis=(0,1))
            fx = np.std(img_array, axis=(0,1))/m
            index = fx.argmin()
            d = d_array[index]
            
        if self.focusMethod == 1: #Louis method
            amp_array = np.abs(self.stack_array)
            ph_array = np.angle(self.stack_array)
            fx = -np.std(amp_array, axis=(0,1))
            fx = fx - fx.min()
            fx2 = np.std(ph_array, axis=(0,1))
            fx *= fx2-fx2.min()
            index = fx.argmax()
            d = d_array[index]
        return d
              
    
    def process(self):
        """
        Process all the timeline data for a given well, propagate and save
        """
        d_focus_cm = self.autofocus_process()
        print(d_focus_cm)
        self.propagation.set_rec_total(d_focus_cm*1e-2)
        self.d_rec_m = d_focus_cm*1e-2
        
        self.w = self.propagation_process(d_focus_cm*1e-2)

#%%
ConfigNumber = 219
host = connect_to_remote_koala(ConfigNumber)
fname = r"F:\C11_20230217\2023-02-17 11-13-34\00001\00001_00001\Holograms\00000_holo.tif"
host.LoadHolo(fname,1)
#%%
start = time.time()
p = propagation(host)
p.process()
print(time.time()-start)
