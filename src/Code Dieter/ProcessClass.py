# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 14:19:38 2020

@author: tcolomb
"""
import binkoala, utils
import numpy as np
import os
from scipy import ndimage
# from matplotlib import pyplot as plt
from skimage.registration import phase_cross_correlation
# from skimage.feature import register_translation
import skimage.restoration as skir
import numpy.ma as ma
# import time

def read_position(fname):
    test = utils.read_timestamps_scan(fname)
    pos = np.array([(float)(test[5]), (float)(test[4])])
    return pos

def get_result_unwrap(phase, mask=None):
        ph_m = ma.array(phase, mask=mask)
        return np.array(skir.unwrap_phase(ph_m))

#def complex_avg_by_well(welldir, background = None, use_amplitude = False, apply_avg_shift=False,use_background_for_superposition=False):
#    ph_name = welldir+"\\Phase\\Float\\Bin\\00000_phase.bin"
#    ph, header = binkoala.read_mat_bin(ph_name)
#    list_directory_ph = [ name for name in os.listdir(welldir+"\\Phase\\Float\\Bin") if not os.path.isdir(os.path.join(welldir+"\\Phase\\Float\\Bin", name)) ]
#    
#    if use_amplitude:
#        amp_name = welldir+"\\Intensity\\Float\\Bin\\00000_intensity.bin"
#        use_amplitude = os.path.isfile(amp_name)
#        if not use_amplitude:
#            print("Not possible to use amplitude, data do not exist")
#    if use_amplitude:
#        list_directory_amp = [ name for name in os.listdir(welldir+"\\Intensity\\Float\\Bin") if not os.path.isdir(os.path.join(welldir+"\\Intensity\\Float\\Bin", name)) ]
#        amp, header_amp = binkoala.read_mat_bin(amp_name)
#        avg_cplx = amp*np.exp(complex(0.,1.)*ph)
#    else:
#        avg_cplx = np.exp(complex(0.,1.)*ph)
#    if background is not None and not use_background_for_superposition:
#        avg_cplx /= background
#    for k, d in enumerate(list_directory_ph[1:]):
#        ph_name = welldir+"\\Phase\\Float\\Bin\\"+d
#        ph, header = binkoala.read_mat_bin(ph_name)
#        if use_amplitude:
#            d = list_directory_amp[k+1]
#            amp_name = welldir+"\\Intensity\\Float\\Bin\\"+d
#            amp, header_amp = binkoala.read_mat_bin(amp_name)
#            w = amp*np.exp(complex(0.,1.)*ph)
#        else:
#            w = np.exp(complex(0.,1.)*ph)
#        if background is not None and not use_background_for_superposition:
#            w /= background
#        if apply_avg_shift: #if correction of xy shift of images (pixel precision)
#            a = np.copy(avg_cplx)
#            b = np.copy(w)
#            if background is not None and use_background_for_superposition:
#                a /= background
#                b /= background
#            shift_measured, error, diffphase = register_translation(a, b,10)
#            shiftVector = (shift_measured[0],shift_measured[1])
#            #interpolation to apply the computed shift (has to be performed on float array)
#            real = ndimage.interpolation.shift(np.real(w), shift=shiftVector, mode='wrap')
#            imaginary = ndimage.interpolation.shift(np.imag(w), shift=shiftVector, mode='wrap')
#            w=real+complex(0.,1.)*imaginary
#        #perform complex averaging
#        z= np.angle(np.multiply(w,np.conj(avg_cplx))) #phase differenc between actual phase and avg_cplx phase
#        #measure offset using the mode of the histogram, instead of mean,better for noisy images (rough sample)
#        hist = np.histogram(z,bins=1000,range=(np.min(z),np.max(z)))
#        index = np.argmax(hist[0])
#        offset_value = hist[1][index]
#    
#        s = np.std(z)
#        signmap = (z < 0).astype(float)*2.*np.pi
#        z2 = z + signmap
#        s2 = np.std(z2)
#        if (s2 < s - 0.05):
#            hist = np.histogram(z2,bins=1000, range=(np.min(z),np.max(z)))
#            index = np.argmax(hist[0])
#            offset_value = hist[1][index]
#        w *= np.exp(-offset_value*complex(0.,1))#compensate the offset for the new wavefront
#        avg_cplx += w #add the new wavefront
#    avg_cplx /= len(list_directory_ph )
#    return avg_cplx, header

def complex_avg(thedir,iteration, background, use_amplitude, apply_avg_shift,use_background_for_superposition):
    
    list_directory = [ name for name in os.listdir(thedir) if os.path.isdir(os.path.join(thedir, name)) ]
    ph_name = thedir+"\\"+list_directory[0]+"\\Phase\\Float\\Bin\\"+iteration+"_phase.bin"
    ph_Koala, header = binkoala.read_mat_bin(ph_name)  #ph is the phase image
    
    ## Relevel all images with a plane before averaging. This removes most errors with missalignment due to DHM errors
    ## Stolen from https://stackoverflow.com/questions/35005386/fitting-a-plane-to-a-2d-array     
    m = header['height'][0] #size of the matrix
    X1, X2 = np.mgrid[:m, :m]
    
    #Regression
    X = np.hstack(   ( np.reshape(X1, (m*m, 1)) , np.reshape(X2, (m*m, 1)) ) )
    X = np.hstack(   ( np.ones((m*m, 1)) , X ))
    YY = np.reshape(ph_Koala, (m*m, 1))

    theta = np.dot(np.dot( np.linalg.pinv(np.dot(X.transpose(), X)), X.transpose()), YY)
    plane = np.reshape(np.dot(X, theta), (m, m));
    
    ph = ph_Koala - plane
    
    if use_amplitude:
        amp_name = thedir+"\\"+list_directory[0]+"\\Intensity\\Float\\Bin\\00000_intensity.bin"
        use_amplitude = os.path.isfile(amp_name)
        if not use_amplitude:
            print("Not possible to use amplitude, data do not exist")
    if use_amplitude:
        amp, header_amp = binkoala.read_mat_bin(amp_name)
        avg_cplx = amp*np.exp(complex(0.,1.)*ph)
    else:
        avg_cplx = np.exp(complex(0.,1.)*ph)
    if background is not None and not use_background_for_superposition:
        avg_cplx /= background
    for d in list_directory[0:]:
        ph_name = thedir+"\\"+d+"\\Phase\\Float\\Bin\\"+iteration+"_phase.bin"
        #ph, header = binkoala.read_mat_bin(ph_name)
        
        ph_Koala, header = binkoala.read_mat_bin(ph_name)  #ph is the phase image
        

        ## Stolen from https://stackoverflow.com/questions/35005386/fitting-a-plane-to-a-2d-array     
        m = header['height'][0] #size of the matrix
        X1, X2 = np.mgrid[:m, :m]
        
        #Regression
        X = np.hstack(   ( np.reshape(X1, (m*m, 1)) , np.reshape(X2, (m*m, 1)) ) )
        X = np.hstack(   ( np.ones((m*m, 1)) , X ))
        YY = np.reshape(ph_Koala, (m*m, 1))
    
        theta = np.dot(np.dot( np.linalg.pinv(np.dot(X.transpose(), X)), X.transpose()), YY)
        plane = np.reshape(np.dot(X, theta), (m, m));
        
        ph = ph_Koala - plane
        
        
        if use_amplitude:
            amp_name = thedir+"\\"+d+"\\Intensity\\Float\\Bin\\00000_intensity.bin"
            amp, header_amp = binkoala.read_mat_bin(amp_name)
            w = amp*np.exp(complex(0.,1.)*ph)
        else:
            w = np.exp(complex(0.,1.)*ph)
        if background is not None and not use_background_for_superposition:
            w /= background
        if apply_avg_shift: #if correction of xy shift of images (pixel precision)
            # plt.figure()
            # plt.imshow(np.angle(avg_cplx))
            # plt.title("avg_cplx_"+ d)
            # plt.figure()
            # plt.imshow(np.angle(w))
            # plt.title("w_"+ d)
            a = np.copy(avg_cplx)
            b = np.copy(w)
            if background is not None and use_background_for_superposition:
                a /= background
                b /= background
            ##shift_measured, error, diffphase = register_translation(np.angle(a), np.angle(b),10) #old code, depreciated
            shift_measured, error, diffphase = phase_cross_correlation(np.angle(a), np.angle(b),upsample_factor=10)

            
            shiftVector = (shift_measured[0],shift_measured[1])
            # print(shift_measured, error, diffphase)
            #interpolation to apply the computed shift (has to be performed on float array)
            real = ndimage.interpolation.shift(np.real(w), shift=shiftVector, mode='wrap')
            imaginary = ndimage.interpolation.shift(np.imag(w), shift=shiftVector, mode='wrap')
            w=real+complex(0.,1.)*imaginary
        #perform complex averaging
        z= np.angle(np.multiply(w,np.conj(avg_cplx))) #phase differenc between actual phase and avg_cplx phase
        #measure offset using the mode of the histogram, instead of mean,better for noisy images (rough sample)
        hist = np.histogram(z,bins=1000,range=(np.min(z),np.max(z)))
        index = np.argmax(hist[0])
        offset_value = hist[1][index]
    
        s = np.std(z)
        signmap = (z < 0).astype(float)*2.*np.pi
        z2 = z + signmap
        s2 = np.std(z2)
        if (s2 < s - 0.05):
            print('test is here')
            hist = np.histogram(z2,bins=1000, range=(np.min(z2),np.max(z2)))
            index = np.argmax(hist[0])
            offset_value = hist[1][index]
        w *= np.exp(-offset_value*complex(0.,1.))#compensate the offset for the new wavefront
        avg_cplx += w #add the new wavefront
    avg_cplx /= len(list_directory)
    return avg_cplx, header