# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 15:05:12 2023

@author: SWW-Bc20
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from skimage.registration import phase_cross_correlation

#%%
def subtract_bacterias(cplx_image):
    # subtracts pixel  that are far away from the mean and replaces them with the mean of the image
    # cut off value is determined by hand and has to be reestimated for different use cases
    cut_off = 0.15
    ph = np.angle(cplx_image)
    ph[cut_off<ph] = np.mean(ph[ph<cut_off])
    return np.absolute(cplx_image)*np.exp(1j*ph)

def subtract_phase_offset(new, avg):
    z= np.angle(np.multiply(new,np.conj(avg))) #phase differenc between actual phase and avg_cplx phase
    #measure offset using the mode of the histogram, instead of mean,better for noisy images (rough sample)
    hist = np.histogram(z,bins=1000,range=(np.min(z),np.max(z)))
    index = np.argmax(hist[0])
    offset_value = hist[1][index]
    new *= np.exp(-offset_value*complex(0.,1.))#compensate the offset for the new wavefront
    return new

def shift_image(reference_image, moving_image):
    shift_measured, error, diffphase = phase_cross_correlation(np.angle(reference_image), np.angle(moving_image), upsample_factor=10, normalization=None)
    shiftVector = (shift_measured[0],shift_measured[1])
    #interpolation to apply the computed shift (has to be performed on float array)
    real = ndimage.shift(np.real(moving_image), shift=shiftVector, mode='wrap')
    imaginary = ndimage.shift(np.imag(moving_image), shift=shiftVector, mode='wrap')
    return real+complex(0.,1.)*imaginary

def mean_background(images_in):
    images = images_in.copy()
    background = images[0]
    for i in range(1, images.shape[0]):
        image = subtract_phase_offset(images[i], background)
        background += image
    return background/images.shape[0]

def mean_background_without_bacterias(images_in):
    images = images_in.copy()
    background = subtract_bacterias(images[0])
    for i in range(1, images.shape[0]):
        image = subtract_bacterias(subtract_phase_offset(images[i], background))
        background += image
    return background/images.shape[0]

def median_background(images_in):
    images = images_in.copy()
    return np.median(np.abs(images), axis=0)*np.exp(1j*np.median(np.angle(images), axis=0))

def confidence_background(images_in, conf_decay):
    images = images_in.copy()
    confidence = np.exp(-conf_decay*np.angle(images))
    return np.sum(images*confidence, axis=0)/np.sum(confidence, axis=0)

def cplx_avg(images_in, background):
    images = images_in.copy()
    avg = images[0]
    avg /= background
    for i in range(1, images.shape[0]):
        image = images[i]
        image /= background
        image = shift_image(avg, image)
        image = subtract_phase_offset(image, avg)
        avg += image
    return avg/images.shape[0]

#%%
images = np.load(r'C:\Users\SWW-Bc20\Documents\GitHub\Imaging-pipeline-for-DHM\tests\spatial_averaging\background_calculation\images.npy')
#%%
bg = np.mean(np.array([np.angle(median_background(images[i])) for i in range(8)]), axis=0)
#%%
i = 8
plt.figure('different backgrounds')
a = mean_background(images[i])
b = mean_background_without_bacterias(images[i])
ab  = np.hstack((np.angle(a),np.angle(b)))
c = median_background(images[i])
d = confidence_background(images[i], 5)
cd = np.hstack((np.angle(c),np.angle(d)))
plt.imshow(np.vstack((ab,cd)))

plt.figure('image averages')
a = cplx_avg(images[i], a)
b = cplx_avg(images[i], b)
ab  = np.hstack((np.angle(a),np.angle(b)))
c = cplx_avg(images[i], c)
d = cplx_avg(images[i], d)
cd = np.hstack((np.angle(c),np.angle(d)))
plt.imshow(np.vstack((ab,cd)))

#%%
plt.figure('a')
plt.imshow(np.angle(median_background(images[1]))-np.angle(median_background(images[0])))
