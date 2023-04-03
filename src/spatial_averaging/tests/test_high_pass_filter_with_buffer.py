# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 09:57:48 2023

@author: SWW-Bc20
"""
import os
os.chdir(r'C:\Users\SWW-Bc20\Documents\GitHub\Imaging-pipeline-for-DHM\src\spatial_averaging')
import binkoala

import numpy as np
import matplotlib.pyplot as plt

#import matplotlib.image as img
#import PIL.Image as Image 
import time

#%%
def constructDuv(N):
    """Constructs the frequency matrix, D(u,v), of size NxN"""
    u = np.arange(N)
    v = np.arange(N)

    idx = np.where(u>N/2)[0]
    u[idx] = u[idx] - N
    idy = np.where(v>N/2)[0]
    v[idy] = v[idx] - N

    [V,U]= np.meshgrid(v,u)
    D = np.sqrt(U**2 + V**2)
    
    return D

def constructD(N, radius):
    Dr = np.arange(-N/2,N/2)**2
    D = np.meshgrid(Dr, Dr)
    H = (D[0]+D[1]<=radius**2).astype(int)
    return H
    

def computeIdealFiltering(D, Do, mode=0):
    """Computes Ideal Filtering based on the cut off frequency (Do).
    If mode=0, it compute Lowpass Filtering otherwise Highpass filtering    
    """
    
    H = np.zeros_like(D)
    if mode==0:
        H = (D<=Do).astype(int)
    else:
        H = (D>Do).astype(int)
    return H

def computeFilteredImage(H, F):
    """Computes a filtered image based on the given fourier transormed image(F) and filter(H)."""
    G = H * F #Element-wise multiplication
    g = np.real(np.fft.ifft2(np.fft.ifftshift((G)))).astype(float)
    return g

fname = r'F:\C11_20230217\2023-02-17 11-13-34 phase averages\00002\timestep_00072.bin'
#Read an image from a file
image, header = binkoala.read_mat_bin(fname)
#%%
N = image.shape[0]
buffers = 10*np.arange(10)

fig, ax = plt.subplots(2,5)
for i,b in enumerate(buffers):
    Ni = N+b
    imagei = np.zeros((image.shape[0]+b, image.shape[1]+b))
    imagei[:image.shape[0], :image.shape[1]] = image
    D = constructD(Ni, 100)
    F = np.fft.fftshift(np.fft.fft2(imagei))
    g = computeFilteredImage(D, F)
    ax[i//5,i%5].imshow(g[:image.shape[0], :image.shape[1]])


