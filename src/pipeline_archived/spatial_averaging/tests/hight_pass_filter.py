# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 14:28:22 2023

@author: SWW-Bc20
"""
# https://github.com/tesfagabir/Digital-Image-Processing/blob/master/05-Image-Filtering-using-Python.ipynb

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

def computeIdealFilters(imge, F, D, Dos):
    """Computes Ideal Filtering for different cut off frequencies.
       It also calculates their running time.
    """

    #Lowpass filtered images
    gsLP = []
    
    #Highpass filtered images
    gsHP = []
    
    #Running Time
    IRunningTime = []
    
    for Do in Dos:
        
        startTime = time.time()
        #Compute Ideal Lowpass Filter (ILPF)
        H = computeIdealFiltering(D, Do, 0)
        
        IRunningTime.append((time.time() - startTime))

        #Compute the filtered image (result in space domain)
        gsLP.append(computeFilteredImage(H, F))

        #Compute Ideal Highpass Filter (IHPF)
        H = computeIdealFiltering(D, Do, 1)

        #Compute the filtered image (result in space domain)
        gsHP.append(computeFilteredImage(H, F))

    return gsLP, gsHP, IRunningTime

def computeButterworthFiltering(D, Do, n, mode=0):
    """Computes Ideal Filtering based on the cut off frequency (Do) and n.
    If mode=0, it compute Lowpass Filtering otherwise Highpass filtering    
    """
    
    H = np.zeros_like(D)
    D = D.astype(float)
    
    if mode==0:
        H = 1.0/(1.0 + ((D/Do)**(2*n)))
    else:
        H = 1.0/(1.0 + ((Do/D)**(2*n)))
    return H

def computeButterFilters(imge, F, D, Dos, ns):
    """Computes Butterworth Filtering for different cut off frequencies."""

    #Lowpass filtered images
    gsLP = []
    
    #Highpass filtered images
    gsHP = []
    
    #Running Time
    BRunningTime = []
    
    for i, Do in enumerate(Dos):
        
        startTime = time.time()
        #Compute Ideal Lowpass Filter (ILPF)
        H = computeButterworthFiltering(D, Do, ns[i], 0)
        
        BRunningTime.append((time.time() - startTime))

        #Compute the filtered image (result in space domain)
        gsLP.append(computeFilteredImage(H, F))

        #Compute Ideal Highpass Filter (IHPF)
        H = computeButterworthFiltering(D, Do, ns[i], 1)

        #Compute the filtered image (result in space domain)
        gsHP.append(computeFilteredImage(H, F))

    return gsLP, gsHP, BRunningTime

def computeFilteredImage(H, F):
    """Computes a filtered image based on the given fourier transormed image(F) and filter(H)."""
    G = H * F #Element-wise multiplication
    g = np.real(np.fft.ifft2(np.fft.ifftshift((G)))).astype(float)
    
    return g

def visualizeFilteringResults(imge, F, gsLP, gsHP, Dos, filterType="Ideal", ns=None):
    """Visualizes the filtered images using different cut-off frequencies."""

    fig, axarr = plt.subplots(2, 5, figsize=[10,5])

    axarr[0, 0].imshow(imge, cmap=plt.get_cmap('gray'), )
    axarr[0, 0].set_title("Original Image")
    axarr[0, 0].axes.get_xaxis().set_visible(False)
    axarr[0, 0].axes.get_yaxis().set_visible(False)

    axarr[1, 0].imshow(imge, cmap=plt.get_cmap('gray'))
    axarr[1, 0].set_title("Original Image")
    axarr[1, 0].axes.get_xaxis().set_visible(False)
    axarr[1, 0].axes.get_yaxis().set_visible(False)

    ##Display the filtering Results
    for i, g in enumerate(gsLP):

        if filterType=='Ideal':
            lp = "ILPF(Do="+str(Dos[i])+")"
            hp = "IHPF(Do="+str(Dos[i])+")"
        else:
            lp = "BLPF(Do="+str(Dos[i])+",n="+str(ns[i])+")"
            hp = "BHPF(Do="+str(Dos[i])+",n="+str(ns[i])+")"

        #For lowpass
        axarr[0, i+1].imshow(gsLP[i], cmap=plt.get_cmap('gray'))
        axarr[0, i+1].set_title(lp)
        axarr[0, i+1].axes.get_xaxis().set_visible(False)
        axarr[0, i+1].axes.get_yaxis().set_visible(False)    

        #For highpass
        axarr[1, i+1].imshow(gsHP[i], cmap=plt.get_cmap('gray'))
        axarr[1, i+1].set_title(hp)
        axarr[1, i+1].axes.get_xaxis().set_visible(False)
        axarr[1, i+1].axes.get_yaxis().set_visible(False)    

    plt.show()
    


#%% 
fname = r'F:\C11_20230217\2023-02-17 11-13-34 phase averages\00002\timestep_00072.bin'
#Read an image from a file
image, header = binkoala.read_mat_bin(fname)
N = image.shape[0]

#Construct the DFT of the image
F = np.fft.fftshift(np.fft.fft2(image))

#Construct the D(u,v) matrix
D = constructDuv(N) 
#%%
Dos = np.array([550, 552, 558, 565]) #Cut off frequencies

#Ideal Filtering
gsILP, gsIHP, IRunningTime = computeIdealFilters(image, F, D, Dos)

#Butterworth Filtering
ns = np.ones(Dos.shape[0])*2
gsBLP, gsBHP, BRunningTime = computeButterFilters(image, F, D, Dos, ns)
#%%
print('Ideal Filtering')
visualizeFilteringResults(image, F, gsILP, gsIHP, Dos, 'Ideal')
#%%
#Plot the running times

fig = plt.figure(figsize=[4,4])
numDo = len(IRunningTime)

plt.plot(range(numDo), IRunningTime, '-d')
plt.plot(range(numDo), BRunningTime, '-d')

xlabels = [str(int(Do)) for Do in Dos]
plt.xticks(range(numDo), xlabels)
plt.xlabel("Do")
plt.ylabel("Time(Sec)")
plt.legend(['Ideal', 'Butterworth'], loc=0)
plt.show()
#%%


Dos = np.ones(4, dtype=int)*100 #Constant Cut-off frequency
ns = [1, 2, 5, 10]

gsBLP, gsBHP, _ = computeButterFilters(image, F, D, Dos, ns)

print('Butterworth Filtering')
visualizeFilteringResults(image, F, gsBLP, gsBHP, Dos, 'Butterworth', ns)
