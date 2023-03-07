# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 16:51:27 2023

@author: Sww-Bc20
"""


import os
import numpy as np
from matplotlib import pyplot as plt

from PyQt5.QtWidgets import QFileDialog

os.chdir('C:\Master_Thesis_Fabian_Braun\Code Dieter')

#import binkoala, utils
import ProcessClass, binkoala

def Open_Directory(directory, message):
    #print(directory)
    fname = QFileDialog.getExistingDirectory(None, message, directory, QFileDialog.ShowDirsOnly)
#    if python_vers == "3.x":
#        fname = fname[0]
    return fname

#Mode for the well: avg_by_well= True multiple data for a given well, avg_by_well=False: single data by directory "
avg_by_well = False
#apply_background only if the displacement between data is large
apply_background = True
#use_background_for_superposition, try using it if superposition fails (no effect if apply_background is applied)
use_background_for_superposition = False
#use_amplitude, if amplitude does not exist, message
use_amplitude = False
#apply_avg_shift in principle always True
apply_avg_shift = True
#polynomial degree of subtracted plane
plane_degree = 1

#Perform path following unwrap on phase image
unwrap = True

default_dir = r'Q:\SomethingFun' 


thedir_base = Open_Directory(default_dir, "Open a scanning directory")

plt.close("all")

#list_directory = [ name for name in os.listdir(thedir) if os.path.isdir(os.path.join(thedir, name)) ]
#firstdir = thedir+"\\"+list_directory[0]+"\\Phase\\Float\\Bin\\"


list_directory_base = [name for name in os.listdir(thedir_base) if os.path.isdir(os.path.join(thedir_base, name)) ]

## get directory of first position
first_pos_dir = thedir_base+"\\"+list_directory_base[0]
list_directory_perPos = [name for name in os.listdir(first_pos_dir) if os.path.isdir(os.path.join(first_pos_dir, name)) ]

## get directory of first sub point for the spatial averaging
first_sa_dir = first_pos_dir+"\\"+list_directory_perPos[0]+"\\Phase\\Float\\Bin\\"


max_pos = len(list_directory_base)
timesteps = len(os.listdir(first_sa_dir))

print(max_pos)
print(timesteps)


#%%
from scipy import ndimage
from skimage.registration import phase_cross_correlation
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def subtract_plane(field, plane_degree):
    X1, X2 = np.mgrid[:field.shape[0], :field.shape[1]]
    X = np.hstack((X1.reshape(-1,1) , X2.reshape(-1,1)))
    X = PolynomialFeatures(degree=plane_degree, include_bias=False).fit_transform(X)
    y = field.reshape(-1)
    reg = LinearRegression().fit(X, y)
    plane = reg.predict(X).reshape(field.shape[0],field.shape[1])
    return field - plane

def phase2complex(phase, use_amplitude, thedir, dir_base, iteration):
    if use_amplitude:
        amp_name = thedir+"\\"+dir_base+"\\Intensity\\Float\\Bin\\" +iteration+ "_intensity.bin"
        use_amplitude = os.path.isfile(amp_name)
        if not use_amplitude:
            print("Not possible to use amplitude, data do not exist")
    if use_amplitude:
        amp, header_amp = binkoala.read_mat_bin(amp_name)
        cplx = amp*np.exp(complex(0.,1.)*phase)
    else:
        cplx = np.exp(complex(0.,1.)*phase)
    return cplx

def shfit_image(reference_image, moving_image, background, use_background_for_superposition):
    a = np.copy(reference_image)
    b = np.copy(moving_image)
    if background is not None and use_background_for_superposition:
        a /= background
        b /= background

    shift_measured, error, diffphase = phase_cross_correlation(np.angle(a), np.angle(b), upsample_factor=10, normalization=None)
    
    shiftVector = (shift_measured[0],shift_measured[1])
    #interpolation to apply the computed shift (has to be performed on float array)
    real = ndimage.shift(np.real(moving_image), shift=shiftVector, mode='wrap')
    imaginary = ndimage.shift(np.imag(moving_image), shift=shiftVector, mode='wrap')
    return real+complex(0.,1.)*imaginary
    
def generate_complex_image(thedir, dir_base, iteration, background, plane_degree, use_amplitude, use_background_for_superposition):
    ph_name = thedir+"\\"+dir_base+"\\Phase\\Float\\Bin\\"+iteration+"_phase.bin"

    ph_Koala, header = binkoala.read_mat_bin(ph_name)  #ph is the phase image
    
    ph = subtract_plane(ph_Koala, plane_degree)
    
    cplx = phase2complex(ph, use_amplitude, thedir, dir_base, iteration)
    if background is not None and not use_background_for_superposition:
        cplx /= background
    return cplx, header

def subtract_phase_offset(new, avg):
    z= np.angle(np.multiply(new,np.conj(avg))) #phase differenc between actual phase and avg_cplx phase
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
    new *= np.exp(-offset_value*complex(0.,1.))#compensate the offset for the new wavefront
    return new

def complex_avg(thedir,iteration, background, use_amplitude, plane_degree, apply_avg_shift, use_background_for_superposition):
    list_directory = [ name for name in os.listdir(thedir) if os.path.isdir(os.path.join(thedir, name)) ]
    avg_cplx, header = generate_complex_image(thedir, list_directory[0], iteration, background, plane_degree, use_amplitude, use_background_for_superposition)
    for d in list_directory[0:]:
        w, header = generate_complex_image(thedir, d, iteration, background, plane_degree, use_amplitude, use_background_for_superposition)
        if apply_avg_shift: #if correction of xy shift of images (pixel precision)
            w = shfit_image(avg_cplx, w, background, use_background_for_superposition)
        w = subtract_phase_offset(w, avg_cplx)
        avg_cplx += w #add the new wavefront
    avg_cplx /= len(list_directory)
    return avg_cplx, header


#%%

#output_in_nm = pd.DataFrame(columns = ("image_name", "data"))

for pos in range(max_pos): ## max_pos
    
    current_dir = thedir_base+"\\"+list_directory_base[pos]  
    #print(current_dir)

    for i in range(timesteps): #max is timesteps
    
        timestep = i
    
        iteration = str(timestep).zfill(5)
    
#        print(iteration)
    
        background, header = complex_avg(current_dir, iteration, None, use_amplitude, plane_degree, apply_avg_shift=False,use_background_for_superposition=False)
        plt.figure(pos+100)
        plt.imshow(np.angle(background))
##        
        avg_cplx, header = complex_avg(current_dir, iteration, background, use_amplitude, plane_degree, apply_avg_shift,use_background_for_superposition)
        
        plt.figure(pos)
        plt.title("Background")
        plt.imshow(np.angle(avg_cplx))
        
        # fname = thedir +"\\avg_phase_"+list_directory[pos]+".bin" # use when bulk measurement, single timepoint
        fname = current_dir +"\\avg_phase_"+iteration+".bin" #use when timerseries
        width = (int)(header["width"])
        height = (int)(header["height"])
        px_size = (float)(header["px_size"])
        hconv = (float)(header["hconv"])
        unit_code = (int)(header["unit_code"])
        if unwrap:
            ph = ProcessClass.get_result_unwrap(np.angle(avg_cplx))
        else:
            ph = np.angle(avg_cplx)
        binkoala.write_mat_bin(fname, ph, width, height, px_size, hconv, unit_code)  
#        
#        img_nm = np.angle(avg_cplx)*794/(2*math.pi)
#        
#        df = pd.DataFrame([[list_directory[pos], img_nm]], columns = ("image_name", "data"))
#        
#        output_in_nm = output_in_nm.append(df, ignore_index=True)
#        
#
