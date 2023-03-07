# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 13:17:19 2021

@author: Dieter A. Baumgartner

Based on Average_displacement_scanning_ by 
@author: tcolomb

"""

import os
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import math

from PyQt5.QtWidgets import QFileDialog

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

def complex_avg(thedir,iteration, background, use_amplitude, apply_avg_shift,use_background_for_superposition):
    
    list_directory = [ name for name in os.listdir(thedir) if os.path.isdir(os.path.join(thedir, name)) ]
    ph_name = thedir+"\\"+list_directory[0]+"\\Phase\\Float\\Bin\\"+iteration+"_phase.bin"
    
#    print(ph_name)

    ph_Koala, header = binkoala.read_mat_bin(ph_name)  #ph is the phase image
    
#    print(thedir)
#    print(list_directory)
    
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
        amp_name = thedir+"\\"+list_directory[0]+"\\Intensity\\Float\\Bin\\" +iteration + "_intensity.bin"
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
        
#        print(ph_name)
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
        
    
#        plt.figure()
#        plt.imshow(ph)
#        plt.title('ph image ' + d)
    
        if use_amplitude:
            amp_name = thedir+"\\"+d+"\\Intensity\\Float\\Bin\\" +iteration+ "_intensity.bin"
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
              
            # print(a)
            # print(b)
            ##shift_measured, error, diffphase = register_translation(np.angle(a), np.angle(b),10) #old code, depreciated
            shift_measured, error, diffphase = phase_cross_correlation(np.angle(a), np.angle(b),upsample_factor=10)

            
            shiftVector = (shift_measured[0],shift_measured[1])
            # print(shift_measured, error, diffphase)
            #interpolation to apply the computed shift (has to be performed on float array)
            real = ndimage.shift(np.real(w), shift=shiftVector, mode='wrap')
            imaginary = ndimage.shift(np.imag(w), shift=shiftVector, mode='wrap')
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


#%%


#output_in_nm = pd.DataFrame(columns = ("image_name", "data"))

for pos in range(max_pos): ## max_pos
    
    current_dir = thedir_base+"\\"+list_directory_base[pos]  
    print(current_dir)

    for i in range(timesteps): #max is timesteps
    
        timestep = i
    
        iteration = str(timestep).zfill(5)
    
#        print(iteration)
    
        background, header = complex_avg(current_dir, iteration, None, use_amplitude, apply_avg_shift=False,use_background_for_superposition=False)
        
#        plt.figure(pos+100)
#        plt.title("Background")
#        plt.imshow(np.angle(background))
##        
        avg_cplx, header = complex_avg(current_dir, iteration, background, use_amplitude, apply_avg_shift,use_background_for_superposition)
        
        plt.figure(pos)
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
        print(fname)
#        
#        img_nm = np.angle(avg_cplx)*794/(2*math.pi)
#        
#        df = pd.DataFrame([[list_directory[pos], img_nm]], columns = ("image_name", "data"))
#        
#        output_in_nm = output_in_nm.append(df, ignore_index=True)
#        
#
#%%
#        
#output_in_nm.to_pickle(thedir + "\data_in_nm.pkl")  
#        
#        
##%%
# 
#thedir = utils.Open_Directory(default_dir, "Open a scanning directory")
##plt.close("all")
#       
#import pandas as pd
#from matplotlib import pyplot as plt
#
#list_directory = [name for name in os.listdir(thedir) if os.path.isfile(os.path.join(thedir, name)) ]
#
#df = pd.read_pickle(thedir + "\\" + list_directory[3])
#
#for jj in range(len(df)):
#    
#    plt.figure(dpi = 200)
#    plt.title(df["image_name"][jj])
#    plt.imshow(df["data"][jj])
#
#
#%%
#
#
#Neuron13 = run1.loc[[13]]
#
#Neuron13 = Neuron13.append(run2.loc[[4]])
#Neuron13 = Neuron13.append(run3.loc[[4]])
#Neuron13 = Neuron13.append(run4.loc[[3]])
#
#Neuron13 = Neuron13.sort_values(by = ['image_name'], ascending = True)
#
#Neuron13 = Neuron13.reset_index(drop = True)
#
#
#Neuron13.to_pickle(r'Z:\Francois\20220506\Neuron13.pkl')  