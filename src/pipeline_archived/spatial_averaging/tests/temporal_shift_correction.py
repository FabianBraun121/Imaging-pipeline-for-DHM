# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 16:56:24 2023

@author: SWW-Bc20
"""
import os
os.chdir(r'C:\Users\SWW-Bc20\Documents\GitHub\Imaging-pipeline-for-DHM\src\spatial_averaging')
import numpy as np
import binkoala
os.chdir(r'C:\Users\SWW-Bc20\Documents\GitHub\Imaging-pipeline-for-DHM\src\spatial_averaging\tests')
from test_utilities import interactive_image_player
from skimage.registration import phase_cross_correlation
from scipy import ndimage

image_folder = r'F:\C11_20230217\2023-02-17 11-13-34 phase averages\00001'
images_names = os.listdir(image_folder)

images = np.zeros((len(images_names), 800,800))
shift_vector_list = [(0,0)]
error_list = [0]
diffphase_list = [0]

for i in range(len(images_names)):
    ph, __ = binkoala.read_mat_bin(image_folder + os.sep + images_names[i])
    images[i] = ph

for i in range(1, len(images_names)):
    shift_measured, error, diffphase = phase_cross_correlation(images[i-1], images[i], upsample_factor=10, normalization=None)
    shift_vector = (shift_measured[0],shift_measured[1])
    images[i] = ndimage.shift(images[i], shift=shift_vector, mode='wrap')
    shift_vector_list.append(shift_vector)
    error_list.append(error)
    diffphase_list.append(diffphase)

#%%
titles = [f'shift vecotr:{shift_vector_list[i]}, error:{error_list[i]}, diffphase:{diffphase_list[i]}' for i in range(len(images_names))]
interactive_image_player(images, titles=titles)
#interactive_image_player(images)