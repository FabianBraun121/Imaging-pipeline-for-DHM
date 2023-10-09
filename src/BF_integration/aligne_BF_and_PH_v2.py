# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 16:54:12 2023

@author: fabia
"""

import os
os.chdir(os.path.dirname(__file__))
import binkoala
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import skimage.transform as trans
from skimage.registration import phase_cross_correlation
import time
from scipy import ndimage


def zoom(I, zoomlevel):
    oldshape = I.shape
    I_zoomed = np.zeros_like(I)
    I = trans.rescale(I, zoomlevel, mode="edge")
    if zoomlevel<1:
        i0 = (
            round(oldshape[0]/2 - I.shape[0]/2),
            round(oldshape[1]/2 - I.shape[1]/2),
        )
        I_zoomed[i0[0]:i0[0]+I.shape[0], i0[1]:i0[1]+I.shape[1]] = I
        return I_zoomed
    else:
        I = trans.rescale(I, zoomlevel, mode="edge")
        i0 = (
            round(I.shape[0] / 2 - oldshape[0] / 2),
            round(I.shape[1] / 2 - oldshape[1] / 2),
        )
        I = I[i0[0] : (i0[0] + oldshape[0]), i0[1] : (i0[1] + oldshape[1])]
        return I

def evaluate_error(image1, image2, rotation, zoomlevel):
    im = trans.rotate(image2, rotation, mode="edge")
    im = zoom(im, zoomlevel)
    shift_measured, error, phasediff = phase_cross_correlation(image1, im, upsample_factor=10, normalization=None)
    return error

def grid_search(image1, image2, x_mid, y_mid, x_length, y_length):
    # Initialize the initial grid boundaries.
    x_start, x_end = x_mid - x_length/2, x_mid + x_length/2
    y_start, y_end = y_mid - y_length/2, y_mid + y_length/2
    
    count = 0
    while count in range(3):
        # Create a grid based on the current boundaries.
        x_values = np.linspace(x_start, x_end, 5)
        y_values = np.linspace(y_start, y_end, 5)
        
        # Initialize variables to track the minimum and its location.
        min_value = float('inf')
        min_x, min_y = None, None
        
        # Evaluate the function at each point in the grid.
        for i,x in enumerate(x_values):
            for j,y in enumerate(y_values):
                if (i+j)%2==0:
                    value = evaluate_error(image1, image2, x, y)
                    if value < min_value:
                        min_value = value
                        min_x, min_y = x, y
        
        # Check if the minimum is at the edge or in the middle.
        if (
            min_x == x_start or min_x == x_end or
            min_y == y_start or min_y == y_end
        ):
            # If the minimum is at the edge, expand the search space.
            x_start, x_end = min_x - x_length/2, min_x + x_length/2
            y_start, y_end = min_y - y_length/2, min_y + y_length/2
        else:
            count += 1
            # If the minimum is in the middle, reduce the grid size.
            x_length /= 3
            y_length /= 3
            x_start, x_end = min_x - x_length/2, min_x + x_length/2
            y_start, y_end = min_y - y_length/2, min_y + y_length/2
    return min_x,min_y

def gradient_squared(image):
    grad_x = ndimage.sobel(image, axis=0)
    grad_y = ndimage.sobel(image, axis=1)
    return (grad_x**2+grad_y**2)

base_path = r'C:\Users\fabia\Documents\GitHub\Imaging-pipeline-for-DHM\data\E_coli_steady_state_100_pos'
save_base_path = r'C:\Users\fabia\Documents\GitHub\Imaging-pipeline-for-DHM\data\E_coli_steady_state_100_pos\Aligned_images\E_coli_1430'
bf_path = base_path + os.sep + ""
os.listdir(base_path)
image_nums = np.arange(1,101)
rots = []
zooms = []
shift_vectors = []
image_stack = []

for image_num in image_nums:
    start = time.time()
    bf_path = base_path + os.sep + 'E_coli_1430' + os.sep + f'{str(image_num).zfill(5)}' + os.sep + '00000_BF.tif'
    ph_path = base_path + os.sep + 'E_coli_1430 phase averages' + os.sep +  f'{str(image_num).zfill(5)}' + os.sep + 'ph_timestep_00000.bin'
    
    bf = tifffile.imread(bf_path)[512:1536,512:1536]
    bf = np.fliplr(bf)
    bf = trans.rotate(bf, -90, mode="edge")
    ph, _ = binkoala.read_mat_bin(ph_path)
    ph_ = np.zeros(bf.shape)
    ph_[:ph.shape[0], :ph.shape[1]] = ph
    
    bf_ = gradient_squared(bf)
    ph_ = gradient_squared(ph_)
    rot, zoomlevel = grid_search(ph_, bf_, 0, 0.9, 0.5, 0.1)
    bf_rz = zoom(trans.rotate(bf_, rot, mode="edge"),zoomlevel)
    shift_measured = phase_cross_correlation(ph_, bf_rz, upsample_factor=10, normalization=None)[0]
    shift_vector = (shift_measured[0], shift_measured[1])
    
    bf_out = ndimage.shift(zoom(trans.rotate(bf, rot, mode="edge"),zoomlevel), shift_vector)[:ph.shape[0], :ph.shape[1]]
    fname = save_folder + os.sep + f'{str(image_num).zfill(5)}_BF.tif'
    tifffile.imwrite(fname, bf_out)
    
    rots.append(rot)
    zooms.append(zoomlevel)
    shift_vectors.append(shift_vector)
    image_stack.append((ph-ph.min())/(ph.max()-ph.min()))
    image_stack.append((bf_out-bf_out.min())/(bf_out.max()-bf_out.min()))
    
    print(f'image {image_num} done in {time.time()-start} seconds')