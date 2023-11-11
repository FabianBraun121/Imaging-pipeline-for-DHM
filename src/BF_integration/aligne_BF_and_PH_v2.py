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
    I = trans.rescale(I, zoomlevel, mode="edge", preserve_range=True)
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
    shift_measured, error, phasediff = phase_cross_correlation(image1, im, upsample_factor=10)
    return error

def grid_search(image1, image2, x_mid, y_mid, x_length, y_length):
    # Initialize the initial grid boundaries.
    x_start, x_end = x_mid - x_length/2, x_mid + x_length/2
    y_start, y_end = y_mid - y_length/2, y_mid + y_length/2
    
    count = 0
    while count in range(4):
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

base_path = r'D:\data\brightfield\20230905-1643'
save_base_path = base_path + os.sep + 'aligned_images'
if not os.path.exists(save_base_path):
    os.makedirs(save_base_path)
bf_base_path = base_path + os.sep + '20230905-1643 BF'
ph_base_path = base_path + os.sep + '20230905-1643 phase averages'
postitions = os.listdir(bf_base_path)
for postition in postitions[:10]:
    bf_positions_path = bf_base_path + os.sep + postition
    ph_positions_path = ph_base_path + os.sep + postition
    timesteps = [int(''.join(filter(str.isdigit, s))) for s in os.listdir(ph_positions_path)]
    save_folder_path = save_base_path + os.sep + postition
    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)
        
    rots = []
    zooms = []
    shift_vectors = []
    for timestep in timesteps[:30]:
        start = time.time()
        bf_fname = bf_positions_path + os.sep + f'{str(timestep).zfill(5)}_BF.tif'
        ph_fname = ph_positions_path + os.sep + f'ph_timestep_{str(timestep).zfill(5)}.bin'
        
        bf = tifffile.imread(bf_fname)[512:1536,512:1536]
        bf = np.fliplr(bf)
        bf = trans.rotate(bf, -90, mode="edge", preserve_range=True)
        ph, _ = binkoala.read_mat_bin(ph_fname)
        ph_ = np.zeros(bf.shape)
        ph_[:ph.shape[0], :ph.shape[1]] = ph
        
        bf_ = gradient_squared(bf)
        ph_ = gradient_squared(ph_)
        rot, zoomlevel = grid_search(ph_, bf_, 0, 0.9, 0.5, 0.1)
        bf_rz = zoom(trans.rotate(bf_, rot, mode="edge"),zoomlevel)
        shift_measured = phase_cross_correlation(ph_, bf_rz, upsample_factor=10)[0]
        shift_vector = (shift_measured[0], shift_measured[1])
        
        bf_out = ndimage.shift(zoom(trans.rotate(bf, rot, mode="edge", preserve_range=True),zoomlevel), shift_vector)[:ph.shape[0], :ph.shape[1]]
        fname = save_folder_path + os.sep + f'{str(timestep).zfill(5)}_BF.tif'
        tifffile.imwrite(fname, bf_out)
        
        rots.append(rot)
        zooms.append(zoomlevel)
        shift_vectors.append(shift_vector)
        
        print(f'image p:{postition} t:{timestep} done in {np.round(time.time()-start,1)} seconds')
        
    np.save(save_folder_path + os.sep + 'rotations', rots)
    np.save(save_folder_path + os.sep + 'zooms', zooms)
    np.save(save_folder_path + os.sep + 'shift_vectors', shift_vectors)