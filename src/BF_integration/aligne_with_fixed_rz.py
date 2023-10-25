# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 11:55:35 2023

@author: SWW-Bc20
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

def gradient_squared(image):
    grad_x = ndimage.sobel(image, axis=0)
    grad_y = ndimage.sobel(image, axis=1)
    return (grad_x**2+grad_y**2)
    

base_path = r'C:\Users\SWW-Bc20\Documents\GitHub\Imaging-pipeline-for-DHM\data\brightfield\20230905-1643'
save_base_path = base_path + os.sep + 'aligned_images_fixed_zr'
if not os.path.exists(save_base_path):
    os.makedirs(save_base_path)
bf_base_path = base_path + os.sep + '20230905-1643 BF'
ph_base_path = base_path + os.sep + '20230905-1643 phase averages'
postitions = os.listdir(bf_base_path)
for postition in postitions:
    bf_positions_path = bf_base_path + os.sep + postition
    ph_positions_path = ph_base_path + os.sep + postition
    timesteps = [int(''.join(filter(str.isdigit, s))) for s in os.listdir(ph_positions_path)]
    save_folder_path = save_base_path + os.sep + postition
    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)
        
    for timestep in timesteps:
        start = time.time()
        bf_fname = bf_positions_path + os.sep + f'{str(timestep).zfill(5)}_BF.tif'
        ph_fname = ph_positions_path + os.sep + f'ph_timestep_{str(timestep).zfill(5)}.bin'
        
        bf = tifffile.imread(bf_fname)[512:1536,512:1536]
        bf = np.fliplr(bf)
        bf = trans.rotate(bf, -90, mode="edge")
        ph, _ = binkoala.read_mat_bin(ph_fname)
        ph_ = np.zeros(bf.shape)
        ph_[:ph.shape[0], :ph.shape[1]] = ph
        
        bf_ = gradient_squared(bf)
        ph_ = gradient_squared(ph_)
        
        bf_rz = zoom(trans.rotate(bf_, -0.3, mode="edge"), 0.907)
        shift_measured = phase_cross_correlation(ph_, bf_rz, upsample_factor=10)[0]
        shift_vector = (shift_measured[0], shift_measured[1])
        
        bf_out = ndimage.shift(zoom(trans.rotate(bf, -0.3, mode="edge"),0.907), shift_vector)[:ph.shape[0], :ph.shape[1]]
        fname = save_folder_path + os.sep + f'{str(timestep).zfill(5)}_BF.tif'
        tifffile.imwrite(fname, bf_out)
        
        print(f'image p:{postition} t:{timestep} done in {np.round(time.time()-start,1)} seconds')
        