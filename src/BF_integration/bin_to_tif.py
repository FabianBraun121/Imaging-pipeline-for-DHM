# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 11:15:47 2023

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

base_path = r'C:\Users\SWW-Bc20\Documents\GitHub\Imaging-pipeline-for-DHM\data\brightfield\20230905-1643'
ph_base_path = base_path + os.sep + '20230905-1643 phase averages'
ph_save_path = base_path + os.sep + '20230905-1643 phase averages tif'
if not os.path.exists(ph_save_path):
    os.makedirs(ph_save_path)
    
postitions = [f.name for f in os.scandir(ph_base_path) if f.is_dir()]
for postition in postitions:
    ph_pos_path = ph_base_path + os.sep + postition
    timesteps = [int(''.join(filter(str.isdigit, s))) for s in os.listdir(ph_pos_path)]
    
    ph_pos_save_path = ph_save_path + os.sep + postition
    if not os.path.exists(ph_pos_save_path):
        os.makedirs(ph_pos_save_path)
    for timestep in timesteps:
        ph_fname = ph_pos_path + os.sep + f'ph_timestep_{str(timestep).zfill(5)}.bin'
        ph, _ = binkoala.read_mat_bin(ph_fname)
        ph_fname_out = ph_pos_save_path + os.sep + f'{str(timestep).zfill(5)}_PH.tif'
        tifffile.imwrite(ph_fname_out, ph)