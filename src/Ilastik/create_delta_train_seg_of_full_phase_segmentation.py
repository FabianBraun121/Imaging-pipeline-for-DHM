# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 11:20:38 2023

@author: SWW-Bc20
"""
import os
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
import cv2
np.seterr(divide = 'ignore')

base_dir = r'D:\data\brightfield\20230905-1643\20230905-1643 phase averages tif'
save_base_folder = r'C:\Users\SWW-Bc20\Documents\GitHub\Imaging-pipeline-for-DHM\data\delta_assets\trainingsets\2D\training\full_ph_segmentation'
if not os.path.exists(save_base_folder):
    os.mkdir(save_base_folder)
save_img_folder = save_base_folder + os.sep + 'img'
if not os.path.exists(save_img_folder):
    os.mkdir(save_img_folder)
save_seg_folder = save_base_folder + os.sep + 'seg'
if not os.path.exists(save_seg_folder):
    os.mkdir(save_seg_folder)
save_wei_folder = save_base_folder + os.sep + 'wei'
if not os.path.exists(save_wei_folder):
    os.mkdir(save_wei_folder)

positions = os.listdir(base_dir)
for position in positions:
    pos_dir = base_dir + os.sep + position
    seg_images = [i for i in os.listdir(pos_dir) if i.endswith('Identities.tif')]
    
    for seg_image in seg_images:
        timestemp = int(seg_image[:5])
        mask_fname = pos_dir + os.sep + seg_image
        ph_image_fname = pos_dir + os.sep + f'{str(timestemp).zfill(5)}_PH.tif'
        ph_image = tifffile.imread(ph_image_fname)
        mask = tifffile.imread(mask_fname)
        
        mask = np.where(mask!=0,1,0)
        inside = distance_transform_edt(mask)
        inside = np.where(inside != 0, 1 / inside, 0.0)
        outside = distance_transform_edt(np.where(mask == 0, 1, 0))
        outside = np.where(outside != 0, 1 / outside, 0.0)
        weights = np.where(outside==0, inside, outside)
        weights = (np.where(weights<0.05, 0.05, weights)*255).astype(np.uint8)
        
        save_name = f'20230905-1643_{position}_{str(timestemp).zfill(5)}.png'
        ph_scaled = (((ph_image + np.pi/2) / np.pi) * 65535).astype(np.uint16)
        cv2.imwrite(save_img_folder + os.sep + save_name, ph_scaled)
        cv2.imwrite(save_seg_folder + os.sep + save_name, (mask*255).astype(np.uint8))
        cv2.imwrite(save_wei_folder + os.sep + save_name, weights)
        
        