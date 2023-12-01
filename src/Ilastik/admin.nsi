# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 17:20:48 2023

@author: SWW-Bc20
"""
import os
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
import cv2
np.seterr(divide = 'ignore')

base_dir = r'D:\data\Data_for_Delta'
save_base_folder = r'C:\Users\SWW-Bc20\Documents\GitHub\Imaging-pipeline-for-DHM\data\delta_assets\trainingsets\2D\training\bf_segmentation_set'
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


def create_weights(seg):
    inside = distance_transform_edt(seg)
    inside = np.where(inside != 0, 1 / inside, 0.0)
    outside = distance_transform_edt(np.where(seg == 0, 1, 0))
    outside = np.where(outside != 0, 1 / outside, 0.0)
    weights = np.where(outside==0, inside, outside)
    weights = (np.where(weights<0.05, 0.05, weights)*255).astype(np.uint8)
    return weights
    

timeseries_bf_paths = [base_dir + os.sep + i for i in sorted(os.listdir(base_dir)) if i.startswith('BF_stack')]
timeseries_seg_paths = [base_dir + os.sep + i for i in sorted(os.listdir(base_dir)) if i.startswith('BF_segmented')]

for i in range(len(timeseries_bf_paths)):
    timeseries_bf = tifffile.imread(timeseries_bf_paths[i])
    timeseries_seg = tifffile.imread(timeseries_seg_paths[i])
    
    for j in range(timeseries_bf.shape[0]):
        save_name = f'{(timeseries_bf_paths[i][-8:-4]).zfill(5)}_{(str(j)).zfill(5)}.png'
        bf = timeseries_bf[j]
        seg = timeseries_seg[j]
        weights = create_weights(seg)
        
        bf = (65535*((bf - bf.min())/bf.ptp())).astype(np.uint16)
        cv2.imwrite(save_img_folder + os.sep + save_name, bf)
        cv2.imwrite(save_seg_folder + os.sep + save_name, (seg*255).astype(np.uint8))
        cv2.imwrite(save_wei_folder + os.sep + save_name, weights)
