# -*- coding: utf-8 -*-
"""
Created on Thu May 25 14:19:22 2023

@author: SWW-Bc20
"""
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from PIL import Image
import numpy as np
from skimage.morphology import binary_erosion, binary_dilation
import tifffile


def create_seg_and_weight_exponential_decay(mask):
    t = 0.5
    seg = np.zeros_like(mask)
    wei = np.zeros_like(mask)

    # Loop through each unique region value in the mask
    for region in np.unique(mask):
        # Skip the background region (value of 0)
        if region == 0:
            continue
        
        region_mask = (mask == region)
        erosed_mask = binary_erosion(region_mask)
        seg[erosed_mask] = 1.0
        boundary_mask = region_mask & ~erosed_mask
        wei[boundary_mask] = 1.0
        boundary_masks =[]
        while True:
            t = 0.5
            region_mask = erosed_mask
            erosed_mask = binary_erosion(region_mask)
            boundary_masks.append(region_mask & ~erosed_mask)
            if len(np.unique(erosed_mask)) == 1:
                break
        boundary_masks = boundary_masks[::-1]
        for i, shell in enumerate(boundary_masks):
            wei[shell] = int(255*np.exp(-t*i))
            
    return seg, wei


def create_seg_and_weight(mask):
    min_weight = 0.05
    bg_boundary_weight = 0.2
    region_weight = 0.3
    inter_region_boundary_weight = 1.0

    seg = np.zeros_like(mask)
    wei = np.full(mask.shape, min_weight)

    for region in np.unique(mask):
        if region == 0:
            continue

        region_mask = (mask == region)
        erosed_mask = binary_erosion(region_mask)
        seg[erosed_mask] = 1.0

        # Assign weights
        wei[region_mask] = region_weight
        boundary_mask = region_mask & ~erosed_mask
        wei[boundary_mask] = bg_boundary_weight
    
    for region in np.unique(mask):
        if region == 0:
            continue
        
        region_mask = (mask == region)
        dilated_mask = binary_dilation(region_mask)
        dilated_boundry = dilated_mask & ~region_mask
        inter_region = dilated_boundry & (mask != 0)
        wei[inter_region] = inter_region_boundary_weight

    return seg, wei

def rename_file(old_name, new_name):
    # Check if old_name file exists
    if os.path.isfile(old_name):
        # Rename the file
        os.rename(old_name, new_name)
    else:
        print(f"No file with the following name '{old_name}'")

#%%
mask_path_ = r'F:\Ilastik\F3_20230406\2023-04-06 11-07-24 phase averages\00001_identities'
mask_ = tifffile.imread(mask_path_ + os.sep + os.listdir(mask_path_)[0])
seg, wei = create_seg_and_weight(mask_)
#%%
# for j in range(1,6):
#     experiment = 'F3_20230406'
#     avg_folder = '2023-04-06 11-07-24 phase averages'
#     wall_int = j
#     wall_string = str(wall_int).zfill(5)
#     base_folder = r'F:\Ilastik' + os.sep + experiment + os.sep + avg_folder
#     mask_folder = base_folder + os.sep + wall_string + '_identities'
#     mask_list = os.listdir(mask_folder)
    
#     for current_image_index in range(len(mask_list)):
#         mask_path = os.path.join(mask_folder, mask_list[current_image_index])
#         mask = tifffile.imread(mask_path)
#         seg, wei = create_seg_and_weight(mask)
#         seg_folder = base_folder + os.sep + wall_string + '_seg'
#         if not os.path.exists(seg_folder):
#             os.makedirs(seg_folder)
#         wei_folder = base_folder + os.sep + wall_string + '_wei'
#         if not os.path.exists(wei_folder):
#             os.makedirs(wei_folder)
#         fname = experiment + '_' + wall_string + '_' + mask_list[current_image_index][:14] + '.tif'
#         im_seg = Image.fromarray(seg)
#         im_seg.save(seg_folder + os.sep + fname)
#         im_wei = Image.fromarray(wei)
#         im_wei.save(wei_folder + os.sep + fname)


# #%%

# #only renaming
# for j in range(1,6):
#     experiment = 'F3_20230406'
#     avg_folder = '2023-04-06 11-07-24 phase averages'
#     wall_int = j
#     wall_string = str(wall_int).zfill(5)
#     base_folder = r'F:\Ilastik' + os.sep + experiment + os.sep + avg_folder
#     img_folder = base_folder + os.sep + wall_string
#     img_list = os.listdir(img_folder)
    
#     for current_image_index in range(len(img_list)):
#         img_path = os.path.join(img_folder, img_list[current_image_index])
#         fname = experiment + '_' + wall_string + '_' + img_list[current_image_index]
#         rename_file(img_path, img_folder + os.sep + fname)

# #%%
# #only renaming
# for j in range(4,5):
#     experiment = 'F3_20230406'
#     avg_folder = '2023-04-06 11-07-24 phase averages'
#     wall_int = j
#     wall_string = str(wall_int).zfill(5)
#     base_folder = r'F:\Ilastik' + os.sep + experiment + os.sep + avg_folder
#     img_folder = base_folder + os.sep + wall_string
#     img_list = os.listdir(img_folder)
    
#     for current_image_index in range(len(img_list)):
#         img_path = os.path.join(img_folder, img_list[current_image_index])
#         fname = img_folder + os.sep + img_path[-18:]
#         rename_file(img_path, fname)
