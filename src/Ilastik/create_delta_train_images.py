# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 14:19:01 2023

@author: SWW-Bc20
"""

import os
import tifffile
import matplotlib.pyplot as plt
from skimage.morphology import binary_erosion, binary_dilation
import numpy as np
from PIL import Image
import pandas as pd


def create_seg_weight_and_eroded_objectId(objectId):
    min_weight = int(0.05 * 255)
    bg_boundary_weight = int(0.3 * 255)
    region_weight = int(0.4 * 255)
    inter_region_boundary_weight = int(1.0 * 255)

    seg = np.zeros_like(objectId, dtype=np.uint8)
    wei = np.full(objectId.shape, min_weight, dtype=np.uint8)
    eroded_objectId_final = np.zeros_like(objectId, dtype=np.uint8)

    for region in np.unique(objectId):
        if region == 0:
            continue

        region_mask = (objectId == region)
        erosed_mask = binary_erosion(region_mask)

        # Assign weights
        wei[region_mask] = region_weight
        boundary_mask = region_mask & ~erosed_mask
        wei[boundary_mask] = bg_boundary_weight

        # Erode another boundary layer and assign the same weight
        erosed_mask_2 = binary_erosion(erosed_mask)
        seg[erosed_mask_2] = 255
        boundary_mask_2 = erosed_mask & ~erosed_mask_2
        wei[boundary_mask_2] = bg_boundary_weight

        # Update the eroded objectId
        eroded_objectId_final[erosed_mask_2] = int(region)

    for region in np.unique(objectId):
        if region == 0:
            continue

        region_mask = (objectId == region)
        dilated_mask = binary_dilation(region_mask)
        dilated_boundry = dilated_mask & ~region_mask
        inter_region = dilated_boundry & (objectId != 0)
        wei[inter_region] = inter_region_boundary_weight

    return seg, wei, eroded_objectId_final

def save_image_as_png16(data, save_path):
    # Scale the array values from the range [-pi/2, pi/2] to the range [0, 65535]
    scaled_data = ((data + np.pi/2) / np.pi) * 65535
    # Convert the array to a 16-bit integer data type
    scaled_data = scaled_data.astype(np.uint16)
    # Create an image object from the array
    image = Image.fromarray(scaled_data, mode='I;16')
    # Save the image as a PNG file
    image.save(save_path)

def calculate_region_midpoint(image_mask):
    # Find the coordinates where the image mask equals 1
    region_coordinates = np.argwhere(image_mask == 1)

    # Calculate the centroid
    centroid_x = np.mean(region_coordinates[:, 1])
    centroid_y = np.mean(region_coordinates[:, 0])

    return centroid_x, centroid_y

def create_tracking_wei(parents_eroded_objectId_cropped, objectId):
    min_wei_bact = 0.1
    wei_bact = np.ones_like(parents_eroded_objectId_cropped, dtype=np.float32)*min_wei_bact
    min_light_bulb = 0.1
    light_bulb = np.ones_like(parents_eroded_objectId_cropped, dtype=np.float32)*min_light_bulb
    boundary_masks = []
    dilated_masks = []
    decrease_inside = 0.2
    decrease_outside = 0.05
    for region in np.unique(parents_eroded_objectId_cropped):
        if region == 0:
            continue
        
        if region == objectId:
            region_mask = (parents_eroded_objectId_cropped == region)
            while True:
                erosed_mask = binary_erosion(region_mask)
                boundary_masks.append(region_mask & ~erosed_mask)
                region_mask = erosed_mask
                if len(np.unique(erosed_mask)) == 1:
                    break
            boundary_masks = boundary_masks[::-1]
            for i, shell in enumerate(boundary_masks):
                wei_bact[shell] = np.exp(-decrease_inside*i)
            region_mask = (parents_eroded_objectId_cropped == region)
            light_bulb[region_mask] = 1.0
            for i in range(30):
                dilated_mask = binary_dilation(region_mask)
                dilated_masks.append(~region_mask & dilated_mask)
                region_mask = dilated_mask
            for i, shell in enumerate(dilated_masks):
                light_bulb[shell] = np.exp(-decrease_outside*i)
        
        else:
            region_mask = (parents_eroded_objectId_cropped == region)
            wei_bact[region_mask] = 0.25
            
    wei = (wei_bact*light_bulb*255).astype(np.uint8)
    return wei

def calculate_adjusted_centroids(image_mask, half_side_length):
    region_coordinates = np.argwhere(image_mask == 1)

    centroid_x = int(np.mean(region_coordinates[:, 1]))
    centroid_y = int(np.mean(region_coordinates[:, 0]))
    
    centroid_x = max(centroid_x, 2 + half_side_length)
    centroid_y = max(centroid_y, 2 + half_side_length)
    
    centroid_x = min(centroid_x, image_mask.shape[1] - half_side_length - 2)
    centroid_y = min(centroid_y, image_mask.shape[0] - half_side_length - 2)
    
    xmin, ymin = centroid_x - half_side_length, centroid_y - half_side_length
    xmax, ymax = centroid_x + half_side_length, centroid_y + half_side_length
    
    return centroid_x, centroid_y, xmin, ymin, xmax, ymax

#%%
# Change the folder names if neccessary
base_folder = r'C:\Users\SWW-Bc20\Documents\GitHub\ilastik\all'
image_folder = base_folder + os.sep + 'images'
objectId_folder = base_folder + os.sep + 'objectId'
save_base_folder = r'C:\Users\SWW-Bc20\Documents\GitHub\ilastik\delta'
segmentation_folder = save_base_folder + os.sep + 'segmentation_set'
if not os.path.exists(segmentation_folder):
    os.mkdir(segmentation_folder)
tracking_folder = save_base_folder + os.sep + 'tracking_set'
if not os.path.exists(tracking_folder):
    os.mkdir(tracking_folder)
names = os.listdir(image_folder)

unique_timelines = set(list([n[:-10] for n in names]))
# Can add all processed experiment positions here so they dont get processed again.
# At the moment they are all commented out (so all are processed).
processed_timelines = []
"""['F3_20230406_00001',
'F3_20230406_00002',
'F3_20230406_00003',
'F3_20230406_00004',
'NN_20230407_00001',
'NN_20230407_00002',
'NN_20230407_00003',
'NN_20230407_00004',
'NN_20230407_00006',
'NN_20230407_00007',
'NN_20230407_00009',
'NN_20230407_00010',
'E10_20230413_00001',
'E10_20230413_00002',
'E10_20230413_00003',
'E10_20230413_00004',
'E10_20230413_00005',
'E10_20230413_00009']"""
    
for processed_timeline in processed_timelines:
    unique_timelines.remove(processed_timeline)

#%%
prev_image = None
prev_segall = None
prev_eroded_objectId = None
for t in unique_timelines:
    df_tracking = pd.read_csv(base_folder + os.sep + 'tracking' + os.sep + t + '_CSV-Table.csv',
                              usecols= ['frame','labelimageId','trackId','parentTrackId'])
    timesteps = [int(s[-9:-4]) for s in names if s[:-10]== t]
    for timestep in timesteps:
        name = f'{t}_{str(timestep).zfill(5)}'
        image = tifffile.imread(image_folder + os.sep + name + '.tif')
        seg_img_folder = segmentation_folder + os.sep + 'img'
        if not os.path.exists(seg_img_folder):
            os.mkdir(seg_img_folder)
        save_image_as_png16(image, segmentation_folder + os.sep + 'img' + os.sep + name + '.png')
        objectId = tifffile.imread(objectId_folder + os.sep + name + '.tif')
        
        segall, segall_wei, eroded_objectId = create_seg_weight_and_eroded_objectId(objectId)
        seg_seg_folder = segmentation_folder + os.sep + 'seg'
        if not os.path.exists(seg_seg_folder):
            os.mkdir(seg_seg_folder)
        seg_wei_folder = segmentation_folder + os.sep + 'wei'
        if not os.path.exists(seg_wei_folder):
            os.mkdir(seg_wei_folder)
        Image.fromarray(segall, mode='L').save(segmentation_folder + os.sep + 'seg' + os.sep + name + '.png')
        Image.fromarray(segall_wei, mode='L').save(segmentation_folder + os.sep + 'wei' + os.sep + name + '.png')
        if prev_image is not None:
            
            i = 0
            cut_off = 50
            side_length = 300

            parents_eroded_objectId = np.zeros_like(eroded_objectId, dtype=np.uint16)
            for region in np.unique(eroded_objectId[cut_off:-cut_off, cut_off:-cut_off]):
                if region == 0:
                    continue
                region_mask = (eroded_objectId == region)
                twice_errorded_region = binary_erosion(binary_erosion(region_mask))
                
                trackId = df_tracking.loc[(df_tracking['frame'] == timestep) & (df_tracking['labelimageId'] == region), 'trackId'].item()
                if trackId < 0:
                    continue
                parentTrackId = df_tracking.loc[(df_tracking['frame'] == timestep) & (df_tracking['labelimageId'] == region), 'parentTrackId'].item()
                if parentTrackId == 0:
                    parentTrackId = trackId
                filtered_series = df_tracking.loc[(df_tracking['frame'] == timestep-1) & (df_tracking['trackId'] == parentTrackId), 'labelimageId']
                if not filtered_series.empty:
                    parentLabelimageId = filtered_series.item()
                else:
                    prev_id = np.unique(prev_eroded_objectId[twice_errorded_region])
                    # check if bacteria pops into existance somewhere new. If it is somewhere where another bacteria was it is probably a mistake. So unused
                    if len(prev_id)==1 and prev_id[0] == 0:
                        # new unused parent id, if two new bacterias pop into existance i makes sure they are differently labeled
                        parentLabelimageId = np.max(np.unique(prev_eroded_objectId))+1+i
                        i += 1
                    else:
                        continue
                    
                parents_eroded_objectId[region_mask] = parentLabelimageId
                
                # check if bacterias are merged, if so exclude
                prev_id = np.unique(prev_eroded_objectId[binary_erosion(twice_errorded_region)])
                prev_id = prev_id[prev_id != 0]
                current_id = np.unique(parents_eroded_objectId[binary_erosion(twice_errorded_region)])
                current_id = current_id[current_id != 0]
                if not np.array_equal(np.sort(prev_id), np.sort(current_id)) and len(prev_id) != 0:
                    continue
                
            for region in np.unique(parents_eroded_objectId[cut_off:-cut_off, cut_off:-cut_off]):
                if region == 0:
                    continue
                
                tracking_name = f'{name}_{str(region).zfill(5)}'
                region_mask = (parents_eroded_objectId == region)
                centroid_x, centroid_y, xmin, ymin, xmax, ymax = calculate_adjusted_centroids(region_mask, side_length//2)
                
                prev_image_cropped = prev_image[ymin : ymax, xmin : xmax]
                track_previmg_folder = tracking_folder + os.sep + 'previmg'
                if not os.path.exists(track_previmg_folder):
                    os.mkdir(track_previmg_folder)
                save_image_as_png16(prev_image_cropped, tracking_folder + os.sep + 'previmg' + os.sep + tracking_name + '.png')
                image_cropped = image[ymin : ymax, xmin : xmax]
                track_img_folder = tracking_folder + os.sep + 'img'
                if not os.path.exists(track_img_folder):
                    os.mkdir(track_img_folder)
                save_image_as_png16(image_cropped, tracking_folder + os.sep + 'img' + os.sep + tracking_name + '.png')
                segall_cropped = segall[ymin : ymax, xmin : xmax]
                track_segall_folder = tracking_folder + os.sep + 'segall'
                if not os.path.exists(track_segall_folder):
                    os.mkdir(track_segall_folder)
                Image.fromarray(segall_cropped, mode='L').save(tracking_folder + os.sep + 'segall' + os.sep + tracking_name + '.png')
                
                prev_eroded_objectId_cropped = prev_eroded_objectId[ymin : ymax, xmin : xmax]
                prev_seg_cropped = np.zeros_like(prev_eroded_objectId_cropped, dtype=np.uint8)
                region_mask = (prev_eroded_objectId_cropped == region)
                prev_seg_cropped[region_mask] = 255
                track_seg_folder = tracking_folder + os.sep + 'seg'
                if not os.path.exists(track_seg_folder):
                    os.mkdir(track_seg_folder)
                Image.fromarray(prev_seg_cropped, mode='L').save(tracking_folder + os.sep + 'seg' + os.sep + tracking_name + '.png')
                
                parents_eroded_objectId_cropped = parents_eroded_objectId[ymin : ymax, xmin : xmax]
                seg_cropped = np.zeros_like(parents_eroded_objectId_cropped, dtype=np.uint8)
                region_mask = (parents_eroded_objectId_cropped == region)
                seg_cropped[region_mask] = 255
                track_mot_dau_folder = tracking_folder + os.sep + 'mot_dau'
                if not os.path.exists(track_mot_dau_folder):
                    os.mkdir(track_mot_dau_folder)
                Image.fromarray(seg_cropped, mode='L').save(tracking_folder + os.sep + 'mot_dau' + os.sep + tracking_name + '.png')
                
                tracking_wei = create_tracking_wei(parents_eroded_objectId_cropped, region)
                track_wei_folder = tracking_folder + os.sep + 'wei'
                if not os.path.exists(track_wei_folder):
                    os.mkdir(track_wei_folder)
                Image.fromarray(tracking_wei, mode='L').save(tracking_folder + os.sep + 'wei' + os.sep + tracking_name + '.png')

        if os.path.isfile(image_folder + os.sep + f'{t}_{str(timestep+1).zfill(5)}.tif'):
            prev_image = image.copy()
            prev_segall = segall.copy()
            prev_eroded_objectId = eroded_objectId.copy()
        else:
            prev_image = None
            prev_segall = None
            prev_eroded_objectId = None
        
        print(image_folder + os.sep + f'{t}_{str(timestep+1).zfill(5)}.tif')


