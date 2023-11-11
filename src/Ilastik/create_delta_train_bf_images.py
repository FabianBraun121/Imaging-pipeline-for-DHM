# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 10:12:52 2023

@author: SWW-Bc20
"""
import os
import tifffile
import matplotlib.pyplot as plt
from skimage.morphology import binary_erosion, binary_dilation
from scipy import ndimage
import numpy as np
from PIL import Image
import pandas as pd
import h5py

def read_h5_file(file_path):
    with h5py.File(file_path, 'r') as h5_file:
        dataset_names = list(h5_file.keys())
        dataset = h5_file[dataset_names[-1]]  # Replace 'dataset_name' with the actual dataset name in your .h5 file
        array = np.array(dataset)
        
    return np.squeeze(array)

def create_seg_weight_and_objectId(object_ids):
    seg = (object_ids!=0).astype(np.uint8)
    outside_dist = ndimage.distance_transform_edt(seg)
    outside_dist = np.where(outside_dist<0.1,0,1/outside_dist)
    inside_dist = ndimage.distance_transform_edt((object_ids==0).astype(np.uint8))
    inside_dist = np.where(inside_dist<0.1,0,1/inside_dist)
    wei = np.where(outside_dist<inside_dist, inside_dist, outside_dist)
    wei = np.where(wei<0.1,0.1,wei)
    wei = (wei*255).astype(np.uint8)
    return seg, wei, object_ids

def calculate_region_midpoint(image_mask):
    # Find the coordinates where the image mask equals 1
    region_coordinates = np.argwhere(image_mask == 1)

    # Calculate the centroid
    centroid_x = np.mean(region_coordinates[:, 1])
    centroid_y = np.mean(region_coordinates[:, 0])

    return centroid_x, centroid_y

def create_tracking_wei(parents_objectId_cropped, objectId):
    min_wei_bact = 0.1
    wei_bact = np.ones_like(parents_objectId_cropped, dtype=np.float32)*min_wei_bact
    min_light_bulb = 0.1
    light_bulb = np.ones_like(parents_objectId_cropped, dtype=np.float32)*min_light_bulb
    boundary_masks = []
    dilated_masks = []
    decrease_inside = 0.2
    decrease_outside = 0.05
    for region in np.unique(parents_objectId_cropped):
        if region == 0:
            continue
        
        if region == objectId:
            region_mask = (parents_objectId_cropped == region)
            while True:
                erosed_mask = binary_erosion(region_mask)
                boundary_masks.append(region_mask & ~erosed_mask)
                region_mask = erosed_mask
                if len(np.unique(erosed_mask)) == 1:
                    break
            boundary_masks = boundary_masks[::-1]
            for i, shell in enumerate(boundary_masks):
                wei_bact[shell] = np.exp(-decrease_inside*i)
            region_mask = (parents_objectId_cropped == region)
            light_bulb[region_mask] = 1.0
            for i in range(30):
                dilated_mask = binary_dilation(region_mask)
                dilated_masks.append(~region_mask & dilated_mask)
                region_mask = dilated_mask
            for i, shell in enumerate(dilated_masks):
                light_bulb[shell] = np.exp(-decrease_outside*i)
        
        else:
            region_mask = (parents_objectId_cropped == region)
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
base_folder = r'D:\data\brightfield\20230905-1643\bf_timeseries'
image_series = [b for b in os.listdir(base_folder) if b.endswith('.tif')]
objectId_series = [b for b in os.listdir(base_folder) if b.endswith('Identities.h5')]
save_base_folder = r'C:\Users\SWW-Bc20\Documents\GitHub\Imaging-pipeline-for-DHM\data\delta_assets\trainingsets\2D\training'
segmentation_folder = save_base_folder + os.sep + 'bf_segmentation_set'
tracking_folder = save_base_folder + os.sep + 'bf_tracking_set'
#%%
prev_image = None
prev_segall = None
prev_objectId = None
for t in range(len(objectId_series)):
    df_tracking = pd.read_csv(base_folder + os.sep + f'pos_{str(t+1).zfill(5)}_CSV-Table.tif.csv',
                              usecols= ['frame','labelimageId','trackId','parentTrackId'])
    images = tifffile.imread(base_folder + os.sep + image_series[t])
    objectIds = read_h5_file(base_folder + os.sep + objectId_series[t])
    for timestep in range(images.shape[0]):
        name = f'20230905-1643_{str(t+1).zfill(5)}_{str(timestep).zfill(5)}'
        image = images[timestep].astype(np.uint16)
        Image.fromarray(image, mode='I;16').save(segmentation_folder + os.sep + 'img' + os.sep + name + '.png')
        
        objectId = objectIds[timestep]
        
        segall, segall_wei, objectId = create_seg_weight_and_objectId(objectId)
        Image.fromarray(segall, mode='L').save(segmentation_folder + os.sep + 'seg' + os.sep + name + '.png')
        Image.fromarray(segall_wei, mode='L').save(segmentation_folder + os.sep + 'wei' + os.sep + name + '.png')
        if prev_image is not None:
            
            i = 0
            cut_off = 50
            side_length = 300

            parents_objectId = np.zeros_like(objectId, dtype=np.uint16)
            for region in np.unique(objectId[cut_off:-cut_off, cut_off:-cut_off]):
                if region == 0:
                    continue
                region_mask = (objectId == region)
                twice_errorded_region = binary_erosion(binary_erosion(region_mask))
                
                trackId_series = df_tracking.loc[(df_tracking['frame'] == timestep) & (df_tracking['labelimageId'] == region), 'trackId']
                if len(trackId_series) != 1:
                    continue
                trackId = trackId_series.item()
                if trackId < 0:
                    continue
                parentTrackId = df_tracking.loc[(df_tracking['frame'] == timestep) & (df_tracking['labelimageId'] == region), 'parentTrackId'].item()
                if parentTrackId == 0:
                    parentTrackId = trackId
                filtered_series = df_tracking.loc[(df_tracking['frame'] == timestep-1) & (df_tracking['trackId'] == parentTrackId), 'labelimageId']
                if not filtered_series.empty:
                    parentLabelimageId = filtered_series.item()
                else:
                    prev_id = np.unique(prev_objectId[twice_errorded_region])
                    # check if bacteria pops into existance somewhere new. If it is somewhere where another bacteria was it is probably a mistake. So unused
                    if len(prev_id)==1 and prev_id[0] == 0:
                        # new unused parent id, if two new bacterias pop into existance i makes sure they are differently labeled
                        parentLabelimageId = np.max(np.unique(prev_objectId))+1+i
                        i += 1
                    else:
                        continue
                    
                parents_objectId[region_mask] = parentLabelimageId
                
                # check if bacterias are merged, if so exclude
                prev_id = np.unique(prev_objectId[binary_erosion(twice_errorded_region)])
                prev_id = prev_id[prev_id != 0]
                current_id = np.unique(parents_objectId[binary_erosion(twice_errorded_region)])
                current_id = current_id[current_id != 0]
                if not np.array_equal(np.sort(prev_id), np.sort(current_id)) and len(prev_id) != 0:
                    continue
                
            for region in np.unique(parents_objectId[cut_off:-cut_off, cut_off:-cut_off]):
                if region == 0:
                    continue
                
                tracking_name = f'{name}_{str(region).zfill(5)}'
                region_mask = (parents_objectId == region)
                centroid_x, centroid_y, xmin, ymin, xmax, ymax = calculate_adjusted_centroids(region_mask, side_length//2)
                
                prev_image_cropped = prev_image[ymin : ymax, xmin : xmax]
                Image.fromarray(prev_image_cropped.astype(np.uint16), mode='I;16').save(tracking_folder + os.sep + 'previmg' + os.sep + tracking_name + '.png')
                image_cropped = image[ymin : ymax, xmin : xmax]
                Image.fromarray(image_cropped.astype(np.uint16), mode='I;16').save(tracking_folder + os.sep + 'img' + os.sep + tracking_name + '.png')
                segall_cropped = segall[ymin : ymax, xmin : xmax]
                Image.fromarray(segall_cropped, mode='L').save(tracking_folder + os.sep + 'segall' + os.sep + tracking_name + '.png')
                
                prev_objectId_cropped = prev_objectId[ymin : ymax, xmin : xmax]
                prev_seg_cropped = np.zeros_like(prev_objectId_cropped, dtype=np.uint8)
                region_mask = (prev_objectId_cropped == region)
                prev_seg_cropped[region_mask] = 255
                Image.fromarray(prev_seg_cropped, mode='L').save(tracking_folder + os.sep + 'seg' + os.sep + tracking_name + '.png')
                
                parents_objectId_cropped = parents_objectId[ymin : ymax, xmin : xmax]
                seg_cropped = np.zeros_like(parents_objectId_cropped, dtype=np.uint8)
                region_mask = (parents_objectId_cropped == region)
                seg_cropped[region_mask] = 255
                Image.fromarray(seg_cropped, mode='L').save(tracking_folder + os.sep + 'mot_dau' + os.sep + tracking_name + '.png')
                
                tracking_wei = create_tracking_wei(parents_objectId_cropped, region)
                Image.fromarray(tracking_wei, mode='L').save(tracking_folder + os.sep + 'wei' + os.sep + tracking_name + '.png')

        prev_image = image.copy()
        prev_segall = segall.copy()
        prev_objectId = objectId.copy()
    
    prev_image = None
    prev_segall = None
    prev_objectId = None
