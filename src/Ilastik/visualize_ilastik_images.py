# -*- coding: utf-8 -*-
"""
Created on Wed May 24 17:18:05 2023

@author: SWW-Bc20
"""

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from skimage.morphology import binary_erosion, binary_dilation
import h5py
import os
import tifffile
import pandas as pd

def overlay_mask_with_slider(images_array, masks_array, df_tracking):
    current_image_index = 1

    intensity = 50  # Set the initial intensity value
                
    # Overlay the mask on the grayscale image
    def overlay_mask(image, mask, intensity):
        # Normalize the intensity value to the range [0, 1]
        intensity = intensity / 100.0
    
        # Normalize the grayscale image to the range [0, 255]
        image_normalized = (image - np.min(image)) / (np.max(image) - np.min(image))
        image_normalized = (image_normalized * 255).astype(np.uint8)
    
        # Convert the normalized grayscale image to RGB
        image_rgb = Image.fromarray(image_normalized).convert("RGB")
    
        # Create an empty mask overlay image with an alpha channel
        overlay = np.zeros_like(image_rgb)
        overlay_alpha = np.zeros((image_rgb.size[1], image_rgb.size[0]), dtype=np.uint8)
    
        # Loop through each unique region value in the mask
        for region in np.unique(mask):
            # Skip the background region (value of 0)
            if region == 0:
                continue
    
            # Generate a color for the region based on the region value
            color = (region * 10 % 256, region * 50 % 256, region * 100 % 256)
            
            # Compute boundary mask as the outermost part of each region
            region_mask = (mask == region)
            boundary_mask = ~region_mask & binary_dilation(region_mask)
    
            # Set the color of boundary pixels to red
            overlay[boundary_mask] = (255, 0, 0)
            overlay_alpha[boundary_mask] = int(255 * intensity)
    
            # Find the pixels where the mask matches the current region value (excluding the boundary)
            region_pixels = np.where(region_mask & ~boundary_mask)
    
            # Set the corresponding pixels in the overlay image with the adjusted intensity and color
            overlay[region_pixels] = color
            overlay_alpha[region_pixels] = int(255 * intensity)
    
        # Convert the overlay and alpha masks to PIL images
        overlay_image = Image.fromarray(overlay)
        overlay_alpha_image = Image.fromarray(overlay_alpha, mode='L')
    
        # Create an RGBA image by combining the overlay and original image with alpha channel
        overlay_image.putalpha(overlay_alpha_image)
        result_image = Image.alpha_composite(image_rgb.convert("RGBA"), overlay_image)
    
        return result_image
    
    def overlay_masks(image, mask_, prev_image, prev_mask_, intensity):
        mask, prev_mask = mask_.copy(), prev_mask_.copy()
        nonlocal current_image_index
        parents_objectId_mask = np.zeros_like(mask, dtype=np.uint16)
        for region in np.unique(mask):
            if region == 0:
                continue
            
            trackId = df_tracking.loc[(df_tracking['frame'] == current_image_index) & (df_tracking['labelimageId'] == region), 'trackId'].item()
            if trackId < 0:
                continue
            parentTrackId = df_tracking.loc[(df_tracking['frame'] == current_image_index) & (df_tracking['labelimageId'] == region), 'parentTrackId'].item()
            if parentTrackId == 0:
                parentTrackId = trackId
            filtered_series = df_tracking.loc[(df_tracking['frame'] == current_image_index-1) & (df_tracking['trackId'] == parentTrackId), 'labelimageId']
            if not filtered_series.empty:
                parentLabelimageId = filtered_series.item()
            else:
                continue
            
            region_mask = (mask == region)
            boundary_mask = region_mask & ~binary_erosion(region_mask)
            parents_objectId_mask[region_mask] = parentLabelimageId
            parents_objectId_mask[boundary_mask] = 0
        
        # erose outermost layer
        for region in np.unique(mask):
            if region == 0:
                continue
            region_mask = (prev_mask == region)
            boundary_mask = region_mask & ~binary_erosion(region_mask)
            prev_mask[boundary_mask] = 0
        
        result_previous_image = overlay_mask(prev_image, prev_mask, intensity)
        result_current_image = overlay_mask(image, parents_objectId_mask, intensity)
        return result_current_image, result_previous_image

    def load_image_and_mask():
        nonlocal current_image_index
        image = images_array[current_image_index]
        mask = masks_array[current_image_index]
        prev_image = images_array[current_image_index-1]
        prev_mask = masks_array[current_image_index-1]
        return image, mask, prev_image, prev_mask

    def update_intensity(val):
        nonlocal current_image_index
        intensity = slider_intensity.val
        image, mask, prev_image, prev_mask = load_image_and_mask()
        result_current_image, result_previous_image = overlay_masks(image, mask, prev_image, prev_mask, intensity)
        im1.set_data(result_previous_image)
        im2.set_data(result_current_image)
        fig.canvas.draw()

    def update_image(val):
        nonlocal current_image_index
        current_image_index = int(slider_image.val)
        image, mask, prev_image, prev_mask = load_image_and_mask()
        intensity = slider_intensity.val
        result_current_image, result_previous_image = overlay_masks(image, mask, prev_image, prev_mask, intensity)
        im1.set_data(result_previous_image)
        im2.set_data(result_current_image)
        fig.canvas.draw()

    def load_previous_image(event):
        nonlocal current_image_index
        if current_image_index > 0:
            current_image_index -= 1
            slider_image.set_val(current_image_index)
            update_image(None)

    def load_next_image(event):
        nonlocal current_image_index
        if current_image_index < len(images_array) - 1:
            current_image_index += 1
            slider_image.set_val(current_image_index)
            update_image(None)

    def on_key_press(event):
        if event.key == 'left':
            load_previous_image(None)
        elif event.key == 'right':
            load_next_image(None)
        elif event.key == 'up':
            slider_intensity.set_val(slider_intensity.val + 1)
        elif event.key == 'down':
            slider_intensity.set_val(slider_intensity.val - 1)

    # Load the first image and mask
    image, mask, prev_image, prev_mask = load_image_and_mask()

    # Create figure and axes
    fig, (ax1, ax2) = plt.subplots(1, 2)
    plt.subplots_adjust(top=0.95, bottom=0.1)

    # Display the result image with sliders to adjust the intensity and select the image
    result_current_image, result_previous_image = overlay_masks(image, mask, prev_image, prev_mask, intensity)
    im1 = ax1.imshow(result_previous_image)
    ax1.set_title('Previous Image')
    im2 = ax2.imshow(result_current_image)
    ax2.set_title('Current Image')
    
    
    axcolor = 'lightgoldenrodyellow'
    ax_intensity = plt.axes([0.2, 0.01, 0.65, 0.03], facecolor=axcolor)
    slider_intensity = plt.Slider(ax_intensity, 'Intensity', 0, 100, valinit=intensity)
    
    ax_image = plt.axes([0.2, 0.03, 0.65, 0.03], facecolor=axcolor)
    slider_image = plt.Slider(ax_image, 'Image', 1, len(images_array) - 1, valinit=current_image_index, valstep=1)

    slider_intensity.on_changed(update_intensity)
    slider_image.on_changed(update_image)

    # Connect keyboard events
    fig.canvas.mpl_connect('key_press_event', on_key_press)

    plt.show()

def read_h5_file(file_path):
    with h5py.File(file_path, 'r') as h5_file:
        dataset_names = list(h5_file.keys())
        dataset = h5_file[dataset_names[-1]]  # Replace 'dataset_name' with the actual dataset name in your .h5 file
        array = np.array(dataset)
        
    return np.squeeze(array)

def save_tif_float32(array, filename):
    # Convert the array to float32
    array = np.asarray(array, dtype=np.float32)
    
    # Save the array as a float32 .tif image using tifffile
    tifffile.imwrite(filename, array)

def save_tif_uint8(array, filename):
    # Convert the array to uint8
    array = np.asarray(array, dtype=np.uint8)
    
    # Save the array as a uint8 .tif image using tifffile
    tifffile.imwrite(filename, array)

#%%
# Define the positions, change the folder locations if necessary.
base_folder = r'C:\Users\SWW-Bc20\Documents\GitHub\Imaging-pipeline-for-DHM\data\ilastik'
data_series = 'E10_20230413'
position = '00012'
position_folder = base_folder + os.sep + data_series
image_folder = position_folder + os.sep + f'{data_series}_{position}.h5'
mask_folder = position_folder + os.sep + f'{data_series}_{position}_Object-Identities.h5'
tracking = position_folder + os.sep + f'{data_series}_{position}_CSV-Table.csv'
images = read_h5_file(image_folder)
masks = read_h5_file(mask_folder)
df_tracking = pd.read_csv(tracking, usecols= ['frame','labelimageId','trackId','parentTrackId'])

# Use this to see until which frame Ilastik's segmentation worked.
# Remember the frame and put it in the block below
overlay_mask_with_slider(images, masks, df_tracking)

#%%
# Define save location, change the folder locations if necessary.
save_base_folder = r'C:\Users\SWW-Bc20\Documents\GitHub\Imaging-pipeline-for-DHM\data\ilastik\all'
save_images_folder = save_base_folder + os.sep + 'images'
save_objectId_folder = save_base_folder + os.sep + 'objectId'

# IMPORTANT change with the range until which frame images should be saved
# One position at the time
for i in range(19):
    timestep = str(i).zfill(5)
    name = f'{data_series}_{position}_{timestep}.tif'
    image = images[i]
    objectId = masks[i]
    save_tif_float32(image, save_images_folder + os.sep + name)
    save_tif_uint8(objectId, save_objectId_folder + os.sep + name)
