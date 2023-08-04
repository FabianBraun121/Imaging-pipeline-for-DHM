# -*- coding: utf-8 -*-
"""
Created on Mon May 22 11:52:43 2023

@author: SWW-Bc20
"""
import numpy as np
import cv2 as cv
import os
import binkoala
from PIL import Image

def watershed_seeded(Ilastik):
    
       
    
    # Ilastik = Ilastik[300:500, 100:300]
    
       
    
    center = np.where(Ilastik == 1, 255, 0)
    
    center = np.uint8(center)
    
       
    
    seg = np.where(Ilastik <= 2, 255, 0)
    
    seg  = np.uint8(seg)
    
     
    
    unknown = cv.subtract(seg, center)
    
    markers = cv.connectedComponents(center)
    
     
    
    # Add one to all labels so that sure background is not 0, but 1
    
    markers = markers[1]+1
    
    # Now, mark the region of unknown with zero
    
    markers[unknown==255] = 0
    
      
    
    img2 = np.stack((seg,)*3, axis=2)
    
    img2 = cv.normalize(img2, None, 255, 0, cv.NORM_MINMAX, cv.CV_8U)
    
       
    
    markers = cv.watershed(img2,markers)
    
       
    
    labeled = np.where(markers < 2, 0, markers)
    
       
    
    return labeled

 

#%%

thedir = r'F:\Ilastik\F3_20230406' 

for pp in range(1,2):
 

    position = str(pp).zfill(5)

    print(position)

   

    pos_dir = thedir + '/' + position

   

    list_files_bin = [ name for name in os.listdir(pos_dir) if os.path.isfile(os.path.join(pos_dir, name)) and name.endswith('.bin')]

    list_files_seg = [ name for name in os.listdir(pos_dir) if os.path.isfile(os.path.join(pos_dir, name)) and name.endswith('Segmentation.tif')]

    list_files_bin = list_files_bin[0:150]
    list_files_seg = list_files_seg[0:150]

   

    bin_list = []

    label_list = []


    # read in all bin files

   

    try:

        os.mkdir(pos_dir + '/segmented_centroids')

    except:

        print('folder exists')

   

    for fname in list_files_bin:

   

        ph_image, header = binkoala.read_mat_bin(os.path.join(pos_dir, fname))   

        scaling = header[0][5]*10**6 # in [um]

        size_pixel = scaling*scaling # in [um^2]

       

        bin_list.append(ph_image)

       

    for fname in list_files_seg:

   

        Ilastik = np.array(Image.open(os.path.join(pos_dir, fname)))

       

        labeled = watershed_seeded(Ilastik)

        label_list.append(labeled)

#%%
import h5py
import numpy as np
import matplotlib.pyplot as plt
import tifffile
from PIL import Image
from scipy.ndimage import binary_erosion
import os


def read_hdf5_image(file_path):
    images = []
    with h5py.File(file_path, 'r') as file:
        dataset = file['exported_data']
        for image_data in dataset:
            # Assuming the dataset contains images as arrays
            image = np.array(image_data).reshape(-1)
            images.append(image)
    return np.array(images)

def overlay_mask_with_slider(image, mask):
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
            boundary_mask = region_mask & ~binary_erosion(region_mask)
    
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

    intensity = 50  # Set the initial intensity value

    # Overlay the mask on the grayscale image
    result_image = overlay_mask(image, mask, intensity)

    # Display the result image with a slider to adjust the intensity
    fig, ax = plt.subplots()
    plt.subplots_adjust(top=0.95, bottom=0.07)

    im = ax.imshow(result_image)
    axcolor = 'lightgoldenrodyellow'
    ax_intensity = plt.axes([0.2, 0.01, 0.65, 0.03], facecolor=axcolor)
    slider_intensity = plt.Slider(ax_intensity, 'Intensity', 0, 100, valinit=intensity)

    def update_intensity(val):
        intensity = slider_intensity.val
        result_image = overlay_mask(image, mask, intensity)
        im.set_data(result_image)
        fig.canvas.draw()

    slider_intensity.on_changed(update_intensity)

    plt.show()
    
#%%

num_image = 10
overlay_mask_with_slider(bin_list[num_image], label_list[num_image])