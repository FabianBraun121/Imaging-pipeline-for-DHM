# -*- coding: utf-8 -*-
"""
Created on Fri May  5 15:23:16 2023

@author: SWW-Bc20
"""
import tifffile

def split_tiff_file(input_file_path, output_folder_path):
    with tifffile.TiffFile(input_file_path) as tif:
        # get the number of pages in the TIFF file
        num_pages = len(tif.pages)
        
        # loop through each page and save it as an individual TIFF file
        for i in range(num_pages):
            # get the current page
            current_page = tif.pages[i]
            
            # get the numpy array from the current page
            current_array = current_page.asarray()
            
            # construct the output file path
            output_file_path = f"{output_folder_path}/timestep_{str(i).zfill(5)}.tif"
            
            # save the current page as an individual TIFF file
            tifffile.imwrite(output_file_path, current_array)

input_file_path = r"F:\Delta\C11_20230217\00001_segmentation.tif"
output_folder_path = r"F:\Delta\C11_20230217\00001_segmentation"
split_tiff_file(input_file_path, output_folder_path)

#%%

import os
from PIL import Image

def process_images_in_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.endswith(".tif"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            process_image(input_path, output_path)

def process_image(input_path, output_path):
    image = Image.open(input_path)
    image_array = image.load()

    width, height = image.size
    for y in range(height):
        for x in range(width):
            pixel_value = image_array[x, y]
            if pixel_value > 0.5:
                image_array[x, y] = 1
            else:
                image_array[x, y] = 0

    image.save(output_path)

# Example usage:
input_folder = r"F:\Delta\C11_20230217\00001_probabilities"
output_folder = r"F:\Delta\C11_20230217\00001_segmentation"
process_images_in_folder(input_folder, output_folder)