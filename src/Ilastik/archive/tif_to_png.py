# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 16:31:17 2023

@author: SWW-Bc20
"""
import cv2
import os
import tifffile
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def save_image_as_png16(data, save_path):
    # Scale the array values from the range [-pi/2, pi/2] to the range [0, 65535]
    scaled_data = ((data + np.pi/2) / np.pi) * 65535
    # Convert the array to a 16-bit integer data type
    scaled_data = scaled_data.astype(np.uint16)
    # Create an image object from the array
    image = Image.fromarray(scaled_data, mode='I;16')
    # Save the image as a PNG file
    image.save(save_path)

def save_image_as_png8(data, save_path):
    # Scale the array values from the range [-pi/2, pi/2] to the range [0, 65535]
    scaled_data = ((data + np.pi/2) / np.pi) * 255
    # Convert the array to a 8-bit integer data type
    scaled_data = scaled_data.astype(np.uint8)
    # Create an image object from the array
    image = Image.fromarray(scaled_data, mode='L')
    # Save the image as a PNG file
    image.save(save_path)

def save_tif_as_png8(data, save_path):
    scaled_data = (data * 255).astype(np.uint8)
    image = Image.fromarray(scaled_data, mode='L')
    image.save(save_path)
    
def scale_tif8Bit(path):
    data = tifffile.imread(path)
    scaled_data = ((data + np.pi/2) / np.pi) * 255
    scaled_data = scaled_data.astype(np.uint8)
    tifffile.imwrite(path, scaled_data)

def scale_tif16Bit(path):
    data = tifffile.imread(path)
    scaled_data = ((data + np.pi/2) / np.pi) * 65535
    scaled_data = scaled_data.astype(np.uint16)
    tifffile.imwrite(path, scaled_data)


#%%
folder = r'C:\Users\SWW-Bc20\Anaconda3\envs\delta_env\Lib\site-packages\delta\assets\trainingsets\2D\training\segmentation_set\img'
for file in os.listdir(folder):
    if file.endswith(".tif"):
        save_path = folder + os.sep + file[:-4] + ".png"
        data = tifffile.imread(folder + os.sep + file)
        save_image_as_png16(data, save_path)

#%%
folder = r'C:\Users\SWW-Bc20\Anaconda3\envs\delta_env\Lib\site-packages\delta\assets\trainingsets\2D\training\segmentation_set\test_seg'
for file in os.listdir(folder):
    if file.endswith(".tif"):
        save_path = folder + os.sep + file[:-4] + ".png"
        data = tifffile.imread(folder + os.sep + file)
        save_image_as_png8(data, save_path)
        
#%%

folder = r'C:\Users\SWW-Bc20\Anaconda3\envs\delta_env\Lib\site-packages\delta\assets\trainingsets\2D\training\segmentation_set\seg'
for file in os.listdir(folder):
    if file.endswith(".tif"):
        save_path = folder + os.sep + file[:-4] + ".png"
        data = tifffile.imread(folder + os.sep + file)
        save_tif_as_png8(data, save_path)
        
#%%
folder = r'C:\Users\SWW-Bc20\Anaconda3\envs\delta_env\Lib\site-packages\delta\assets\eval_movies\2D\movie_tifs'
for file in os.listdir(folder):
    if file.endswith(".tif"):
        save_path = folder + os.sep + file
        scale_tif16Bit(save_path)
