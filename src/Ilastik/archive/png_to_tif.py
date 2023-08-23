# -*- coding: utf-8 -*-
"""
Created on Tue May 30 09:54:47 2023

@author: SWW-Bc20
"""
from PIL import Image
import os
import numpy as np
import tifffile as tf

def convert_png_to_tiff(png_file, tiff_file):
    # Open the PNG file
    image = Image.open(png_file)
    
    # Convert the image to grayscale if it has multiple channels
    if image.mode != 'L':
        image = image.convert('L')
    
    # Convert the pixel values from integers (0-255) to floats (0-1)
    image = image.point(lambda x: x / 255.0)
    
    # Convert the image to a NumPy array with floating-point values
    data = np.array(image)
    data = data.astype(np.float32)
    
    # Save the image as a TIFF file with floating-point pixel values
    tf.imwrite(tiff_file, data)
    
    print(f"Conversion complete. Saved as {tiff_file}")

seg_path = r'F:\Ilastik\seg'
wei_path = r'F:\Ilastik\wei'

for seg in os.listdir(seg_path):
    convert_png_to_tiff(seg_path + os.sep + seg, seg_path + os.sep + seg[:-4] + '.tif')

for wei in os.listdir(wei_path):
    convert_png_to_tiff(wei_path + os.sep + wei, wei_path + os.sep + wei[:-4] + '.tif')