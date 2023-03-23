# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 15:07:26 2023

@author: SWW-Bc20
"""
import os
os.chdir(r'C:\Users\SWW-Bc20\Documents\GitHub\Imaging-pipeline-for-DHM\src\spatial_averaging')
import numpy as np
import matplotlib.pyplot as plt
import binkoala
import tifffile as tiff

def import_tiff_image_with_header(file_path):
    """
    Imports a Tagged Image File Format (TIFF) image and returns a tuple containing
    the header information and the image data as a NumPy array.
    """
    with tiff.TiffFile(file_path) as tif:
        header = tif.pages[0].tags
        img_array = np.array(tif.pages[0].asarray())
    return header, img_array

fname =  r"F:\F3_20230302\2023-03-02 11-30-06\00003\00001_00001\Holograms\00001_holo.tif"
fname_background = r"C:\ProgramData\LynceeTec\Koala\518\rh.bin"

header, img_array = import_tiff_image_with_header(fname)
a = binkoala.read_mat_cplx_bin(fname_background)

#%%
plt.figure()
plt.imshow(np.angle(a))


#%%
# Define the hologram
hologram = img_array

# Define the reconstruction distance
distance = 0

# Define the wavelength of the light used to record the hologram
wavelength = 794*1e-9

# Define the size of the hologram in meters
hologram_size = 0.01

# Define the pixel size of the hologram in meters
pixel_size = hologram_size / hologram.shape[0]

# Calculate the spatial frequency
spatial_frequency = np.fft.fftfreq(hologram.shape[0], d=pixel_size)

# Generate the transfer function
transfer_function = np.exp(-1j * 2 * np.pi * distance * np.sqrt(1 - (wavelength * spatial_frequency)**2))

# Apply the transfer function to the hologram
reconstructed_wave = np.fft.fft2(hologram) * transfer_function

# Calculate the amplitude and phase images
amplitude = np.abs(reconstructed_wave[112:912,112:912])
phase = np.angle(reconstructed_wave[112:912,112:912])

# Display the amplitude and phase images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(amplitude, cmap='gray')
plt.title('Amplitude Image')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(phase, cmap='gray')
plt.title('Phase Image')
plt.axis('off')
plt.show()