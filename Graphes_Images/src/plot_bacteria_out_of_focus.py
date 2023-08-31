# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 18:43:27 2023

@author: SWW-Bc20
"""
import os
os.chdir(r'C:\Users\SWW-Bc20\Documents\GitHub\Imaging-pipeline-for-DHM\src\spatial_averaging')
import binkoala
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from skimage.morphology import binary_dilation
from skimage.registration import phase_cross_correlation
import json
import cv2

base_path = r'F:\Out_of_focus\E10_20230216\00001'
focus_folders = os.listdir(base_path)
in_focus_path = base_path + os.sep + focus_folders[0]
image_list = os.listdir(in_focus_path)
mae_difs = np.zeros((10,len(image_list)))
mean_weight = np.zeros((11,len(image_list)))
for i, image_name in enumerate(image_list):
    image_f, _ = binkoala.read_mat_bin(in_focus_path + os.sep + image_name)
    mask = image_f>0.25
    mask = binary_dilation(binary_dilation(binary_dilation(mask)))
    mean_weight[0,i] = np.mean(image_f[mask])
    for j, focus_folder in enumerate(focus_folders[1:]):
        image_uf, _ = binkoala.read_mat_bin(base_path + os.sep + focus_folder + os.sep + image_name)
        dif = (image_f-image_uf)[mask]*794/(2*np.pi)
        mae_difs[j,i] = np.mean(np.abs(dif))
        mean_weight[j+1,i] = np.mean(image_uf[mask])
#%%

# Create bar chart
percentage_weight = ((mean_weight/mean_weight[0])[1:])*100
plt.bar(np.arange(5), np.mean(percentage_weight, axis=1)[::2], yerr=np.std(percentage_weight, axis=1)[::2], align='center', alpha=0.7, capsize=15, color='blue')

plt.yticks(fontsize=20)
# Add labels and title
plt.xticks(np.arange(5), np.arange(1,11)[::2], fontsize=20)
plt.ylabel('Normalized mean of all bacteria [%]', fontsize=20)
plt.xlabel('Reconstructed images out of focus', fontsize=20)
# plt.title('Decrease in Noise Level Across Processing Steps')
plt.ylim([75,100])

# Display the plot
plt.tight_layout()
plt.show()




