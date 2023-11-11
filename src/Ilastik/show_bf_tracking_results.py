# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 16:52:09 2023

@author: SWW-Bc20
"""
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

base_dir = r'C:\Users\SWW-Bc20\Documents\GitHub\Imaging-pipeline-for-DHM\data\delta_assets\trainingsets\2D\training\bf_tracking_set'
image_folders = os.listdir(base_dir)

for j in range(10,30):
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    
    image_nr = 10*j
    image_name = os.listdir(base_dir + os.sep + image_folders[0])[image_nr]
    for i, image_folder in enumerate(image_folders):
        axs[i//3, i%3].imshow(np.array(cv2.imread(base_dir + os.sep + image_folder + os.sep + image_name, cv2.IMREAD_UNCHANGED)))
        axs[i//3, i%3].set_title(image_folder)
    fig.show()

