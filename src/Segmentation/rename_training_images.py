# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 15:39:01 2023

@author: SWW-Bc20
"""
import os

#%%
folder_path = r'C:\Users\SWW-Bc20\Anaconda3\envs\delta_env\Lib\site-packages\delta\assets\trainingsets\2D\training\segmentation_set'

img = folder_path + os.sep + 'img'
seg = folder_path + os.sep + 'seg'
wei = folder_path + os.sep + 'wei'
image_names = os.listdir(img)

for i, name in enumerate(image_names):
    stri = str(i).zfill(6)
    new_name = 'Sample' + stri +'.png'
    os.rename(img + os.sep + name, img + os.sep + new_name)
    os.rename(seg + os.sep + name, seg + os.sep + new_name)
    os.rename(wei + os.sep + name, wei + os.sep + new_name)

#%%
folder_path = r'C:\Users\SWW-Bc20\Anaconda3\envs\delta_env\Lib\site-packages\delta\assets\trainingsets\2D\training\segmentation_set'

img = folder_path + os.sep + 'img'
seg = folder_path + os.sep + 'seg'
wei = folder_path + os.sep + 'wei'
image_names = os.listdir(img)

for i, name in enumerate(image_names):
    stri = str(i).zfill(6)
    new_name = 'Sample' + stri +'.png'
    os.rename(img + os.sep + name, img + os.sep + new_name)
    os.rename(seg + os.sep + name, seg + os.sep + new_name)
    os.rename(wei + os.sep + name, wei + os.sep + new_name)