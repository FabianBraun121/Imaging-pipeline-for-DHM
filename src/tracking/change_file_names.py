# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 09:50:27 2023

@author: SWW-Bc20
"""
import os
#%%
input_dir = r"C:\Users\SWW-Bc20\Anaconda3\envs\delta_env\Lib\site-packages\delta\assets\eval_movies\2D\movie_tifs"  # Change this to your source directory
#input_dir = r"C:\Users\SWW-Bc20\Anaconda3\envs\delta_env\Lib\site-packages\delta\assets\trainingsets\2D\training\tracking_set\test_tracking\Segmentation"  # Change this to your source directory
file_extension = ".tif"  # Change this to your file extension

for i, filename in enumerate(os.listdir(input_dir)):
    if filename.endswith(file_extension):
        old_file = os.path.join(input_dir, filename)
        new_file = os.path.join(input_dir, f"Position01_Chamber01_Frame{str(i).zfill(3)}{file_extension}")
        os.rename(old_file, new_file)

print("Renaming of images completed!")


