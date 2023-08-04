# -*- coding: utf-8 -*-
"""
Created on Fri May 26 17:08:33 2023

@author: SWW-Bc20
"""
import os


base_path = r'C:\Users\SWW-Bc20\Anaconda3\envs\delta_env\Lib\site-packages\delta\assets\trainingsets\2D\training\segmentation_set'
img_path = base_path + os.sep + 'img'
seg_path = base_path + os.sep + 'seg'
wei_path = base_path + os.sep + 'wei'

img = os.listdir(img_path)
seg = os.listdir(seg_path)
wei = os.listdir(wei_path)

img = [f[:-4] for f in img]
seg = [f[:-4] for f in seg]
wei = [f[:-4] for f in wei]

if wei == seg == img:
    print('wei and seg and img are equal.')
else:
    print('wei and seg and img are not equal.')