# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 11:17:48 2023

@author: SWW-Bc20
"""
import os 
import tifffile
import numpy as np

base_path = r'D:\data\brightfield\20230905-1643\aligned_images'
save_base_path = r'D:\data\brightfield\20230905-1643\bf_timeseries'
positions = os.listdir(base_path)

for position in positions[8:10]:
    pos_path = base_path + os.sep + position
    image_fnames = [f for f in os.listdir(pos_path) if f.endswith('BF.tif')]
    
    img_seq = []
    
    for fname in image_fnames[:30]:
        image = tifffile.imread(pos_path + os.sep + fname).astype(np.uint16)
        img_seq.append(image)
    
    tifffile.imwrite(save_base_path + os.sep + f'pos_{position}.tif', np.array(img_seq))
        