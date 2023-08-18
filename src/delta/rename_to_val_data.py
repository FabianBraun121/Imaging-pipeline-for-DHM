# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 20:53:23 2023

@author: SWW-Bc20
"""
import os

base_path = r'C:\Users\SWW-Bc20\Documents\GitHub\Imaging-pipeline-for-DHM\data\delta_assets\trainingsets\2D'
train_path = base_path + os.sep + 'training'
val_path = base_path + os.sep + 'validation'
validation_bacteria = ['E10_']

if not os.path.exists(val_path):
    os.mkdir(val_path)
a = 0
sets = os.listdir(train_path)
for s in sets:
    train_set_path = train_path + os.sep + s
    val_set_path = val_path + os.sep + s
    if not os.path.exists(val_set_path):
        os.mkdir(val_set_path)
    
    folders = os.listdir(train_set_path)
    for f in folders:
        train_folder_path = train_set_path + os.sep + f
        val_folder_path = val_set_path + os.sep + f
        if not os.path.exists(val_folder_path):
            os.mkdir(val_folder_path)
        
        files = os.listdir(train_folder_path)
        for file in files:
            for prefix in validation_bacteria:
                if file.startswith(prefix):
                    file_old = train_folder_path + os.sep + file
                    file_new = val_folder_path + os.sep + file
                    os.rename(file_old, file_new)