# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 14:08:13 2023

@author: SWW-Bc20
"""
import json
import numpy as np

data_file_path = r'F:\20230720_EcoliTest_2 phase averages\data file 2023-07-26 13-42-41.json'
with open(data_file_path, 'r') as file:
    data = json.load(file)
    
print(data['settings'])

foci = np.array([data['images'][d]['foci'] for d in list(data['images'])])
time = np.array([data['images'][d]['time'] for d in list(data['images'])])