# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 14:08:13 2023

@author: SWW-Bc20
"""
import json
import numpy as np
import matplotlib.pyplot as plt

data_file_path = r'F:\F3_20230302\2023-03-02 11-30-06 phase averages\data file 2023-07-27 17-08-53.json'
with open(data_file_path, 'r') as file:
    data = json.load(file)
    
print(data['settings'])

foci = np.array([data['images'][d]['foci'] for d in list(data['images'])])
time = np.array([data['images'][d]['time'] for d in list(data['images'])])
function_eval = np.array([data['images'][d]['function_evaluations'] for d in list(data['images'])])

data_file_path1 = r'F:\F3_20230302\2023-03-02 11-30-06 phase averages\data file 2023-07-27 18-59-01.json'
with open(data_file_path1, 'r') as file:
    data1 = json.load(file)

foci1 = np.array([data1['images'][d]['foci'] for d in list(data1['images'])])
time1 = np.array([data1['images'][d]['time'] for d in list(data1['images'])])
function_eval1 = np.array([data1['images'][d]['function_evaluations'] for d in list(data1['images'])])

#%%
fig, ax1 = plt.subplots()
ax1.plot(time,'b', label='time')
ax1.set_ylabel('time [s]', color='b')
ax1.set_xlabel('image number, reset each 10 images')
ax2 = ax1.twinx()
ax2.plot(function_eval,'g', label='function evaluations')
ax2.set_ylabel('function evaluations []', color='g')
plt.show()

