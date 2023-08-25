# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 14:22:54 2023

@author: SWW-Bc20
"""
import pickle
import os
import matplotlib.pyplot as plt

path = r'C:\Users\SWW-Bc20\Documents\GitHub\Imaging-pipeline-for-DHM\tests\delta\data_augmentation'
historie_files = [h for h in os.listdir(path) if h.endswith('.pkl')]

histories = []
for h in historie_files:
    with open(path+os.sep+h, 'rb') as f:
        histories.append(pickle.load(f))

for i in range(len(histories)):
    plt.plot(histories[i]['val_loss'], label=historie_files[i])
plt.legend()
plt.show()