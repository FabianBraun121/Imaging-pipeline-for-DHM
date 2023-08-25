# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 14:22:54 2023

@author: SWW-Bc20
"""
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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
#%%
names = np.array(historie_files).reshape(3,3)
data = np.array([np.min(histories[i]['val_loss']) for i in range(9)]).reshape(3,3)


ax = sns.heatmap(data, annot=True, fmt=".1f", cmap='coolwarm', 
                 xticklabels=['0.0024, 0.08', '0.012, 0.4', '0.06, 2'], 
                 yticklabels=['0', '10', '50'],
                 annot_kws={"size": 16}) # Font size of annotations

plt.xlabel('Gaussian noise and blur', fontsize=16) # Font size of X-axis label
plt.ylabel('Sigma of elastic deformation', fontsize=16) # Font size of Y-axis label
plt.xticks(fontsize=14) # Font size of x-tick labels
plt.yticks(fontsize=14) # Font size of y-tick labels
plt.show()