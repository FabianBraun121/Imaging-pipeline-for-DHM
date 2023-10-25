# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 18:16:51 2023

@author: SWW-Bc20
"""
import os 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""
Position 9: Bild 148 -> Nur einzelne Zelle zoom funktioniert nicht richtig.
Position 9: Bild 260 -> Dominiert durch einzelner Grosser Fleck in der Mitte, Rotation funktioniert nicht
"""

def flatten(l):
    return [item for sublist in l for item in sublist]

base_path = r'C:\Users\SWW-Bc20\Documents\GitHub\Imaging-pipeline-for-DHM\data\brightfield\20230905-1643\aligned_images'

postitions = os.listdir(base_path)
all_rots = []
all_zooms = []
all_shift_vectors = []
for postition in postitions:
    positions_path = base_path + os.sep + postition
    all_rots.append(list(np.load(positions_path + os.sep + 'rotations.npy')))
    all_zooms.append(list(np.load(positions_path + os.sep + 'zooms.npy')))
    all_shift_vectors.append(list(np.load(positions_path + os.sep + 'shift_vectors.npy')))

#%%

plt.figure('zooms')
plt.plot(flatten(all_zooms))

#%%
pos = 0
plt.figure('zooms')
plt.plot(all_zooms[pos])
plt.figure('rots')
plt.plot(all_rots[pos])
plt.figure('shifts')
plt.plot(all_shift_vectors[pos])

#%%
plt.figure('mean zooms')
sns.violinplot(data=all_zooms[:7])
plt.figure('mean rots')
sns.violinplot(data=all_rots[:7])
