# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 21:18:28 2023

@author: SWW-Bc20
"""
import os
os.chdir(os.path.dirname(__file__))
import delta
from utilities import get_cell_cycles, get_cell_cycle_frames, display_frames_with_plot, display_image_series

#%%
pos = delta.pipeline.load_position(r'F:\C11_20230217\2023-02-17 11-13-34 phase averages\00001\delta_results\Position00001.pkl')
cell_cycles = get_cell_cycles(pos)
img_stack = pos.rois[0].img_stack

#%%
avg_length_cell_cycles = [cell_cycle for cell_cycle in cell_cycles if len(cell_cycle['mass'])<26 and len(cell_cycle['mass'])>23]

#%%
import matplotlib.pyplot as plt
import numpy as np

plt.figure()
for c in avg_length_cell_cycles:
    plt.plot(np.arange(len(c['mass']))*2, c['mass'])
plt.xlabel('time [min]', fontsize=14)
plt.ylabel('mass [pg]', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

#%%
plt.figure()
for c in cell_cycles[:10]:
    plt.plot(np.arange(len(c['mass']))*2, c['mass'])
plt.xlabel('time [min]', fontsize=14)
plt.ylabel('mass [pg]', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12) 
