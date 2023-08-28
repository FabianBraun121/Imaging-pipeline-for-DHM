# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 21:55:40 2023

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
nr = 3
frames = get_cell_cycle_frames(cell_cycles[nr], img_stack)
values = cell_cycles[nr]['mass']
display_frames_with_plot(frames, values)

#%%
image_series_list = []
values_list = []
start = 50
for i in range(start, start+9):
    image_series_list.append(get_cell_cycle_frames(cell_cycles[i], img_stack))
    values_list.append(cell_cycles[i]['mass'])
display_image_series(image_series_list, values_list)

#%%