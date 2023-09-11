# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 17:32:41 2023

@author: SWW-Bc20
"""
import os
os.chdir(r'C:\Users\SWW-Bc20\Documents\GitHub\Imaging-pipeline-for-DHM\src\delta')
import delta
import numpy as np
import cv2
import copy
import matplotlib.pyplot as plt

def get_cell_cycle_frames(cell_cycle, img_stack, frame_size: int = 80):
    frames = []
    mid_point = np.mean([cell_cycle['new_pole'][1],cell_cycle['old_pole'][1]], axis=0).astype(np.uint16)
    for i in range(len(cell_cycle['frames'])):
        frame_nr = cell_cycle['frames'][i]
        frame = np.repeat(img_stack[frame_nr][:, :, np.newaxis], 3, axis=-1)
        frame = cv2.drawContours(frame, [cell_cycle['contours'][i]], 0, (1.0,0,0), thickness=1)
        frame = frame[mid_point[0]-frame_size//2:mid_point[0]+frame_size//2, mid_point[1]-frame_size//2:mid_point[1]+frame_size//2]
        frames.append(frame)
    return frames

def add_contours_to_linage(pos):
    lin = pos.rois[0].lineage
    label_stack = pos.rois[0].label_stack
    for cell in lin.cells:
        cell['contours'] = []
    for i in range(len(label_stack)):
        labels = label_stack[i]
        cell_nos, ind = np.unique(labels, return_index=True)
        cell_nos = [cell_no - 1 for _, cell_no in sorted(zip(ind, cell_nos))]  # Sorting along Y axis, 1-based to 0-based, & removing 1st value (background)
        cell_nos = [cell_no for cell_no in cell_nos if cell_no != -1]  # Remove background
        for c, cell_no in enumerate(cell_nos):
            # Have to do it this way to avoid cases where border is shrunk out
            cv2_contours, _ = cv2.findContours((labels == cell_no + 1).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            lin.cells[cell_no]['contours'].append(cv2_contours[0])
    return lin

def check_splits(cell_cycle):
    cell_opl = cell_cycle['integrated_opl']
    first_split = cell_opl[1]/cell_opl[0]
    second_split = cell_opl[-1]/cell_opl[-2]
    if first_split < 0.3 or 0.7 < first_split:
        return False
    elif second_split < 0.3 or 0.7 < second_split:
        return False
    else:
        return True

def ckeck_growth_steps(cell_cycle):
    cell_opl = cell_cycle['integrated_opl']
    growth = (np.roll(cell_opl,-1)/cell_opl)[1:-2]
    return np.all((growth >= 0.9) & (growth <= 1.3))

def check_absolute_growth(cell_cycle):
    cell_opl = cell_cycle['integrated_opl']
    abs_growth = cell_opl[-2]/cell_opl[1]
    if abs_growth < 1.5 or 3 < abs_growth:
        return False
    else:
        return True

def check_movement(cell_cycle):
    midpoints = np.mean([cell_cycle['new_pole'],cell_cycle['old_pole']], axis=0)
    midpoint_shifts = np.roll(midpoints,-1, axis=0)-midpoints
    distance = np.sqrt(np.sum(midpoint_shifts**2, axis=1))[1:-2]
    if np.all(distance < 10):
        return True
    else:
        return False


def get_cell_cycles(pos, with_contours: bool = True, only_valid: bool = True):
    # returns list with all complete cell cycles. One image before first split one after secand
    if with_contours:
        lin = add_contours_to_linage(pos)
    
    cell_cycles = []
    list_keys = [key for key, value in lin.cells[0].items() if isinstance(value, list)]
    for cell_no in range(len(lin.cells)):
        cell = copy.deepcopy(lin.cells[cell_no])
        # adding last mother frame if it exists for first split of this cell
        if cell['mother'] is not None:
            cell['daughters'][0] = cell['id'] # cell itself is the daughter of previous cell
            mother_cell = lin.cells[cell['mother']]
            mother_index_before_split = mother_cell['frames'].index(cell['frames'][0]-1)
            for list_key in list_keys:
                cell[list_key].insert(0, mother_cell[list_key][mother_index_before_split])
            cell['daughters'][0] = None # if two devisons are right after each other this can be not None (indicater for mistake)
        splits = np.where(np.array(cell['daughters']) != None)[0]
        # needs at least two splits to be a complete cell cycle
        if 2 <= len(splits):
            for i in range(len(splits)-1):
                if splits[i+1] == len(cell['frames'])-1: # checks if cell is still present after it splits
                    continue
                # copy full cell, crop out the cell cycle
                cell_cycle = copy.deepcopy(cell)
                for list_key in list_keys:
                    cell_cycle[list_key] = cell_cycle[list_key][splits[i]-1:splits[i+1]+1]
                cell_cycles.append(cell_cycle)
                del cell_cycle
            del cell
    
    if not only_valid:
        return cell_cycles
    else:
        valid_cell_cycles = []
        split_fail = []
        growth_steps_fail = []
        growth_abs_fail = []
        for cell_cycle in cell_cycles:
            if not check_splits(cell_cycle):
                split_fail.append(cell_cycle)
            if not ckeck_growth_steps(cell_cycle):
                growth_steps_fail.append(cell_cycle)
            if not ckeck_growth_steps(cell_cycle):
                growth_abs_fail.append(cell_cycle)
            if check_splits(cell_cycle) and ckeck_growth_steps(cell_cycle) and check_absolute_growth(cell_cycle) and check_movement(cell_cycle):
                valid_cell_cycles.append(cell_cycle)
        return valid_cell_cycles, split_fail, growth_steps_fail, growth_abs_fail

def plot_valid_cycles(cell_cycles, img_stack):
    # Create a figure with two rows and five columns
    fig, axes = plt.subplots(len(cell_cycles), 6, figsize=(12, 2*len(cell_cycles)))
    
    # Plot the first set of images in the top row
    for i in range(len(cell_cycles)):
        cycle_frames = get_cell_cycle_frames(cell_cycles[i], img_stack)
        frames = np.take(cycle_frames, [0,1,cycle_frames.shape//2,-2,-1], axis=0)
        for j in range(5):
            axes[i, j].imshow(frames[i])
            axes[i, j].axis('off')
        axes.plot(cell_cycles['mass'])

    plt.tight_layout()
    plt.show()

#%%
pos = delta.pipeline.load_position(r'F:\C11_20230217\2023-02-17 11-13-34 phase averages\00001\delta_results\Position00001.pkl')
cell_cycles, split_fail, growth_steps_fail, growth_abs_fail = get_cell_cycles(pos)
img_stack = pos.rois[0].img_stack
#%%
plt.plot(np.arange(len(split_fail[74]['mass']))*2, np.array(split_fail[74]['mass'])*1000, label='no split')
plt.plot(np.arange(len(split_fail[69]['mass']))*2, np.array(split_fail[69]['mass'])*1000, label='no mother')
plt.plot(np.arange(len(growth_steps_fail[4]['mass']))*2, np.array(growth_steps_fail[4]['mass'])*1000, label='impossible growth step')
plt.plot(np.arange(len(growth_abs_fail[2]['mass']))*2, np.array(growth_abs_fail[2]['mass'])*1000, label='Insufficient absolute growth')
plt.legend(fontsize=12)
plt.xlabel('time [min]', fontsize=12)
plt.ylabel('mass [fg]', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

#%%

def plot_valid_cycles(cell_cycles, img_stack):
    # Create a figure with two rows and five columns
    fig, axes = plt.subplots(len(cell_cycles), 6, figsize=(12, 2*len(cell_cycles)))
    
    # Plot the first set of images in the top row
    for i in range(len(cell_cycles)):
        cycle_frames = get_cell_cycle_frames(cell_cycles[i], img_stack)
        frames = np.take(cycle_frames, [0,1,len(cycle_frames)//2,-2,-1], axis=0)
        frames_mass = [cell_cycles[i]['mass'][0], cell_cycles[i]['mass'][1], cell_cycles[i]['mass'][len(cycle_frames)//2], cell_cycles[i]['mass'][-2], cell_cycles[i]['mass'][-1]]
        frames_time = [0, 2, len(cycle_frames)//2*2, len(cycle_frames)*2-4, len(cycle_frames)*2-2]
        axes[i, 5].plot(np.arange(len(cell_cycles[i]['mass']))*2, cell_cycles[i]['mass'])
        axes[i, 5].set_xlabel('time [min]')
        axes[i, 5].set_ylabel('mass [pg]')
        for j in range(5):
            axes[i, j].imshow(frames[j])
            axes[i, j].axis('off')
            axes[i, 5].plot(frames_time[j], frames_mass[j], 'ro', label='Current Value', markersize=6)

    plt.tight_layout()
    plt.show()

plot_valid_cycles(cell_cycles[2:5], img_stack)

#%%

cycle_frames = get_cell_cycle_frames(cell_cycles[1], img_stack)
frames = np.take(cycle_frames, [0,1,len(cycle_frames)//2,-2,-1], axis=0)