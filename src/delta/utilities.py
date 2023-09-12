# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 08:14:37 2023

@author: SWW-Bc20
"""
import delta
import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2

def add_contours_to_linage(pos):
    "adds contours of the cell boundaries to the lineages features."
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
    "checks if the bacteria as between 0.3 and 0.7 times the mass after the split compared to before."
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
    "checks if the induvidual growth steps are between 0.9 and 1.3 times the previous mass."
    cell_opl = cell_cycle['integrated_opl']
    growth = (np.roll(cell_opl,-1)/cell_opl)[1:-2]
    return np.all((growth >= 0.9) & (growth <= 1.3))

def check_absolute_growth(cell_cycle):
    "checks the absolute growth of a cell cycle."
    cell_opl = cell_cycle['integrated_opl']
    abs_growth = cell_opl[-2]/cell_opl[1]
    if abs_growth < 1.5 or 3 < abs_growth:
        return False
    else:
        return True

def check_movement(cell_cycle):
    "checks if movemnts are not too big, indicating tracking problems"
    midpoints = np.mean([cell_cycle['new_pole'],cell_cycle['old_pole']], axis=0)
    midpoint_shifts = np.roll(midpoints,-1, axis=0)-midpoints
    distance = np.sqrt(np.sum(midpoint_shifts**2, axis=1))[1:-2]
    if np.all(distance < 10):
        return True
    else:
        return False

def display_frames_with_plot(images, values):
    """
    Display frames alongside their corresponding function values in an interactive plot.
    
    Parameters:
        - images (list): A list of images (each image as a numpy array)
        - values (list): A list of corresponding function values for each image
    """
    
    current_index = 0
    running = True

    # Function to update the displayed image and plot
    def update_display():
        nonlocal current_index
        
        # Clear previous images/lines
        ax_image.imshow(images[current_index])
        ax_image.axis('off')
        
        ax_plot.cla()
        ax_plot.plot(values, '-o', label='Function values')
        ax_plot.plot(current_index, values[current_index], 'ro', label='Current Value')
        ax_plot.legend()
        ax_plot.set_xlim(0, len(values)-1)
        ax_plot.set_ylim(min(values), max(values))
        
        fig.canvas.draw()

    # Functions to handle keyboard inputs
    def on_key_press(event):
        nonlocal current_index, running

        if event.key == 'right' and current_index < len(images)-1:
            current_index += 1
            update_display()
        elif event.key == 'left' and current_index > 0:
            current_index -= 1
            update_display()
        elif event.key in ['k', 'space']:
            running = not running

    def automatic_update(frame):
        nonlocal current_index, running
        if running:
            current_index += 1
            if current_index >= len(images):
                current_index = 0
            update_display()
        return []

    # Create the figure and subplots
    fig, (ax_image, ax_plot) = plt.subplots(1, 2, figsize=(10, 5))
    
    # Initial display
    update_display()
    
    fig.canvas.mpl_connect('key_press_event', on_key_press)
    plt.tight_layout()
    plt.show()

def display_image_series(image_series_list, values_list):
    """
    Display multiple image series in a 3x3 grid alongside their corresponding function values.
    
    Parameters:
        - image_series_list (list of lists): Each inner list contains images.
        - values_list (list of lists): Each inner list contains corresponding function values.
    """
    if len(image_series_list) != 9 or len(values_list) != 9:
        raise ValueError("Both image_series_list and values_list should have exactly 9 items for a 3x3 grid.")

    fig = plt.figure(figsize=(15, 15))
    grid = plt.GridSpec(3, 6, wspace=0.2, hspace=0.2)

    current_indices = [0] * 9  # Indices for the current frame of each subplot

    ax_images = []
    ax_plots = []
    lines = []
    points = []

    for idx in range(9):
        i, j = divmod(idx, 3)
        ax_img = fig.add_subplot(grid[i, 2*j])
        ax_plot = fig.add_subplot(grid[i, 2*j+1])
        
        ax_img.axis('off')
        line, = ax_plot.plot([], [], '-o')
        point, = ax_plot.plot([], [], 'ro')
        ax_plot.axis('off')

        ax_images.append(ax_img)
        ax_plots.append(ax_plot)
        lines.append(line)
        points.append(point)

    def update_display():
        for idx, (images, values, ax_image, line, point) in enumerate(zip(image_series_list, values_list, ax_images, lines, points)):
            ax_image.imshow(images[current_indices[idx]])
            
            line.set_data(range(len(values)), values)
            point.set_data(current_indices[idx], values[current_indices[idx]])
            ax_plots[idx].set_xlim(0, len(values)-1)
            ax_plots[idx].set_ylim(min(values), max(values))

    def on_key_press(event):
        if event.key == 'right':
            current_indices[:] = [(x + 1) % len(image_series_list[i]) for i, x in enumerate(current_indices)]
        elif event.key == 'left':
            current_indices[:] = [(x - 1) % len(image_series_list[i]) for i, x in enumerate(current_indices)]
        update_display()
        fig.canvas.draw_idle()

    update_display()
    fig.canvas.mpl_connect('key_press_event', on_key_press)
    plt.tight_layout()
    plt.show()

def get_cell_cycle_frames(cell_cycle, img_stack, frame_size: int = 80):
    "returns a list of the with the cut out frames of the cell cycle."
    frames = []
    mid_point = np.mean([cell_cycle['new_pole'][1],cell_cycle['old_pole'][1]], axis=0).astype(np.uint16)
    for i in range(len(cell_cycle['frames'])):
        frame_nr = cell_cycle['frames'][i]
        frame = np.repeat(img_stack[frame_nr][:, :, np.newaxis], 3, axis=-1)
        frame = cv2.drawContours(frame, [cell_cycle['contours'][i]], 0, (1.0,0,0), thickness=1)
        frame = frame[mid_point[0]-frame_size//2:mid_point[0]+frame_size//2, mid_point[1]-frame_size//2:mid_point[1]+frame_size//2]
        frames.append(frame)
    return frames
    
def get_cell_cycles(pos, with_contours: bool = True, only_valid: bool = True):
    """calculates all cell cycles of a DeLTA time-lapse position. 
    First cells are split into possible cell cycle, including last mother cell timestep, 
    then unplausable cell cycles are fitered out."""
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
        for cell_cycle in cell_cycles:
            if check_splits(cell_cycle) and ckeck_growth_steps(cell_cycle) and check_absolute_growth(cell_cycle) and check_movement(cell_cycle):
                valid_cell_cycles.append(cell_cycle)
        return valid_cell_cycles