# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 21:55:40 2023

@author: SWW-Bc20
"""
import os
os.chdir(r'C:\Users\SWW-Bc20\Documents\GitHub\Imaging-pipeline-for-DHM\src\delta')
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
start = 10
for i in range(start, start+9):
    image_series_list.append(get_cell_cycle_frames(cell_cycles[i], img_stack))
    values_list.append(cell_cycles[i]['mass'])
display_image_series(image_series_list, values_list)

#%%

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

def display_frames_with_plot(images, values):
    current_index = -2
    running = True

    # Create the figure and subplots
    fig, (ax_image, ax_plot) = plt.subplots(1, 2, figsize=(10, 5))
    
    # Clear previous images/lines
    ax_image.imshow(images[current_index])
    ax_image.axis('off')

    ax_plot.cla()
    ax_plot.plot(np.arange(len(values))*2,values, '-o', label='Function values', linewidth=2, markersize=8)
    ax_plot.plot(current_index*2, values[current_index], 'ro', label='Current Value', markersize=10)
    ax_plot.legend(fontsize=14)
    ax_plot.set_ylim(min(values), max(values))
    ax_plot.set_xlabel('time [min]', fontsize=14)
    ax_plot.set_ylabel('mass [fg]', fontsize=14)
    ax_plot.tick_params(axis='both', which='both', labelsize=12)
    ax_plot.grid(True)

    def update_display():
        nonlocal current_index
    
        # Clear previous images/lines
        ax_image.imshow(images[current_index])
        ax_image.axis('off')
    
        ax_plot.cla()
        ax_plot.plot(np.arange(len(values))*2,values, '-o', label='Function values', linewidth=2, markersize=8)
        ax_plot.plot(current_index*2, values[current_index], 'ro', label='Current Value', markersize=10)
        ax_plot.legend(fontsize=14)
        ax_plot.set_ylim(min(values), max(values))
        ax_plot.set_xlabel('time [min]', fontsize=14)
        ax_plot.set_ylabel('mass [fg]', fontsize=14)
        ax_plot.tick_params(axis='both', which='both', labelsize=12)
        ax_plot.grid(True)
        
        fig.canvas.draw()

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
            if current_index < len(images) - 1:
                current_index += 1
                update_display()
            else:
                running = False  # Stop the animation when reaching the end
                ani.event_source.stop()  # Stop the animation loop
        return []
    
    def on_close(event):
        # This function is called when the plot window is closed
        ani.event_source.stop()  # Stop the animation loop
        plt.close()  # Close the plot window
    
    fig.canvas.mpl_connect('key_press_event', on_key_press)
    fig.canvas.mpl_connect('close_event', on_close) 

    update_display()  # Display the initial frame before animation starts
    ani = animation.FuncAnimation(fig, automatic_update, interval=600, frames=range(0, len(images)))
    plt.show()
    
    # Save animation as a video
    ani.save(r'C:\Users\SWW-Bc20\Documents\GitHub\Imaging-pipeline-for-DHM\Graphes_Images\Graphes_Images\cell_cycles\single_cell.mp4', writer='ffmpeg')  # Uncomment this line to save the animation as a video


nr = 5
frames = get_cell_cycle_frames(cell_cycles[nr], img_stack)
values = np.array(cell_cycles[nr]['mass'])*1000
display_frames_with_plot(frames, values)

#%%
def display_image_series(image_series_list, values_list):
    """
    Display multiple image series in a 3x3 grid alongside their corresponding function values.
    
    Parameters:
        - image_series_list (list of lists): Each inner list contains images.
        - values_list (list of lists): Each inner list contains corresponding function values.
    """
    max_length = 30
    f_n = 0
    running = True
    
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
    
    def automatic_update(frame):
        nonlocal current_indices, running, f_n
        if running:
            if f_n < max_length - 1:
                f_n += 1
                current_indices[:] = [(x + 1) % len(image_series_list[i]) for i, x in enumerate(current_indices)]
                update_display()
            else:
                running = False  # Stop the animation when reaching the end
                ani.event_source.stop()  # Stop the animation loop
        return []
    
    def on_close(event):
        # This function is called when the plot window is closed
        ani.event_source.stop()  # Stop the animation loop
        plt.close()  # Close the plot window
        
    fig.canvas.mpl_connect('key_press_event', on_key_press)
    fig.canvas.mpl_connect('close_event', on_close) 

    update_display()  # Display the initial frame before animation starts
    ani = animation.FuncAnimation(fig, automatic_update, interval=500, save_count=max_length)
    plt.show()
    
    # Save animation as a video
    ani.save(r'C:\Users\SWW-Bc20\Documents\GitHub\Imaging-pipeline-for-DHM\Graphes_Images\Graphes_Images\cell_cycles\multiple_cells.mp4', writer='ffmpeg')  # Uncomment this line to save the animation as a video

image_series_list = []
values_list = []
start = 10
for i in range(start, start+9):
    image_series_list.append(get_cell_cycle_frames(cell_cycles[i], img_stack))
    values_list.append(cell_cycles[i]['mass'])
display_image_series(image_series_list, values_list)

#%%
def display_image_series_six(image_series_list, values_list):
    """
    Display multiple image series in a 3x3 grid alongside their corresponding function values.
    
    Parameters:
        - image_series_list (list of lists): Each inner list contains images.
        - values_list (list of lists): Each inner list contains corresponding function values.
    """
    max_length = 30
    f_n = 0
    running = True
    
    if len(image_series_list) != 6 or len(values_list) != 6:
        raise ValueError("Both image_series_list and values_list should have exactly 9 items for a 3x3 grid.")

    fig = plt.figure(figsize=(15, 10))
    grid = plt.GridSpec(2, 6, wspace=0.2, hspace=0.2)

    current_indices = [0] * 6  # Indices for the current frame of each subplot

    ax_images = []
    ax_plots = []
    lines = []
    points = []

    for idx in range(6):
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
    
    def automatic_update(frame):
        nonlocal current_indices, running, f_n
        if running:
            if f_n < max_length - 1:
                f_n += 1
                current_indices[:] = [(x + 1) % len(image_series_list[i]) for i, x in enumerate(current_indices)]
                update_display()
            else:
                running = False  # Stop the animation when reaching the end
                ani.event_source.stop()  # Stop the animation loop
        return []
    
    def on_close(event):
        # This function is called when the plot window is closed
        ani.event_source.stop()  # Stop the animation loop
        plt.close()  # Close the plot window
        
    fig.canvas.mpl_connect('key_press_event', on_key_press)
    fig.canvas.mpl_connect('close_event', on_close) 

    update_display()  # Display the initial frame before animation starts
    ani = animation.FuncAnimation(fig, automatic_update, interval=500, save_count=max_length)
    plt.show()
    
    # Save animation as a video
    ani.save(r'C:\Users\SWW-Bc20\Documents\GitHub\Imaging-pipeline-for-DHM\Graphes_Images\Graphes_Images\cell_cycles\multiple_cells_six.mp4', writer='ffmpeg')  # Uncomment this line to save the animation as a video

image_series_list = []
values_list = []
start = 11
for i in range(start, start+6):
    image_series_list.append(get_cell_cycle_frames(cell_cycles[i], img_stack))
    values_list.append(cell_cycles[i]['mass'])
display_image_series_six(image_series_list, values_list)

