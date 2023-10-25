# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 12:04:27 2023

@author: SWW-Bc20
"""
import os
os.chdir(os.path.dirname(__file__))
import binkoala
import tifffile
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

def normalize(image):
    return (image-image.min())/(image.max()-image.min())

class ImageViewer:
    def __init__(self, image_stack):
        self.image_stack = image_stack
        self.current_index = 0

        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(bottom=0.2)

        self.img = self.ax.imshow(self.image_stack[self.current_index], cmap='gray')
        self.ax.set_title(f'Image {self.current_index + 1}/{len(self.image_stack)}')

        axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
        axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
        self.bnext = Button(axnext, 'Next')
        self.bnext.on_clicked(self.next_image)
        self.bprev = Button(axprev, 'Previous')
        self.bprev.on_clicked(self.prev_image)

        self.connect_key_events()

    def connect_key_events(self):
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

    def on_key(self, event):
        if event.key == 'right':
            self.next_image(None)
        elif event.key == 'left':
            self.prev_image(None)

    def next_image(self, event):
        self.current_index = (self.current_index + 1) % len(self.image_stack)
        self.update_image()

    def prev_image(self, event):
        self.current_index = (self.current_index - 1) % len(self.image_stack)
        self.update_image()

    def update_image(self):
        self.img.set_data(self.image_stack[self.current_index])
        self.ax.set_title(f'Image {self.current_index + 1}/{len(self.image_stack)}')
        self.fig.canvas.draw()

base_path = r'C:\Users\SWW-Bc20\Documents\GitHub\Imaging-pipeline-for-DHM\data\brightfield\20230905-1643'
ph_base_path = base_path + os.sep + '20230905-1643 phase averages'
aligned_base_path = base_path + os.sep + 'aligned_images'
fixed_zr_base_path = base_path + os.sep + 'aligned_images_fixed_zr'
postitions = os.listdir(ph_base_path)

pos = 0
include_in_stack = ['aligned', 'fixed_zr'] # 'ph', 'aligned', 'fixed_zr'

ph_pos_path = ph_base_path + os.sep + postitions[pos]
aligned_pos_path = aligned_base_path + os.sep + postitions[pos]
fixed_zr_pos_path = fixed_zr_base_path + os.sep + postitions[pos]

image_stack = []
timesteps = [int(''.join(filter(str.isdigit, s))) for s in os.listdir(ph_pos_path)]
for timestep in timesteps:
    ph_fname = ph_pos_path + os.sep + f'ph_timestep_{str(timestep).zfill(5)}.bin'
    aligned_fname = aligned_pos_path + os.sep + f'{str(timestep).zfill(5)}_BF.tif'
    fixed_zr_fname =  fixed_zr_pos_path + os.sep + f'{str(timestep).zfill(5)}_BF.tif'
    
    if 'ph' in include_in_stack:
        ph, _ = binkoala.read_mat_bin(ph_fname)
        image_stack.append(normalize(ph).copy())
    
    if 'aligned' in include_in_stack:
        aligned = tifffile.imread(aligned_fname)
        image_stack.append(normalize(aligned).copy())
    
    if 'fixed_zr' in include_in_stack:
        fixed_zr = tifffile.imread(fixed_zr_fname)
        image_stack.append(normalize(fixed_zr).copy())

viewer = ImageViewer(image_stack)
plt.show()